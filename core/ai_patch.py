from __future__ import annotations 

"""
AI JPEG Patch Reconstruction modülü.

Bu modül:
- Klasik onarımın ürettiği çıktı listesinden en iyi çıktıyı seçer.
- repair_engine.compute_damage_heatmap() ile hasar ısı haritası çıkarır.
- Bu ısı haritasından piksel seviyesinde bir maske üretir.
- İsteğe bağlı olarak:
    * Real-ESRGAN ile süper çözünürlük & detay geri getirme
    * GFPGAN ile yüz onarımı
    * Stable Diffusion Inpainting ile bozuk blokları doldurma
- Ortaya çıkan sonucu *_ai_patch.jpg adıyla kaydeder.

GÜNCEL STRATEJİ (FAZ 1 SON HALİ):
    - Hasar oranı düşükse (low_damage_ratio):
        -> Sadece Real-ESRGAN çalıştırılır (GFPGAN + Inpainting kapalı).
    - Hasar oranı orta seviyedeyse:
        -> Real-ESRGAN + (isteğe bağlı) GFPGAN,
           Inpainting sadece kullanıcı açmışsa ve anlamlı maske varsa.
    - Hasar oranı yüksekse (high_damage_ratio):
        -> Real-ESRGAN + (isteğe bağlı) GFPGAN + Stable Diffusion Inpainting
           mutlaka devreye alınır (maske anlamlıysa).

NOT:
    - Aşağıdaki paketler isteğe bağlıdır; yoksa ilgili adım otomatik atlanır:
        pip install torch diffusers realesrgan gfpgan opencv-python numpy
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter

from .repair_engine import (
    compute_damage_heatmap,
    summarize_heatmap,
    pick_best_output,
)

LogFunc = Callable[[str, Optional[str]], None]

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

# Global model cache
_REAL_ESRGAN_MODEL = None
_GFPGANER = None
_SD_INPAINT_PIPE = None

# Hasar oranı bantları (blokların threshold üstü oranı)
# Örn: high_ratio < 0.05  -> düşük hasar
#      0.05–0.25          -> orta hasar
#      >=0.25             -> yüksek hasar
LOW_DAMAGE_RATIO = 0.05
HIGH_DAMAGE_RATIO = 0.25


# =========================================================
# Yardımcı log
# =========================================================

def _safe_log(log: Optional[LogFunc], msg: str, color: Optional[str] = None) -> None:
    if log is not None:
        try:
            log(msg, color)
        except Exception:
            pass


# =========================================================
# Heatmap -> Maske
# =========================================================

def _build_damage_mask_from_heatmap(
    hm: Dict[str, Any],
    image_size: Tuple[int, int],
    threshold: float = 0.7,
    expand_kernel: int = 3,
) -> Image.Image:
    """
    compute_damage_heatmap() çıktısını kullanarak piksel seviyesinde L (0..255)
    maske üretir. threshold üstü bloklar 255 (beyaz), diğerleri 0 (siyah).
    """
    w, h = image_size
    block_size = int(hm.get("block_size", 16)) or 16
    rows = int(hm.get("rows", 0))
    cols = int(hm.get("cols", 0))
    values = hm.get("values") or []

    mask = Image.new("L", (w, h), 0)
    px = mask.load()

    for r in range(rows):
        if r >= len(values):
            break
        row_vals = values[r]
        for c in range(cols):
            if c >= len(row_vals):
                break
            try:
                v = float(row_vals[c])
            except Exception:
                continue
            if v < threshold:
                continue

            x0 = c * block_size
            y0 = r * block_size
            x1 = min(x0 + block_size, w)
            y1 = min(y0 + block_size, h)

            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    px[xx, yy] = 255

    if expand_kernel > 1:
        try:
            mask = mask.filter(ImageFilter.MaxFilter(size=expand_kernel))
        except Exception:
            # Pillow bazı sürümlerde MaxFilter size kısıtı atabilir, hata olursa olduğu gibi bırak
            pass

    return mask


# =========================================================
# Real-ESRGAN
# =========================================================

def _load_realesrgan_model(log: Optional[LogFunc]):
    """
    Real-ESRGAN modelini lazy şekilde yükler.

    Gerekli paketler:
        pip install realesrgan torch
    """
    global _REAL_ESRGAN_MODEL
    if _REAL_ESRGAN_MODEL is not None:
        return _REAL_ESRGAN_MODEL

    try:
        import torch  # type: ignore
        from realesrgan import RealESRGAN  # type: ignore
    except Exception as exc:
        _safe_log(
            log,
            f"[AI] Real-ESRGAN yüklenemedi (paket yok veya hata): {exc}",
            "orange",
        )
        return None

    # CUDA uyarısı
    if not torch.cuda.is_available():
        _safe_log(
            log,
            "[AI] CUDA bulunamadı, Real-ESRGAN CPU üzerinde çok yavaş çalışabilir.",
            "orange",
        )

    try:
        from pathlib import Path as _Path

        models_dir = _Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        models_path = models_dir / "RealESRGAN_x4plus.pth"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RealESRGAN(device, scale=4)
        # Ağırlıklar yoksa otomatik indirir (models/ dizinine)
        model.load_models(str(models_path), download=True)
        _REAL_ESRGAN_MODEL = model
        _safe_log(log, f"[AI] Real-ESRGAN yüklendi (device={device}, models={models_path})", "green")
        return model
    except Exception as exc:
        _safe_log(
            log,
            f"[AI] Real-ESRGAN modeli başlatılamadı: {exc}",
            "orange",
        )
        return None


def _apply_realesrgan(
    image: Image.Image,
    log: Optional[LogFunc],
) -> Optional[Image.Image]:
    model = _load_realesrgan_model(log)
    if model is None:
        return None

    try:
        out = model.predict(image)
        _safe_log(log, "[AI] Real-ESRGAN süper çözünürlük uygulandı.", "green")
        return out
    except Exception as exc:
        _safe_log(
            log,
            f"[AI] Real-ESRGAN işlem hatası, adım atlanıyor: {exc}",
            "orange",
        )
        return None


# =========================================================
# GFPGAN (yüz onarımı)
# =========================================================

def _load_gfpgan_restorer(log: Optional[LogFunc]):
    """
    GFPGAN modelini lazy şekilde yükler.

    Gerekli paketler:
        pip install gfpgan opencv-python numpy
    """
    global _GFPGANER
    if _GFPGANER is not None:
        return _GFPGANER

    if np is None:
        _safe_log(
            log,
            "[AI] numpy bulunamadı, GFPGAN yüz onarımı devre dışı.",
            "orange",
        )
        return None

    try:
        import cv2  # type: ignore  # noqa: F401
        from gfpgan import GFPGANer  # type: ignore
    except Exception as exc:
        _safe_log(
            log,
            f"[AI] GFPGAN yüklenemedi (paket yok veya hata): {exc}",
            "orange",
        )
        return None

    try:
        from pathlib import Path as _Path

        models_dir = _Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(models_dir / "GFPGANv1.4.pth")

        if not _Path(model_path).is_file():
            _safe_log(
                log,
                f"[AI] GFPGAN ağırlıkları bulunamadı: {model_path}.\n"
                f"    -> Dosyayı manuel olarak bu konuma yerleştirmeniz gerekebilir.",
                "orange",
            )

        restorer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        _GFPGANER = restorer
        _safe_log(log, f"[AI] GFPGAN yüz onarım modeli yüklendi (models={model_path}).", "green")
        return restorer
    except Exception as exc:
        _safe_log(
            log,
            f"[AI] GFPGAN modeli başlatılamadı: {exc}",
            "orange",
        )
        return None


def _apply_gfpgan(
    image: Image.Image,
    log: Optional[LogFunc],
) -> Optional[Image.Image]:
    restorer = _load_gfpgan_restorer(log)
    if restorer is None or np is None:
        return None

    try:
        np_img = np.array(image.convert("RGB"))  # type: ignore
        _, _, restored = restorer.enhance(
            np_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        out = Image.fromarray(restored)
        _safe_log(log, "[AI] GFPGAN yüz onarımı uygulandı.", "green")
        return out
    except Exception as exc:
        _safe_log(
            log,
            f"[AI] GFPGAN işlem hatası, adım atlanıyor: {exc}",
            "orange",
        )
        return None


# =========================================================
# Stable Diffusion Inpainting
# =========================================================

def _load_sd_inpaint_pipeline(log: Optional[LogFunc]):
    """
    Stable Diffusion Inpainting pipeline'ını lazy şekilde yükler.

    Gerekli paketler:
        pip install diffusers torch
    """
    global _SD_INPAINT_PIPE
    if _SD_INPAINT_PIPE is not None:
        return _SD_INPAINT_PIPE

    try:
        import torch  # type: ignore
        from diffusers import StableDiffusionInpaintPipeline  # type: ignore
    except Exception as exc:
        _safe_log(
            log,
            f"[AI] Stable Diffusion Inpainting yüklenemedi (paket yok veya hata): {exc}",
            "orange",
        )
        return None

    # CUDA uyarısı
    if not torch.cuda.is_available():
        _safe_log(
            log,
            "[AI] CUDA bulunamadı, Stable Diffusion Inpainting CPU üzerinde çok yavaş çalışacaktır.",
            "orange",
        )

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        _SD_INPAINT_PIPE = pipe
        _safe_log(
            log,
            f"[AI] Stable Diffusion Inpainting pipeline yüklendi (device={device})",
            "green",
        )
        return pipe
    except Exception as exc:
        _safe_log(
            log,
            f"[AI] Stable Diffusion Inpainting başlatılamadı: {exc}",
            "orange",
        )
        return None


def _apply_sd_inpaint(
    image: Image.Image,
    mask: Image.Image,
    log: Optional[LogFunc],
    prompt: str = "bozuk fotoğraf onarımı, gerçekçi, fotoğraf kalitesinde",
    negative_prompt: str = "aşırı stilize, çizgi film, karikatür, bozulma",
) -> Optional[Image.Image]:
    pipe = _load_sd_inpaint_pipeline(log)
    if pipe is None:
        return None

    try:
        init_image = image.convert("RGB")
        mask_image = mask.convert("L")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=30,
            guidance_scale=7.5,
        )
        out = result.images[0]
        _safe_log(log, "[AI] Stable Diffusion inpainting uygulandı.", "green")
        return out
    except Exception as exc:
        _safe_log(
            log,
            f"[AI] Stable Diffusion inpainting işlem hatası, adım atlanıyor: {exc}",
            "orange",
        )
        return None


# =========================================================
# Ana fonksiyonlar
# =========================================================

def run_ai_patch_reconstruction(
    base_image: Path,
    output_dir: Optional[Path] = None,
    use_realesrgan: bool = True,
    use_gfpgan: bool = True,
    use_inpaint: bool = True,
    damage_threshold: float = 0.7,
    strategy_mode: str = "NORMAL",  # Şimdilik sadece imza için; ileride kullanılabilir
    log: Optional[LogFunc] = None,
) -> Optional[Path]:
    """
    Tek bir resim için AI tabanlı JPEG Patch Reconstruction uygular.

    - base_image: Klasik onarımdan gelen en iyi JPEG (veya orijinal).
    - output_dir: Çıkış klasörü; None ise base_image.parent kullanılır.
    - use_* bayrakları: Hangi AI adımlarının kullanılacağı.
    - damage_threshold: 0-1 arası; ısı haritası için hasarlı blok eşiği.

    Hasar oranına göre strateji:
        - high_ratio < LOW_DAMAGE_RATIO:
              -> Sadece Real-ESRGAN çalışır, GFPGAN + Inpainting kapalı.
        - LOW_DAMAGE_RATIO <= high_ratio < HIGH_DAMAGE_RATIO:
              -> Real-ESRGAN + (isteğe bağlı) GFPGAN,
                 Inpainting sadece kullanıcı açmışsa ve maske anlamlıysa.
        - high_ratio >= HIGH_DAMAGE_RATIO:
              -> Real-ESRGAN + (isteğe bağlı) GFPGAN + Inpainting devrede.
    """
    base_image = Path(base_image)
    if not base_image.is_file():
        _safe_log(
            log,
            f"[AI] AI patch için temel resim bulunamadı: {base_image}",
            "red",
        )
        return None

    try:
        with Image.open(base_image) as im:
            im.load()
            orig_size = im.size
            work_img = im.convert("RGB")
    except Exception as exc:
        _safe_log(
            log,
            f"[AI] Temel resim açılamadı, AI patch atlanıyor: {exc}",
            "red",
        )
        return None

    # Hasar ısı haritası + özet
    hm = compute_damage_heatmap(base_image)
    hm_summary = summarize_heatmap(hm, threshold=damage_threshold)
    high_ratio = float(hm_summary.get("high_ratio", 0.0))

    _safe_log(
        log,
        f"[AI] Hasar ısı haritası: toplam blok={hm_summary.get('total_blocks')}, "
        f"hasarlı blok={hm_summary.get('high_blocks')} (oran={high_ratio:.3f})",
        "blue",
    )

    # Hasar seviyesini belirle
    if high_ratio < LOW_DAMAGE_RATIO:
        damage_level = "LOW"
        _safe_log(
            log,
            f"[AI] Hasar seviyesi: DÜŞÜK (ratio={high_ratio:.3f} < {LOW_DAMAGE_RATIO:.2f}). "
            f"Sadece Real-ESRGAN uygulanacak.",
            "orange",
        )
    elif high_ratio >= HIGH_DAMAGE_RATIO:
        damage_level = "HIGH"
        _safe_log(
            log,
            f"[AI] Hasar seviyesi: YÜKSEK (ratio={high_ratio:.3f} >= {HIGH_DAMAGE_RATIO:.2f}). "
            f"Real-ESRGAN + (varsa) GFPGAN + Inpainting devrede.",
            "orange",
        )
    else:
        damage_level = "MEDIUM"
        _safe_log(
            log,
            f"[AI] Hasar seviyesi: ORTA (ratio={high_ratio:.3f}). "
            f"Real-ESRGAN + (varsa) GFPGAN, Inpainting opsiyonel.",
            "orange",
        )

    # 1) Real-ESRGAN (tüm hasar seviyelerinde isteğe bağlı çalışır)
    if use_realesrgan:
        out = _apply_realesrgan(work_img, log)
        if out is not None:
            work_img = out
    else:
        _safe_log(log, "[AI] Real-ESRGAN devre dışı bırakılmış.", "orange")

    # 2) GFPGAN (sadece ORTA & YÜKSEK hasar seviyelerinde)
    if damage_level == "LOW":
        if use_gfpgan:
            _safe_log(
                log,
                "[AI] Hasar düşük olduğu için GFPGAN yüz onarımı atlandı (yalnızca Real-ESRGAN kullanıldı).",
                "orange",
            )
    else:
        if use_gfpgan:
            out = _apply_gfpgan(work_img, log)
            if out is not None:
                work_img = out
        else:
            _safe_log(log, "[AI] GFPGAN devre dışı bırakılmış.", "orange")

    # 3) Stable Diffusion Inpainting (hasar seviyesi ve kullanıcı tercihine göre)
    if not use_inpaint:
        _safe_log(log, "[AI] Inpainting (Stable Diffusion) kullanıcı tarafından kapatılmış.", "orange")
    else:
        if damage_level == "LOW":
            _safe_log(
                log,
                "[AI] Hasar oranı düşük olduğu için Stable Diffusion inpainting devre dışı bırakıldı.",
                "orange",
            )
        else:
            # ORTA & YÜKSEK hasarda maske üret
            mask = _build_damage_mask_from_heatmap(
                hm,
                image_size=work_img.size,
                threshold=damage_threshold,
                expand_kernel=3,
            )
            if mask.getbbox() is None:
                _safe_log(
                    log,
                    "[AI] Isı haritasına göre anlamlı maske yok, inpainting atlanıyor.",
                    "orange",
                )
            else:
                out = _apply_sd_inpaint(work_img, mask, log)
                if out is not None:
                    work_img = out

    # (Real-ESRGAN / Inpaint sonrası büyümüş ise) tekrar orijinal boyuta ölçekle
    if work_img.size != orig_size:
        work_img = work_img.resize(orig_size, Image.LANCZOS)

    out_dir = output_dir or base_image.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_image.stem}_ai_patch{base_image.suffix}"

    try:
        work_img.save(out_path, quality=95, optimize=True)
        _safe_log(
            log,
            f"[AI] AI JPEG Patch Reconstruction tamamlandı -> {out_path.name}",
            "green",
        )
        return out_path
    except Exception as exc:
        _safe_log(
            log,
            f"[AI] AI çıktı kaydedilemedi: {exc}",
            "red",
        )
        return None


def apply_ai_reconstruction_to_outputs(
    input_path: Path,
    outputs: List[Path],
    output_dir: Optional[Path] = None,
    use_realesrgan: bool = True,
    use_gfpgan: bool = True,
    use_inpaint: bool = True,
    damage_threshold: float = 0.7,
    strategy_mode: str = "NORMAL",
    log: Optional[LogFunc] = None,
) -> List[Path]:
    """
    repair_engine.repair_image_all_methods() tarafından üretilen outputs listesini alır,
    en iyi klasik çıktıyı seçer, AI patch uygular ve yeni çıktıyı listeye ekleyip geri döner.
    """
    if not outputs:
        _safe_log(
            log,
            f"[AI] {input_path.name} için klasik onarım çıktısı yok, AI patch uygulanmadı.",
            "orange",
        )
        return outputs

    best_path = pick_best_output(outputs, strategy_mode=strategy_mode)
    if not best_path:
        _safe_log(
            log,
            f"[AI] {input_path.name} için en iyi klasik çıktı seçilemedi, AI patch atlandı.",
            "orange",
        )
        return outputs

    _safe_log(
        log,
        f"[AI] {input_path.name} için en iyi klasik çıktı: {best_path.name}",
        "blue",
    )

    out_ai = run_ai_patch_reconstruction(
        base_image=best_path,
        output_dir=output_dir,
        use_realesrgan=use_realesrgan,
        use_gfpgan=use_gfpgan,
        use_inpaint=use_inpaint,
        damage_threshold=damage_threshold,
        strategy_mode=strategy_mode,
        log=log,
    )
    if out_ai is not None:
        outputs.append(out_ai)

    return outputs