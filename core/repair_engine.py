from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple

from PIL import Image

from .jpeg_repair import (
    extract_all_jpegs_from_blob,
    partial_top_recovery,
    fix_with_jpeg_markers,
    fix_with_smart_header_v3,
    fix_with_exif_thumbnail,
)
from .png_repair import fix_with_png_crc

LogFunc = Callable[..., None]


# =======================================================
# Yardımcılar
# =======================================================

def _normalize_strategy_mode(mode: str) -> str:
    m = (mode or "").upper()
    if m not in ("SAFE", "NORMAL", "AGGRESSIVE"):
        m = "NORMAL"
    return m


# =======================================================
# Temel dönüştürme / onarım yöntemleri
# =======================================================

def fix_with_pillow(input_path: Path, output_dir: Path, log: LogFunc) -> Optional[Path]:
    """
    Pillow ile yeniden kaydetme.
    Bozuk meta veriyi sıyırıp daha temiz bir kopya üretmek için temel yöntem.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = input_path.suffix.lower()
        name = input_path.stem
        out_path = output_dir / f"{name}_fixed_pillow{ext}"

        with Image.open(input_path) as im:
            im.load()
            if im.mode not in ("RGB", "RGBA", "L", "P"):
                im = im.convert("RGB")

            save_params: Dict[str, Any] = {}
            if ext in (".jpg", ".jpeg"):
                save_params["quality"] = 95
                save_params["optimize"] = True

            im.save(out_path, **save_params)

        # Basit doğrulama
        with Image.open(out_path) as im2:
            im2.load()

        log(f"[PILLOW][OK] {out_path.name}", color="green")
        return out_path
    except Exception as e:
        log(f"[PILLOW][ERROR] {input_path.name} -> {e}", color="red")
        try:
            if "out_path" in locals() and out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return None


def fix_with_png_roundtrip(input_path: Path, output_dir: Path, log: LogFunc) -> Optional[Path]:
    """
    PNG roundtrip: resmi Pillow ile açıp her halükarda PNG olarak kaydeder.
    JPEG dahil pek çok formatta bozuk meta veriyi temizlemede işe yarar.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        name = input_path.stem
        out_path = output_dir / f"{name}_fixed_roundtrip.png"

        with Image.open(input_path) as im:
            im.load()
            if im.mode not in ("RGB", "RGBA", "L", "P"):
                im = im.convert("RGBA")

            im.save(out_path, format="PNG", optimize=True)

        with Image.open(out_path) as im2:
            im2.load()

        log(f"[ROUNDTRIP][OK] {out_path.name}", color="green")
        return out_path
    except Exception as e:
        log(f"[ROUNDTRIP][ERROR] {input_path.name} -> {e}", color="red")
        try:
            if "out_path" in locals() and out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return None


def fix_with_ffmpeg_multi(
    input_path: Path,
    output_dir: Path,
    ffmpeg_cmd: str,
    log: LogFunc,
    qscale_list: Optional[List[int]] = None,
    strategy_mode: str = "NORMAL",
) -> Optional[Path]:
    """
    FFmpeg ile yeniden encode denemesi.
    qscale_list içindeki kalite seviyeleri sırasıyla denenir.
    Başarılı çıktılar arasından gelişmiş skorlamayla en iyi aday seçilir.
    Çok gri / çok küçük / bozuk görünenler erken elenir.
    """
    if qscale_list is None:
        qscale_list = [2, 4, 6]

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = input_path.suffix.lower()
        name = input_path.stem

        mode = _normalize_strategy_mode(strategy_mode)

        # Windows'ta konsol penceresini gizlemek için
        creationflags = 0
        try:  # type: ignore[attr-defined]
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        except Exception:
            # not on Windows or flag not available
            creationflags = 0

        candidates: List[Path] = []

        for q in qscale_list:
            out_path = output_dir / f"{name}_fixed_ffmpeg_q{q}{ext}"
            cmd = [
                ffmpeg_cmd,
                "-err_detect",
                "ignore_err",
                "-analyzeduration",
                "10000000",
                "-i",
                str(input_path),
                "-map_metadata",
                "-1",
                "-c:v",
                "mjpeg" if ext in (".jpg", ".jpeg") else "png",
                "-qscale:v",
                str(q),
                "-y",
                str(out_path),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                creationflags=creationflags,
            )

            if not out_path.exists() or out_path.stat().st_size <= 0 or result.returncode != 0:
                err_last = (result.stderr.strip().splitlines() or ["Bilinmeyen hata"])[-1]
                log(f"[FFMPEG][ERROR-RUN] q={q} {input_path.name} -> {err_last}", color="red")
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                continue

            # Çıktıyı hızlıca doğrula + temel kalite metriklerine göre ele
            metrics = evaluate_output(out_path, strategy_mode=mode)
            if not metrics.get("verify"):
                log(f"[FFMPEG][WARN-VERIFY] q={q} {out_path.name} doğrulama başarısız, sonuç elendi.", color="orange")
                try:
                    out_path.unlink()
                except Exception:
                    pass
                continue

            score = float(metrics.get("score") or 0.0)
            if score < 0.15:
                # Gereksiz çöp çıktıları doğrudan sil
                log(f"[FFMPEG][WARN-SCORE] q={q} {out_path.name} skor çok düşük ({score:.3f}), sonuç elendi.", color="orange")
                try:
                    out_path.unlink()
                except Exception:
                    pass
                continue

            candidates.append(out_path)
            log(f"[FFMPEG][OK] q={q} -> {out_path.name} (score={score:.3f})", color="green")

        if not candidates:
            return None

        # En iyi adayı, gelişmiş metriklere göre seç
        best = pick_best_output(candidates, strategy_mode=mode)
        if best:
            log(f"[FFMPEG][BEST] {best.name}", color="darkgreen")
        return best

    except FileNotFoundError:
        log("[FFMPEG][ERROR-NOTFOUND] ffmpeg bulunamadı. Yöntem atlandı.", color="red")
        return None
    except Exception as e:
        log(f"[FFMPEG][ERROR-UNEXPECTED] {input_path.name} -> {e}", color="red")
        return None


# =======================================================
# Gelişmiş kalite metrikleri
# =======================================================

def _prepare_analysis_image(im: Image.Image, strategy_mode: str) -> Image.Image:
    """
    Analiz için kullanılacak kopyayı hazırlar.
    Büyük görselleri downscale eder, SAFE modda daha agresif küçültme yapar.
    """
    mode = _normalize_strategy_mode(strategy_mode)
    w, h = im.size
    max_dim = 1600
    if mode == "SAFE":
        max_dim = 1024

    if max(w, h) > max_dim:
        ratio = max_dim / float(max(w, h))
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        return im.resize((new_w, new_h))
    return im


def _estimate_grayness(im: Image.Image) -> float:
    """Görüntünün tek tonda (gri) olup olmadığını yaklaşık olarak ölçer (0.0-1.0)."""
    gray = im.convert("L")
    hist = gray.histogram()
    total = sum(hist)
    if total <= 0:
        return 1.0
    max_bin = max(hist)
    return max_bin / float(total)


def _estimate_truncation(im: Image.Image, slices: int = 10) -> float:
    """
    Üst ve alt bölgelerdeki parlaklık varyans farkına bakarak truncation tahmini yapar (0.0-1.0).
    JPEG dosyalarda alt kısım bozulmuşsa bu değer yükselir.
    """
    gray = im.convert("L")
    width, height = gray.size
    if height <= 0 or slices <= 1:
        return 0.0
    slice_h = max(1, height // slices)

    def slice_variance(y0: int, y1: int) -> float:
        region = gray.crop((0, y0, width, y1))
        hist = region.histogram()
        total = sum(hist)
        if total <= 0:
            return 0.0
        mean = sum(i * c for i, c in enumerate(hist)) / float(total)
        var = sum(((i - mean) ** 2) * c for i, c in enumerate(hist)) / float(total)
        return var

    top_vars: List[float] = []
    bottom_vars: List[float] = []
    for i in range(slices):
        y0 = i * slice_h
        y1 = height if i == slices - 1 else (i + 1) * slice_h
        v = slice_variance(y0, y1)
        if i < slices // 2:
            top_vars.append(v)
        else:
            bottom_vars.append(v)

    if not top_vars or not bottom_vars:
        return 0.0

    top_avg = sum(top_vars) / len(top_vars)
    bottom_avg = sum(bottom_vars) / len(bottom_vars)
    if top_avg <= 0:
        return 0.0

    ratio = (top_avg - bottom_avg) / float(top_avg)
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        return 1.0
    return ratio


def _estimate_entropy(im: Image.Image) -> float:
    """
    Parlaklık histogramına göre normalize entropi (0.0-1.0).
    Çok düşük entropi: tekdüze / anlamsız görüntü.
    """
    import math

    gray = im.convert("L")
    hist = gray.histogram()
    total = sum(hist)
    if total <= 0:
        return 0.0

    ent = 0.0
    for c in hist:
        if c <= 0:
            continue
        p = c / float(total)
        ent -= p * math.log2(p)

    # Maksimum entropi 8 bit gri için log2(256) = 8
    return max(0.0, min(1.0, ent / 8.0))


def _estimate_sharpness(im: Image.Image) -> float:
    """
    Basit Laplacian benzeri kernel ile keskinlik tahmini.
    Büyük görüntülerde performans için 512x512'ye küçültülür.
    Çıktı normalize edilip 0.0-1.0 aralığına sıkıştırılır.
    """
    import math

    gray = im.convert("L")
    w, h = gray.size
    # Aşırı büyük görselleri downscale et
    max_dim = 512
    if max(w, h) > max_dim:
        ratio = max_dim / float(max(w, h))
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        gray = gray.resize((new_w, new_h))
        w, h = gray.size

    pix = gray.load()
    acc = 0.0
    count = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            c = pix[x, y]
            v = (
                pix[x, y - 1]
                + pix[x - 1, y]
                + pix[x + 1, y]
                + pix[x, y + 1]
                - 4 * c
            )
            acc += v * v
            count += 1

    if count == 0:
        return 0.0

    var_lap = acc / float(count)
    # Normalize etmek için log tabanlı sıkıştırma
    sharp = math.log10(1.0 + var_lap) / 4.0  # 10^4 ~ 10000'e kadar makul
    return max(0.0, min(1.0, sharp))


def evaluate_output(path: Path, strategy_mode: str = "NORMAL") -> Dict[str, Any]:
    """
    Üretilen bir çıktının doğrulanabilirliği ve görsel kalitesi hakkında metrikler döndürür.
    strategy_mode, SAFE modda bazı ağır analizleri devre dışı bırakmak için kullanılır.
    """
    info: Dict[str, Any] = {
        "path": path,
        "exists": path.exists(),
        "size": 0,
        "verify": False,
        "pixels": 0,
        "width": 0,
        "height": 0,
        "mode": None,
        "gray_score": None,
        "truncation_score": None,
        "entropy": None,
        "sharpness": None,
        "score": 0.0,
    }
    if not path.exists():
        return info

    mode = _normalize_strategy_mode(strategy_mode)

    try:
        info["size"] = path.stat().st_size
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            info["width"] = w
            info["height"] = h
            info["pixels"] = w * h
            info["mode"] = im.mode

            # Analiz için downscale edilmiş kopya
            im_an = _prepare_analysis_image(im, mode)

            # SAFE modda ağır analizlerin bir kısmını kapat
            if mode == "SAFE":
                gray_score = _estimate_grayness(im_an)
                trunc_score = None
                entropy = _estimate_entropy(im_an)
                sharp = None
            else:
                gray_score = _estimate_grayness(im_an)
                trunc_score = _estimate_truncation(im_an)
                entropy = _estimate_entropy(im_an)
                sharp = _estimate_sharpness(im_an)

            info["gray_score"] = gray_score
            info["truncation_score"] = trunc_score
            info["entropy"] = entropy
            info["sharpness"] = sharp

            # Temel skor: çözünürlüğe göre 0.4-1.0 arası
            if w <= 0 or h <= 0:
                base = 0.0
            else:
                pixel_factor = min(1.0, (w * h) / float(1920 * 1080))
                base = 0.4 + 0.6 * pixel_factor

            # Çok küçük dosyalar (thumbnail olma ihtimali) için ceza
            size_bytes = info["size"]
            if size_bytes <= 0:
                size_factor = 0.0
            elif size_bytes < 30 * 1024:  # 30 KB altı ise güçlü ceza
                size_factor = 0.5
            else:
                size_factor = 1.0

            # Tek tonda (tamamen gri) görüntüler için ceza
            gray_penalty = 1.0
            if gray_score is not None:
                gray_penalty = 1.0 - min(1.0, gray_score) * 0.4

            # Alt kısım bozuk (truncation) ise ceza
            trunc_penalty = 1.0
            if trunc_score is not None:
                trunc_penalty = 1.0 - max(0.0, trunc_score - 0.3) * 0.5

            # Entropi çok düşükse (anlamsız/gürültüsüz) ceza
            entropy_factor = 1.0
            if entropy is not None:
                if entropy < 0.4:
                    entropy_factor = 0.6
                elif entropy < 0.6:
                    entropy_factor = 0.8

            # Keskinlik çok düşükse ceza
            sharp_factor = 1.0
            if sharp is not None:
                if sharp < 0.1:
                    sharp_factor = 0.6
                elif sharp < 0.2:
                    sharp_factor = 0.8

            score = base * size_factor * gray_penalty * trunc_penalty * entropy_factor * sharp_factor
            info["score"] = max(0.0, min(1.0, float(score)))

        info["verify"] = True
    except Exception:
        info["verify"] = False

    return info


def pick_best_output(paths: List[Path], strategy_mode: str = "NORMAL") -> Optional[Path]:
    """
    En iyi çıktıyı, doğrulanabilir olanlar arasından
    skor + boyut + piksel sayısına göre seçer.
    """
    mode = _normalize_strategy_mode(strategy_mode)
    scored: List[Tuple[Path, float, int, int]] = []
    for p in paths:
        metrics = evaluate_output(p, strategy_mode=mode)
        if not metrics.get("verify"):
            continue
        score = float(metrics.get("score") or 0.0)
        size = int(metrics.get("size") or 0)
        pixels = int(metrics.get("pixels") or 0)
        scored.append((p, score, size, pixels))

    if not scored:
        return None

    scored.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)
    return scored[0][0]


# =======================================================
# JPEG header auto-seçim yardımcıları
# =======================================================

def _parse_jpeg_dimensions_from_bytes(data: bytes) -> Optional[Tuple[int, int, int]]:
    """
    Basit JPEG parser: SOF segmentinden width, height, component sayısını çeker.
    Başarısız olursa None döner.
    """
    try:
        i = 0
        n = len(data)
        if n < 4:
            return None

        # SOI beklenmiyorsa bile devam, ama varsa atla
        if data[0] == 0xFF and data[1] == 0xD8:
            i = 2

        while i + 3 < n:
            # 0xFF doldurma byte'ları atla
            if data[i] != 0xFF:
                i += 1
                continue
            # bir veya daha fazla 0xFF geç
            while i < n and data[i] == 0xFF:
                i += 1
            if i >= n:
                break
            marker = data[i]
            i += 1

            # Uzunluk taşımayan marker'lar
            if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7) or marker == 0x01:
                continue

            if i + 1 >= n:
                break
            seg_len = (data[i] << 8) + data[i + 1]
            if seg_len < 2:
                return None
            seg_start = i + 2
            seg_end = seg_start + seg_len - 2
            if seg_end > n:
                break

            # SOF marker'ları
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                seg = data[seg_start:seg_end]
                if len(seg) < 6:
                    return None
                # seg: [precision(1), height(2), width(2), components(1), ...]
                height = (seg[1] << 8) + seg[2]
                width = (seg[3] << 8) + seg[4]
                components = seg[5]
                return (width, height, components)

            i = seg_end

    except Exception:
        return None

    return None


def _parse_jpeg_dimensions_from_file(path: Path) -> Optional[Tuple[int, int, int]]:
    try:
        with open(path, "rb") as f:
            data = f.read(16 * 1024)  # header için genelde yeterli
        return _parse_jpeg_dimensions_from_bytes(data)
    except Exception:
        return None


def _select_best_headers_for_image(
    input_path: Path,
    header_library: Optional[List[bytes]],
    log: LogFunc,
) -> Optional[List[bytes]]:
    """
    Smart Header V3 için: header_library içinden hedef görüntünün
    genişlik/yükseklik/bileşen sayısına en yakın olanları seçer.
    """
    if not header_library:
        return None

    img_dims = _parse_jpeg_dimensions_from_file(input_path)
    if img_dims is None:
        log("[HEADER][AUTO] Giriş JPEG boyutları okunamadı, tüm header'lar kullanılacak.", color="orange")
        return header_library

    img_w, img_h, img_comp = img_dims
    log(f"[HEADER][AUTO] Giriş JPEG boyutları: {img_w}x{img_h}, comp={img_comp}", color="darkgreen")

    scored: List[Tuple[bytes, int, Optional[Tuple[int, int, int]]]] = []

    for hdr in header_library:
        dims = _parse_jpeg_dimensions_from_bytes(hdr)
        if dims is None:
            # boyut okunamayan header'ı en sona at
            scored.append((hdr, 10_000_000, None))
            continue

        hw, hh, hc = dims
        diff = abs(hw - img_w) + abs(hh - img_h)
        if hc != img_comp:
            diff += 2000  # bileşen farkı için ekstra ceza
        scored.append((hdr, diff, dims))

    if not scored:
        return header_library

    scored.sort(key=lambda t: t[1])
    best_diff = scored[0][1]

    # Bir miktar tolerans payı ile en yakınları al
    tolerance = max(1024, best_diff + 256)
    selected: List[bytes] = [hdr for hdr, diff, _dims in scored if diff <= tolerance]

    log(f"[HEADER][AUTO] Toplam {len(header_library)} header içinden {len(selected)} aday seçildi (best_diff={best_diff}).", color="purple")

    return selected or header_library


# =======================================================
# Hafif teşhis / strateji desteği
# =======================================================

def diagnose_image(input_path: Path, is_jpeg: bool, is_png: bool, log: LogFunc) -> Dict[str, Any]:
    """
    Onarım öncesi hafif bir teşhis yapar, log üretir ve basit bir durum sözlüğü döndürür.
    Bu bilgi, hangi yöntemlerin hangi sırayla deneneceğine karar vermek için kullanılır.
    """
    diag: Dict[str, Any] = {
        "can_open": False,
        "width": None,
        "height": None,
        "mode": None,
        "gray_score": None,
        "truncation_score": None,
        "severity": "unknown",  # "light" | "medium" | "heavy"
    }

    try:
        with Image.open(input_path) as im:
            im.load()
            w, h = im.size
            mode = im.mode
            gray_score = _estimate_grayness(_prepare_analysis_image(im, "NORMAL"))
            trunc_score = _estimate_truncation(_prepare_analysis_image(im, "NORMAL"))

        diag["can_open"] = True
        diag["width"] = w
        diag["height"] = h
        diag["mode"] = mode
        diag["gray_score"] = gray_score
        diag["truncation_score"] = trunc_score

        log(f"[DIAG][OK] Pillow ile açıldı: {w}x{h}, mode={mode}", color="darkgreen")

        # Basit şiddet belirleme
        if gray_score is not None and gray_score >= 0.98:
            severity = "heavy"
            log("[DIAG][SEVERITY] Görüntü neredeyse tek ton (gray_score >= 0.98) – ağır veri kaybı ihtimali yüksek.", color="orange")
        elif is_jpeg and trunc_score is not None and trunc_score >= 0.7:
            severity = "medium"
            log("[DIAG][SEVERITY] Üst/alt bölgeler arasında ciddi varyans farkı (truncation_score yüksek) – alt kısım bozulmuş olabilir.", color="orange")
        else:
            severity = "light"

        diag["severity"] = severity

    except Exception as e:
        diag["can_open"] = False
        diag["severity"] = "heavy"
        log(f"[DIAG][ERROR] Pillow ile ön kontrol BAŞARISIZ: {e}", color="red")
        if is_jpeg:
            log("[DIAG][HINT] JPEG dosya hiç açılamıyor; header/marker kaynaklı ağır bozulma olası.", color="orange")
        elif is_png:
            log("[DIAG][HINT] PNG dosya hiç açılamıyor; header/CRC/IDAT bozulması olası.", color="orange")
        else:
            log("[DIAG][HINT] Dosya açılamıyor; ağır bozulma veya desteklenmeyen format.", color="orange")

    return diag


def _build_step_plan(
    is_jpeg: bool,
    is_png: bool,
    diag: Dict[str, Any],
    strategy_mode: str,
    use_flags: Dict[str, bool],
) -> List[str]:
    """
    Dosya türü + teşhis + strateji moduna göre izlenecek adım planını üretir.
    Adımlar: EMBED_SCAN, PNG_CRC, MARKER, HEADER, PARTIAL, FFMPEG, PILLOW, ROUNDTRIP, EXIF
    """
    severity = diag.get("severity") or "unknown"
    can_open = bool(diag.get("can_open"))

    mode = _normalize_strategy_mode(strategy_mode)
    steps: List[str] = []

    if is_jpeg:
        # JPEG için temel havuz
        if use_flags["use_embed_scan"]:
            steps.append("EMBED_SCAN")

        if mode == "SAFE":
            if not can_open:
                # Hiç açılamıyorsa, en azından marker/header deneyelim
                if use_flags["use_marker"]:
                    steps.append("MARKER")
                if use_flags["use_header"]:
                    steps.append("HEADER")
            # Hafif yöntemler
            if use_flags["use_pillow"]:
                steps.append("PILLOW")
            if use_flags["use_png_roundtrip"]:
                steps.append("ROUNDTRIP")
            if use_flags["use_exif_thumb"]:
                steps.append("EXIF")

        elif mode == "AGGRESSIVE":
            # Daha ağır onarımları öncele
            if use_flags["use_header"]:
                steps.append("HEADER")
            if use_flags["use_marker"]:
                steps.append("MARKER")
            if use_flags["use_partial_top"]:
                steps.append("PARTIAL")
            if use_flags["use_ffmpeg"]:
                steps.append("FFMPEG")
            if use_flags["use_pillow"]:
                steps.append("PILLOW")
            if use_flags["use_png_roundtrip"]:
                steps.append("ROUNDTRIP")
            if use_flags["use_exif_thumb"]:
                steps.append("EXIF")

        else:  # NORMAL
            if severity == "light" and can_open:
                # Önce hafif yöntemler, sonra ağır toplar
                if use_flags["use_pillow"]:
                    steps.append("PILLOW")
                if use_flags["use_png_roundtrip"]:
                    steps.append("ROUNDTRIP")
                if use_flags["use_marker"]:
                    steps.append("MARKER")
                if use_flags["use_header"]:
                    steps.append("HEADER")
                if use_flags["use_partial_top"]:
                    steps.append("PARTIAL")
                if use_flags["use_ffmpeg"]:
                    steps.append("FFMPEG")
            elif severity == "heavy":
                # Ağır bozulma: header/marker/partial öncelikli
                if use_flags["use_header"]:
                    steps.append("HEADER")
                if use_flags["use_marker"]:
                    steps.append("MARKER")
                if use_flags["use_partial_top"]:
                    steps.append("PARTIAL")
                if use_flags["use_ffmpeg"]:
                    steps.append("FFMPEG")
                if use_flags["use_pillow"]:
                    steps.append("PILLOW")
                if use_flags["use_png_roundtrip"]:
                    steps.append("ROUNDTRIP")
            else:
                # Orta seviye: klasik sıraya yakın
                if use_flags["use_marker"]:
                    steps.append("MARKER")
                if use_flags["use_header"]:
                    steps.append("HEADER")
                if use_flags["use_partial_top"]:
                    steps.append("PARTIAL")
                if use_flags["use_ffmpeg"]:
                    steps.append("FFMPEG")
                if use_flags["use_pillow"]:
                    steps.append("PILLOW")
                if use_flags["use_png_roundtrip"]:
                    steps.append("ROUNDTRIP")

            if use_flags["use_exif_thumb"]:
                steps.append("EXIF")

    elif is_png:
        # PNG için
        if use_flags["use_png_crc"]:
            steps.append("PNG_CRC")

        if mode == "SAFE":
            if use_flags["use_pillow"]:
                steps.append("PILLOW")
            if use_flags["use_png_roundtrip"]:
                steps.append("ROUNDTRIP")
        elif mode == "AGGRESSIVE":
            if use_flags["use_png_crc"]:
                steps.append("PNG_CRC")
            if use_flags["use_pillow"]:
                steps.append("PILLOW")
            if use_flags["use_png_roundtrip"]:
                steps.append("ROUNDTRIP")
            if use_flags["use_ffmpeg"]:
                steps.append("FFMPEG")
        else:  # NORMAL
            if use_flags["use_pillow"]:
                steps.append("PILLOW")
            if use_flags["use_png_roundtrip"]:
                steps.append("ROUNDTRIP")
            if use_flags["use_ffmpeg"]:
                steps.append("FFMPEG")

    else:
        # Diğer formatlar
        if mode == "SAFE":
            if use_flags["use_pillow"]:
                steps.append("PILLOW")
        elif mode == "AGGRESSIVE":
            if use_flags["use_pillow"]:
                steps.append("PILLOW")
            if use_flags["use_png_roundtrip"]:
                steps.append("ROUNDTRIP")
            if use_flags["use_ffmpeg"]:
                steps.append("FFMPEG")
        else:  # NORMAL
            if use_flags["use_pillow"]:
                steps.append("PILLOW")
            if use_flags["use_png_roundtrip"]:
                steps.append("ROUNDTRIP")
            if use_flags["use_ffmpeg"]:
                steps.append("FFMPEG")

    # Aynı adımı birden fazla kez eklediysek sadeleştir
    seen = set()
    ordered: List[str] = []
    for s in steps:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


# =======================================================
# Ana onarım fonksiyonu
# =======================================================

def repair_image_all_methods(
    input_path: Path,
    base_output_dir: Path,
    ref_header_bytes: Optional[bytes],
    ffmpeg_cmd: Optional[str],
    use_pillow: bool,
    use_png_roundtrip: bool,
    use_header: bool,
    use_marker: bool,
    use_ffmpeg: bool,
    ffmpeg_qscale_list: List[int],
    stop_on_first_success: bool,
    header_size: int,
    log: LogFunc,
    keep_apps: bool = True,
    keep_com: bool = True,
    header_library: Optional[List[bytes]] = None,
    use_embed_scan: bool = True,
    use_partial_top: bool = True,
    use_exif_thumb: bool = True,
    use_png_crc: bool = True,
    exif_thumb_upscale: bool = False,
    png_crc_skip_ancillary: bool = False,
    strategy_mode: str = "NORMAL",  # SAFE | NORMAL | AGGRESSIVE
) -> List[Path]:
    """
    Tüm onarım yöntemlerini, yapılandırmaya ve dosya türüne göre
    STRATEJİ MODUNA uygun bir sırayla dener.
    Başarılı olan tüm çıktıları döndürür.
    """
    successes: List[Path] = []
    ext = input_path.suffix.lower()
    is_jpeg = ext in (".jpg", ".jpeg")
    is_png = ext == ".png"

    mode = _normalize_strategy_mode(strategy_mode)

    log(f"\n--- {input_path.name} işleniyor ({ext}) ---", color="blue")

    # Hafif teşhis
    diag = diagnose_image(input_path, is_jpeg=is_jpeg, is_png=is_png, log=log)

    # JPEG header auto-seçim (Smart Header V3 için)
    selected_header_library = header_library
    if is_jpeg and use_header and header_library:
        selected_header_library = _select_best_headers_for_image(input_path, header_library, log)

    # Kullanıcı ayarlarını tek yerde topla
    use_flags = {
        "use_pillow": use_pillow,
        "use_png_roundtrip": use_png_roundtrip,
        "use_header": use_header,
        "use_marker": use_marker,
        "use_ffmpeg": use_ffmpeg and bool(ffmpeg_cmd),
        "use_embed_scan": use_embed_scan and is_jpeg,
        "use_partial_top": use_partial_top and is_jpeg,
        "use_exif_thumb": use_exif_thumb and is_jpeg,
        "use_png_crc": use_png_crc and is_png,
    }

    # Adım planını üret
    step_plan = _build_step_plan(
        is_jpeg=is_jpeg,
        is_png=is_png,
        diag=diag,
        strategy_mode=mode,
        use_flags=use_flags,
    )
    log(f"[STRATEGY] Adım planı: {', '.join(step_plan) if step_plan else 'Boş'}", color="purple")

    # Çıktı klasörleri
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Planı sırayla uygula
    for step in step_plan:
        if step == "EMBED_SCAN":
            if not use_flags["use_embed_scan"]:
                continue
            log("[PIPE][EMBED_SCAN] Gömülü JPEG taraması başlatılıyor...", color="blue")
            embeds = extract_all_jpegs_from_blob(input_path, base_output_dir, log, min_size=1024)
            if embeds:
                log(f"[PIPE][EMBED_SCAN][OK] {len(embeds)} adet gömülü JPEG çıkarıldı.", color="green")
            successes.extend(embeds)
            if stop_on_first_success and embeds:
                return successes

        elif step == "PNG_CRC":
            if not use_flags["use_png_crc"]:
                continue
            log("[PIPE][PNG_CRC] PNG CRC onarım denemesi başlatılıyor...", color="blue")
            p_crc = fix_with_png_crc(
                input_path,
                base_output_dir,
                log,
                skip_ancillary_on_crc_error=png_crc_skip_ancillary,
            )
            if p_crc:
                log(f"[PIPE][PNG_CRC][OK] {p_crc.name}", color="green")
                successes.append(p_crc)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][PNG_CRC][FAIL] Onarım başarısız veya değişiklik yok.", color="orange")

        elif step == "MARKER":
            if not use_flags["use_marker"]:
                continue
            log("[PIPE][JPEG_MARKER] JPEG marker onarımı başlatılıyor...", color="blue")
            p_mark = fix_with_jpeg_markers(input_path, base_output_dir, log)
            if p_mark:
                log(f"[PIPE][JPEG_MARKER][OK] {p_mark.name}", color="green")
                successes.append(p_mark)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][JPEG_MARKER][FAIL] Marker ile onarım yapılamadı.", color="orange")

        elif step == "HEADER":
            if not use_flags["use_header"]:
                continue
            log("[PIPE][SMART_HEADER] Smart Header V3 onarımı başlatılıyor...", color="blue")
            p_hdr = fix_with_smart_header_v3(
                input_path,
                base_output_dir,
                ref_header_bytes,
                log,
                header_size=header_size,
                keep_apps=keep_apps,
                keep_com=keep_com,
                header_library=selected_header_library,
            )
            if p_hdr:
                log(f"[PIPE][SMART_HEADER][OK] {p_hdr.name}", color="green")
                successes.append(p_hdr)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][SMART_HEADER][FAIL] Uygun header ile onarım başarısız.", color="orange")

        elif step == "PARTIAL":
            if not use_flags["use_partial_top"]:
                continue
            log("[PIPE][PARTIAL_TOP] Kısmi üst kısım kurtarma başlatılıyor...", color="blue")
            p_parts = partial_top_recovery(input_path, base_output_dir, log)
            if p_parts:
                log(f"[PIPE][PARTIAL_TOP][OK] {len(p_parts)} adet kısmi çıktı üretildi.", color="green")
                successes.extend(p_parts)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][PARTIAL_TOP][FAIL] Kısmi kurtarma başarısız.", color="orange")

        elif step == "FFMPEG":
            if not use_flags["use_ffmpeg"] or not ffmpeg_cmd:
                continue
            log("[PIPE][FFMPEG] FFmpeg yeniden kodlama başlatılıyor...", color="blue")
            p_ff = fix_with_ffmpeg_multi(
                input_path,
                base_output_dir,
                ffmpeg_cmd,
                log,
                qscale_list=ffmpeg_qscale_list,
                strategy_mode=mode,
            )
            if p_ff:
                log(f"[PIPE][FFMPEG][OK] {p_ff.name}", color="green")
                successes.append(p_ff)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][FFMPEG][FAIL] FFmpeg ile anlamlı bir çıktı üretilemedi.", color="orange")

        elif step == "PILLOW":
            if not use_flags["use_pillow"]:
                continue
            log("[PIPE][PILLOW] Pillow yeniden kaydetme başlatılıyor...", color="blue")
            p_pil = fix_with_pillow(input_path, base_output_dir, log)
            if p_pil:
                log(f"[PIPE][PILLOW][OK] {p_pil.name}", color="green")
                successes.append(p_pil)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][PILLOW][FAIL] Pillow ile yeniden kaydetme başarısız.", color="orange")

        elif step == "ROUNDTRIP":
            if not use_flags["use_png_roundtrip"]:
                continue
            log("[PIPE][ROUNDTRIP] PNG roundtrip başlatılıyor...", color="blue")
            p_png = fix_with_png_roundtrip(input_path, base_output_dir, log)
            if p_png:
                log(f"[PIPE][ROUNDTRIP][OK] {p_png.name}", color="green")
                successes.append(p_png)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][ROUNDTRIP][FAIL] Roundtrip ile anlamlı bir çıktı alınamadı.", color="orange")

        elif step == "EXIF":
            if not use_flags["use_exif_thumb"]:
                continue
            log("[PIPE][EXIF_THUMB] EXIF thumbnail kurtarma başlatılıyor...", color="blue")
            p_exif = fix_with_exif_thumbnail(
                input_path,
                base_output_dir,
                log,
                upscale=exif_thumb_upscale,
            )
            if p_exif:
                log(f"[PIPE][EXIF_THUMB][OK] {p_exif.name}", color="green")
                successes.append(p_exif)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][EXIF_THUMB][FAIL] EXIF thumbnail ile kurtarma yapılamadı.", color="orange")

    # ---------------------------------------------------
    # Özet
    # ---------------------------------------------------
    if not successes:
        log(f"[SUMMARY][FAIL] {input_path.name} için hiçbir yöntem başarılı olmadı.", color="red")
    else:
        best = pick_best_output(successes, strategy_mode=mode)
        if best:
            log(f"[SUMMARY][BEST] En iyi çıktı önerisi: {best.name}", color="darkgreen")
        log(f"[SUMMARY][COUNT] Toplam {len(successes)} çıktı üretildi.", color="darkgreen")

    return successes
