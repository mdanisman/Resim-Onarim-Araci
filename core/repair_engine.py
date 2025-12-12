from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple

from PIL import Image, ImageStat

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
        out_path = output_dir / f"{name}_pillow_fixed{ext}"

        with Image.open(input_path) as im:
            # EXIF'i mümkün olduğunca saklamadan sade kaydet
            save_kwargs: Dict[str, Any] = {}
            if im.format == "JPEG":
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True
            im.save(out_path, **save_kwargs)

        # Hızlı doğrulama
        try:
            with Image.open(out_path) as test_im:
                test_im.load()
        except Exception as e:
            log(f"[PILLOW][VERIFY] {out_path.name} doğrulanamadı -> {e}", color="orange")
            try:
                out_path.unlink()
            except Exception:
                pass
            return None

        log(f"[PILLOW] OK {input_path.name} -> {out_path.name}", color="green")
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
        out_path = output_dir / f"{name}_roundtrip.png"

        with Image.open(input_path) as im:
            im.save(out_path, format="PNG")

        try:
            with Image.open(out_path) as test_im:
                test_im.load()
        except Exception as e:
            log(f"[ROUNDTRIP][VERIFY] {out_path.name} doğrulanamadı -> {e}", color="orange")
            try:
                out_path.unlink()
            except Exception:
                pass
            return None

        log(f"[ROUNDTRIP] OK {input_path.name} -> {out_path.name}", color="green")
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
) -> List[Path]:
    """
    FFmpeg ile birden fazla kalite (qscale) denemesi yaparak onarım dener.
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
            import sys
            if sys.platform.startswith("win"):
                creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        except Exception:
            pass

        outputs: List[Path] = []
        for q in qscale_list:
            out_path = output_dir / f"{name}_ffmpeg_q{q}.jpg"

            cmd = [
                ffmpeg_cmd,
                "-y",
                "-i",
                str(input_path),
                "-qscale:v",
                str(q),
                str(out_path),
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=creationflags,
                    check=False,
                )
                if proc.returncode != 0:
                    log(
                        f"[FFMPEG][q={q}] Hata kodu {proc.returncode}. stderr: {proc.stderr.decode(errors='ignore')[:200]}",
                        color="orange",
                    )
                    if out_path.exists():
                        try:
                            out_path.unlink()
                        except Exception:
                            pass
                    continue

                # Çıktıyı hızlıca Pillow ile doğrula
                try:
                    with Image.open(out_path) as im:
                        im.load()
                except Exception as e:
                    log(f"[FFMPEG][q={q}] Çıktı bozuk -> {e}", color="orange")
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                    continue

                log(f"[FFMPEG][q={q}] OK -> {out_path.name}", color="green")
                outputs.append(out_path)

                if mode == "SAFE":
                    # SAFE modda tek başarılı denemeden sonra bırak
                    break

            except Exception as e:
                log(f"[FFMPEG][q={q}] Çalıştırma hatası -> {e}", color="red")
                try:
                    if out_path.exists():
                        out_path.unlink()
                except Exception:
                    pass

        return outputs

    except Exception as e:
        log(f"[FFMPEG][ERROR] {input_path.name} -> {e}", color="red")
        return []


# =======================================================
# Analiz / skor fonksiyonları
# =======================================================

def _prepare_analysis_image(im: Image.Image, strategy_mode: str) -> Image.Image:
    """
    Analiz için resmi makul boyuta indirir, gerekirse griye çevirir.
    SAFE modda daha hafif işlemler yapılır.
    """
    mode = _normalize_strategy_mode(strategy_mode)
    max_dim = 1024 if mode == "AGGRESSIVE" else 768
    w, h = im.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        im = im.resize(new_size)
    return im


def _estimate_grayness(im: Image.Image) -> float:
    """
    Görüntünün ne kadar "gri / renksiz" göründüğünü kabaca tahmin eder (0.0-1.0).
    1.0'a yakın = çok gri, 0.0'a yakın = renkli / normal.
    """
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    w, h = im.size
    if w <= 0 or h <= 0:
        return 1.0

    pixels = im.resize((64, 64)).getdata()
    import math

    total_sat = 0.0
    count = 0
    for r, g, b in pixels:
        maxc = max(r, g, b)
        minc = min(r, g, b)
        if maxc == 0:
            sat = 0.0
        else:
            sat = (maxc - minc) / float(maxc)
        total_sat += sat
        count += 1
    if count == 0:
        return 1.0
    avg_sat = total_sat / count
    # Düşük sat -> gri, yüksek sat -> renkli
    return 1.0 - max(0.0, min(1.0, avg_sat))


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
    import statistics

    vars_list: List[float] = []
    for i in range(slices):
        top = i * slice_h
        bottom = height if i == slices - 1 else (i + 1) * slice_h
        crop = gray.crop((0, top, width, bottom))
        stat = ImageStat.Stat(crop)
        # var bir liste/tuple olabilir
        v = stat.var[0] if isinstance(stat.var, (list, tuple)) else stat.var
        vars_list.append(float(v))

    if len(vars_list) < 2:
        return 0.0

    # Üst yarı ile alt yarının ortalama varyansını karşılaştır
    mid = len(vars_list) // 2
    top_avg = statistics.mean(vars_list[:mid])
    bottom_avg = statistics.mean(vars_list[mid:])

    if top_avg <= 0:
        return 0.0

    ratio = (bottom_avg - top_avg) / max(top_avg, 1e-6)
    # Negatifse truncation yok say
    if ratio <= 0:
        return 0.0
    # 0.0-1.0 aralığına sıkıştır
    return max(0.0, min(1.0, ratio))


def _estimate_entropy(im: Image.Image) -> float:
    """
    Histogram entropisine göre içerik karmaşıklığını tahmin eder (0.0-1.0).
    Çok düşük entropi -> aşırı düz / bozuk, çok yüksek -> gürültü / aşırı keskin olabilir.
    """
    gray = im.convert("L")
    hist = gray.histogram()
    total = sum(hist)
    if total <= 0:
        return 0.0

    import math

    ent = 0.0
    for c in hist:
        if c <= 0:
            continue
        p = c / float(total)
        ent -= p * math.log2(p)

    # 8 bit gri için maksimum ~8 bit
    ent_norm = ent / 8.0
    return max(0.0, min(1.0, ent_norm))


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
        scale = max_dim / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        gray = gray.resize(new_size)
        w, h = gray.size

    pix = gray.load()
    if pix is None:
        return 0.0

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


def compute_damage_heatmap(image_path: Path, block_size: int = 16) -> Dict[str, Any]:
    """
    Piksel düzeyinde basit bir "hasar ısı haritası" üretir.
    Şimdilik sadece parlaklık varyansına bakar; Faz 1 için yeterli bir temel sağlar.
    Dönüş değeri:
        {
            "block_size": int,
            "rows": int,
            "cols": int,
            "values": List[List[float]],  # 0.0-1.0 arası normalize edilmiş varyans
        }
    """
    try:
        with Image.open(image_path) as im:
            gray = im.convert("L")
            width, height = gray.size
            if width <= 0 or height <= 0:
                return {
                    "block_size": block_size,
                    "rows": 0,
                    "cols": 0,
                    "values": [],
                }

            cols = (width + block_size - 1) // block_size
            rows = (height + block_size - 1) // block_size

            values: List[List[float]] = []
            max_var = 0.0

            for row in range(rows):
                row_vals: List[float] = []
                for col in range(cols):
                    left = col * block_size
                    upper = row * block_size
                    right = min(left + block_size, width)
                    lower = min(upper + block_size, height)

                    crop = gray.crop((left, upper, right, lower))
                    stat = ImageStat.Stat(crop)
                    if isinstance(stat.var, (list, tuple)):
                        var_val = float(stat.var[0])
                    else:
                        var_val = float(stat.var)
                    row_vals.append(var_val)
                    if var_val > max_var:
                        max_var = var_val
                values.append(row_vals)

            # 0-1 aralığına normalize et
            if max_var > 0.0:
                norm = [[v / max_var for v in row_vals] for row_vals in values]
            else:
                norm = values

            return {
                "block_size": block_size,
                "rows": rows,
                "cols": cols,
                "values": norm,
            }
    except Exception:
        # Heatmap analizi başarısız olursa boş veri döndür, akışı bozma
        return {
            "block_size": block_size,
            "rows": 0,
            "cols": 0,
            "values": [],
        }


def summarize_heatmap(hm: Dict[str, Any], threshold: float = 0.7) -> Dict[str, Any]:
    """
    Isı haritası için basit bir özet metrik üretir:
        - toplam blok sayısı
        - eşik üstü "yüksek hasarlı" blok sayısı
        - oran
    """
    values = hm.get("values") or []
    total = 0
    high = 0
    for row in values:
        for v in row:
            total += 1
            try:
                if float(v) >= threshold:
                    high += 1
            except Exception:
                continue

    ratio = (high / float(total)) if total > 0 else 0.0
    return {
        "total_blocks": total,
        "high_blocks": high,
        "high_ratio": ratio,
        "threshold": threshold,
    }


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
        "entropy_score": None,
        "sharpness_score": None,
        "score": 0.0,
    }

    if not path.exists():
        return info

    try:
        stat = path.stat()
        info["size"] = stat.st_size

        with Image.open(path) as im:
            im.load()
            info["mode"] = im.mode
            w, h = im.size
            info["width"] = w
            info["height"] = h
            info["pixels"] = w * h

            mode = _normalize_strategy_mode(strategy_mode)
            im_a = _prepare_analysis_image(im, strategy_mode=mode)

            gray_score = _estimate_grayness(im_a)
            trunc_score = _estimate_truncation(im_a)
            entropy_score = _estimate_entropy(im_a)
            sharp = _estimate_sharpness(im_a)

            info["gray_score"] = gray_score
            info["truncation_score"] = trunc_score
            info["entropy_score"] = entropy_score
            info["sharpness_score"] = sharp

            # Basit skor: birkaç faktörü çarpıp 0-1 aralığına sıkıştır
            base = 1.0

            # Çok gri ise cezalandır
            gray_penalty = 1.0
            if gray_score is not None:
                if gray_score > 0.7:
                    gray_penalty = 0.4
                elif gray_score > 0.5:
                    gray_penalty = 0.7

            # Truncation yüksekse cezalandır
            trunc_penalty = 1.0
            if trunc_score is not None:
                if trunc_score > 0.7:
                    trunc_penalty = 0.3
                elif trunc_score > 0.5:
                    trunc_penalty = 0.7

            # Entropi çok düşük veya çok yüksek ise cezalandır
            entropy_factor = 1.0
            if entropy_score is not None:
                if entropy_score < 0.3:
                    entropy_factor = 0.5
                elif entropy_score > 0.9:
                    entropy_factor = 0.7

            # Keskinliği çok düşükse hafif ceza
            sharp_factor = 1.0
            if sharp is not None:
                if sharp < 0.1:
                    sharp_factor = 0.6
                elif sharp < 0.2:
                    sharp_factor = 0.8

            # Boyuta göre basit bir normalizasyon (çok küçük dosyaları cezalandır)
            size_factor = 1.0
            if stat.st_size < 20_000:  # 20KB'dan küçükler şüpheli
                size_factor = 0.5
            elif stat.st_size < 50_000:
                size_factor = 0.8

            score = base * size_factor * gray_penalty * trunc_penalty * entropy_factor * sharp_factor
            info["score"] = max(0.0, min(1.0, float(score)))

        info["verify"] = True
    except Exception:
        info["verify"] = False

    return info


def pick_best_output(paths: List[Path], strategy_mode: str = "NORMAL") -> Optional[Path]:
    """
    Birden fazla çıktı arasından en iyi görüneni seçer.
    """
    if not paths:
        return None

    best_path: Optional[Path] = None
    best_score = -1.0

    for p in paths:
        info = evaluate_output(p, strategy_mode=strategy_mode)
        if not info.get("verify"):
            continue
        score = float(info.get("score") or 0.0)
        if score > best_score:
            best_score = score
            best_path = p

    return best_path


def _parse_jpeg_dimensions_from_bytes(data: bytes) -> Optional[Tuple[int, int, int]]:
    """
    JPEG baytlarından genişlik, yükseklik ve bileşen sayısını (C) okumaya çalışır.
    Basit bir parser; sadece boyut için kullanılır.
    """
    try:
        i = 0
        n = len(data)
        if n < 4:
            return None

        # SOI kontrolü
        if not (data[0] == 0xFF and data[1] == 0xD8):
            return None
        i = 2

        while i + 4 <= n:
            if data[i] != 0xFF:
                i += 1
                continue

            marker = data[i + 1]
            i += 2

            if marker in (0xD8, 0xD9):  # SOI / EOI
                continue

            if i + 2 > n:
                break
            seg_len = (data[i] << 8) + data[i + 1]
            i += 2
            if seg_len < 2:
                break
            seg_start = i
            seg_end = i + seg_len - 2
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
            data = f.read(4096)  # baştan 4KB genelde SOF için yeter
        return _parse_jpeg_dimensions_from_bytes(data)
    except Exception:
        return None


def _select_best_headers_for_image(
    input_path: Path,
    header_library: List[bytes],
    log: LogFunc,
) -> List[bytes]:
    """
    Header kütüphanesinden, giriş görselinin boyut/sampling bilgisine en çok uyan birkaç header seçer.
    Faz 1 için basit bir heuristik: boyut farkı ve bileşen sayısı benzer olanları öne al.
    """
    try:
        dim = _parse_jpeg_dimensions_from_file(input_path)
    except Exception:
        dim = None

    if not header_library or dim is None:
        return header_library

    in_w, in_h, in_c = dim
    scored: List[Tuple[float, bytes]] = []

    for hdr in header_library:
        hdim = _parse_jpeg_dimensions_from_bytes(hdr)
        if hdim is None:
            continue
        w, h, c = hdim
        # Boyut farkını ve bileşen farkını basitçe ölç
        size_diff = abs(w - in_w) + abs(h - in_h)
        comp_diff = abs(c - in_c)
        score = size_diff + comp_diff * 1000
        scored.append((float(score), hdr))

    if not scored:
        return header_library

    scored.sort(key=lambda x: x[0])
    best_headers = [hdr for _, hdr in scored[:5]]
    log(f"[HEADER-LIB] {len(header_library)} header içinden {len(best_headers)} aday seçildi.", color="blue")
    return best_headers


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
            diag["can_open"] = True
            diag["width"], diag["height"] = im.size
            diag["mode"] = im.mode

            im_a = _prepare_analysis_image(im, strategy_mode="SAFE")
            diag["gray_score"] = _estimate_grayness(im_a)
            diag["truncation_score"] = _estimate_truncation(im_a)

            # Basit bir şiddet tahmini
            trunc = diag["truncation_score"] or 0.0
            if trunc < 0.2:
                diag["severity"] = "light"
            elif trunc < 0.5:
                diag["severity"] = "medium"
            else:
                diag["severity"] = "heavy"

        log(
            f"[DIAG] {input_path.name}: can_open={diag['can_open']}, "
            f"size={diag['width']}x{diag['height']}, severity={diag['severity']}",
            color="blue",
        )
    except Exception as e:
        diag["can_open"] = False
        diag["severity"] = "heavy"
        log(f"[DIAG] {input_path.name} açılamadı -> {e}", color="orange")

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
            if use_flags["use_marker"]:
                steps.append("MARKER")
            if use_flags["use_header"]:
                steps.append("HEADER")
            if use_flags["use_pillow"] and can_open:
                steps.append("PILLOW")

        elif mode == "AGGRESSIVE":
            if use_flags["use_marker"]:
                steps.append("MARKER")
            if use_flags["use_header"]:
                steps.append("HEADER")
            if use_flags["use_png_crc"] and is_png:
                steps.append("PNG_CRC")
            if use_flags["use_partial_top"] and severity in ("medium", "heavy"):
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
            if use_flags["use_marker"]:
                steps.append("MARKER")
            if use_flags["use_header"]:
                steps.append("HEADER")
            if use_flags["use_partial_top"] and severity == "heavy":
                steps.append("PARTIAL")
            if use_flags["use_ffmpeg"]:
                steps.append("FFMPEG")
            if use_flags["use_pillow"] and can_open:
                steps.append("PILLOW")
            if use_flags["use_png_roundtrip"] and not can_open:
                steps.append("ROUNDTRIP")
            if use_flags["use_exif_thumb"]:
                steps.append("EXIF")

    elif is_png:
        # PNG için
        if use_flags["use_png_crc"]:
            steps.append("PNG_CRC")
        if use_flags["use_pillow"]:
            steps.append("PILLOW")
        if mode == "AGGRESSIVE" and use_flags["use_ffmpeg"]:
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
            if use_flags["use_png_roundtrip"] and not can_open:
                steps.append("ROUNDTRIP")

    return steps


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
        "use_embed_scan": use_embed_scan,
        "use_partial_top": use_partial_top,
        "use_exif_thumb": use_exif_thumb,
        "use_png_crc": use_png_crc and is_png,
    }

    steps = _build_step_plan(
        is_jpeg=is_jpeg,
        is_png=is_png,
        diag=diag,
        strategy_mode=mode,
        use_flags=use_flags,
    )

    log(f"[PIPE] Adım planı: {steps}", color="blue")

    # EMBED_SCAN için ayrı çıktı klasörü
    embed_output_dir = base_output_dir / "embedded"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for step in steps:
        if step == "EMBED_SCAN":
            if not use_flags["use_embed_scan"]:
                continue
            log("[PIPE][EMBED] Gömülü JPEG taraması başlatılıyor...", color="blue")
            emb = extract_all_jpegs_from_blob(input_path, embed_output_dir, log)
            if emb:
                successes.extend(emb)
                if stop_on_first_success:
                    return successes

        elif step == "PNG_CRC":
            if not use_flags["use_png_crc"]:
                continue
            log("[PIPE][PNG-CRC] PNG CRC onarımı başlatılıyor...", color="blue")
            p = fix_with_png_crc(
                input_path,
                base_output_dir,
                log,
                skip_ancillary_on_crc_error=png_crc_skip_ancillary,
            )
            if p:
                successes.append(p)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][PNG-CRC][FAIL] PNG CRC onarımı başarısız.", color="orange")

        elif step == "MARKER":
            if not use_flags["use_marker"]:
                continue
            log("[PIPE][MARKER] JPEG marker onarımı başlatılıyor...", color="blue")
            p_m = fix_with_jpeg_markers(input_path, base_output_dir, log)
            if p_m:
                successes.append(p_m)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][MARKER][FAIL] Marker onarımı başarısız.", color="orange")

        elif step == "HEADER":
            if not use_flags["use_header"]:
                continue
            log("[PIPE][HEADER] Smart Header V3 onarımı başlatılıyor...", color="blue")
            p_h = fix_with_smart_header_v3(
                input_path=input_path,
                output_dir=base_output_dir,
                ref_header_bytes=ref_header_bytes,
                log=log,
                header_size=header_size,
                keep_apps=keep_apps,
                keep_com=keep_com,
                header_library=selected_header_library,
            )
            if p_h:
                successes.append(p_h)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][HEADER][FAIL] Smart Header V3 onarımı başarısız.", color="orange")

        elif step == "PARTIAL":
            if not use_flags["use_partial_top"]:
                continue
            log("[PIPE][PARTIAL_TOP] Kısmi üstten kesme kurtarma başlatılıyor...", color="blue")
            p_parts = partial_top_recovery(input_path, base_output_dir, log)
            if p_parts:
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
                successes.extend(p_ff)
                if stop_on_first_success:
                    return successes
            else:
                log("[PIPE][FFMPEG][FAIL] FFmpeg ile anlamlı bir çıktı alınamadı.", color="orange")

        elif step == "PILLOW":
            if not use_flags["use_pillow"]:
                continue
            log("[PIPE][PILLOW] Pillow ile yeniden kaydetme başlatılıyor...", color="blue")
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
            # En iyi çıktı için basit bir ısı haritası özeti üret
            hm = compute_damage_heatmap(best)
            hm_summary = summarize_heatmap(hm, threshold=0.7)
            high_ratio = hm_summary.get("high_ratio", 0.0)
            total_blocks = hm_summary.get("total_blocks", 0)
            high_blocks = hm_summary.get("high_blocks", 0)
            log(
                f"[SUMMARY][HEATMAP] Blok sayısı: {total_blocks}, yüksek hasarlı blok: {high_blocks} "
                f"({high_ratio:.1%})",
                color="darkgreen",
            )
        log(f"[SUMMARY][COUNT] Toplam {len(successes)} çıktı üretildi.", color="darkgreen")

    return successes