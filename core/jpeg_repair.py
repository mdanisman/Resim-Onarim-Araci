from __future__ import annotations

import io
import mmap
import tempfile
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple

from PIL import Image

from .jpeg_parser import (
    parse_jpeg_segments,
    extract_sof_sampling,
    get_sof_dimensions,
    JpegSegment,
    SOI,
    EOI,
    SOS,
    DQT,
    DHT,
    COM,
    APP0,
    APP15,
    SOF0,
    SOF2,
)

# EXIF thumbnail için
try:
    import piexif
    HAS_PIEXIF = True
except Exception:
    HAS_PIEXIF = False


# -------------------------------------------------------
# Header + sampling destekli onarım (Smart Header V3)
# -------------------------------------------------------

def rebuild_header_with_tables_v2(
    original_data: bytes,
    ref_header_bytes: bytes,
    log: Callable[..., None],
    keep_apps: bool = True,
    keep_com: bool = True,
) -> Optional[Tuple[bytes, Dict[str, Any]]]:
    """
    Orijinal JPEG'in header kısmını parser ile temizler,
    referans header'dan DQT/DHT tablolarını alıp yeniden inşa eder.

    Başarılı ise:
        (new_data, meta) döner,
    başarısız ise:
        None döner.

    meta:
        dqt_count: int
        dht_count: int
        sampling_warning: Optional[str]
    """
    try:
        # Orijinal tüm segmentleri parse et
        orig_segments = parse_jpeg_segments(original_data)
        if not orig_segments or orig_segments[0].marker != SOI:
            log("[SMART-HEADER V3] Orijinal dosyada geçerli SOI yok.", color="orange")
            return None

        # SOF ve SOS konumunu bul
        sos_index: Optional[int] = None
        sof_seg: Optional[JpegSegment] = None
        for seg in orig_segments:
            if seg.marker in (SOF0, SOF2):
                sof_seg = seg
            if seg.marker == SOS:
                sos_index = seg.start
                break

        if sos_index is None:
            log("[SMART-HEADER V3] SOS (FFDA) bulunamadı.", color="orange")
            return None

        # SOS sonrası bitişi hesapla
        sos_list = parse_jpeg_segments(original_data[sos_index:])
        if not sos_list or sos_list[0].marker != SOS:
            log("[SMART-HEADER V3] SOS segmenti parse edilemedi.", color="orange")
            return None

        sos_seg_rel = sos_list[0]
        sos_abs_end = sos_index + sos_seg_rel.length
        scan_data = original_data[sos_abs_end:]

        # Header prefix (SOI - SOS arası)
        header_prefix = original_data[:sos_index]
        header_segments = parse_jpeg_segments(header_prefix)

        # Header'dan önce temiz bir temel oluştur
        cleaned_header_parts: List[bytes] = []
        for seg in header_segments:
            # DQT/DHT komple temizlenecek, referanstan yeniden alınacak
            if seg.marker in (DQT, DHT):
                continue
            # İstenmezse COM / APP segmentlerini buda
            if seg.marker == COM and not keep_com:
                continue
            if APP0 <= seg.marker <= APP15 and not keep_apps:
                continue
            cleaned_header_parts.append(seg.data)

        # Referans header’dan DQT/DHT tablolarını al
        ref_segments = parse_jpeg_segments(ref_header_bytes)
        ref_tables: List[bytes] = []
        dqt_count = 0
        dht_count = 0
        for rseg in ref_segments:
            if rseg.marker == DQT:
                ref_tables.append(rseg.data)
                dqt_count += 1
            elif rseg.marker == DHT:
                ref_tables.append(rseg.data)
                dht_count += 1

        if dqt_count == 0 or dht_count == 0:
            log("[SMART-HEADER V3] Referansta DQT/DHT bulunamadı, parser tabanlı onarım atlandı.", color="orange")
            return None

        # Sampling uyumu hakkında bilgi topla (sadece uyarı amaçlı)
        sof_info = extract_sof_sampling(sof_seg) if sof_seg else None
        ref_sof = None
        for rseg in ref_segments:
            if rseg.marker in (SOF0, SOF2):
                ref_sof = extract_sof_sampling(rseg)
                break

        sampling_warning = None
        if sof_info and ref_sof:
            orig_comp = sof_info["components"]
            ref_comp = ref_sof["components"]
            if len(orig_comp) == len(ref_comp) == 3:
                mismatches = []
                for idx in range(3):
                    o = orig_comp[idx]
                    r = ref_comp[idx]
                    if (o["h"], o["v"]) != (r["h"], r["v"]):
                        mismatches.append((idx, (o["h"], o["v"]), (r["h"], r["v"])))
                if mismatches:
                    sampling_warning = f"Sampling mismatch: {mismatches} (decode riski)"
                    log(f"[SMART-HEADER V3] {sampling_warning}", color="orange")

        # Temiz header + referans tablolar + orijinal SOS segmenti
        rebuilt_header = b"".join(cleaned_header_parts + ref_tables + [original_data[sos_index:sos_abs_end]])
        new_data = rebuilt_header + scan_data

        meta = {"dqt_count": dqt_count, "dht_count": dht_count, "sampling_warning": sampling_warning}
        return new_data, meta
    except Exception as e:
        log(f"[SMART-HEADER V3] Header yeniden inşa hatası: {e}", color="red")
        return None


# -------------------------------------------------------
# Gömülü JPEG tarama
# -------------------------------------------------------

def extract_all_jpegs_from_blob(
    input_path: Path,
    output_dir: Path,
    log: Callable[..., None],
    min_size: int = 1024,
    *,
    chunk_size: int = 4 * 1024 * 1024,
    max_bytes: int = 200 * 1024 * 1024,
) -> List[Path]:
    out_files: List[Path] = []
    try:
        file_size = input_path.stat().st_size
        if file_size > max_bytes:
            log(
                f"[EMBED] {input_path.name} boyutu {file_size / (1024*1024):.1f}MB, limit {max_bytes / (1024*1024):.0f}MB. Tarama atlandı.",
                color="orange",
            )
            return out_files

        with open(input_path, "rb") as f:
            buffer = b""
            count = 0

            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk

                while True:
                    soi = buffer.find(b"\xff\xd8")
                    if soi == -1:
                        # Yersiz bellek büyümesini engellemek için kuyruk verisini sınırla
                        if len(buffer) > chunk_size * 2:
                            buffer = buffer[-chunk_size:]
                        break

                    eoi = buffer.find(b"\xff\xd9", soi + 2)
                    if eoi == -1:
                        buffer = buffer[soi:]
                        break

                    blob = buffer[soi:eoi + 2]
                    buffer = buffer[eoi + 2:]

                    if len(blob) >= min_size:
                        count += 1
                        name = f"{input_path.stem}_embedded_{count}.jpg"
                        out_path = output_dir / name
                        with open(out_path, "wb") as wf:
                            wf.write(blob)
                        out_files.append(out_path)
                        try:
                            with Image.open(out_path) as im:
                                im.verify()
                            log(f"[EMBED] OK -> {name}", color="green")
                        except Exception as e:
                            log(f"[EMBED] UYARI verify -> {name}: {e}", color="orange")

        if out_files:
            log(
                f"[EMBED] {input_path.name} içinde {len(out_files)} gömülü JPEG çıkarıldı.",
                color="darkgreen",
            )
        else:
            log(f"[EMBED] {input_path.name} içinde gömülü JPEG bulunamadı.", color="orange")
    except Exception as e:
        log(f"[EMBED] HATA {input_path.name} -> {e}", color="red")
    return out_files

# -------------------------------------------------------
# Partial Top Recovery
# -------------------------------------------------------

def partial_top_recovery(
    input_path: Path,
    output_dir: Path,
    log: Callable[..., None],
    step_ratio: float = 0.07,
    min_keep_ratio: float = 0.35,
    max_variants: int = 4,
    *,
    max_bytes: int = 200 * 1024 * 1024,
) -> List[Path]:
    results: List[Path] = []
    try:
        file_size = input_path.stat().st_size
        if file_size > max_bytes:
            log(
                f"[PARTIAL] {input_path.name} boyutu {file_size / (1024*1024):.1f}MB, limit {max_bytes / (1024*1024):.0f}MB. İşlem atlandı.",
                color="orange",
            )
            return results

        with open(input_path, "rb") as f, mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as data:
            total = len(data)
            min_keep = int(total * min_keep_ratio)
            step = max(int(total * step_ratio), 1024)

            keep_len = total
            variant_index = 0

            while keep_len >= min_keep and variant_index < max_variants:
                keep = data[:keep_len]
                soi = keep.find(b"\xff\xd8")
                eoi = keep.rfind(b"\xff\xd9")
                candidate = keep[soi:eoi + 2] if (soi != -1 and eoi != -1 and eoi > soi) else keep

                variant_index += 1
                out_path = output_dir / f"{input_path.stem}_partial_top_{variant_index}.jpg"
                try:
                    with open(out_path, "wb") as wf:
                        wf.write(candidate)
                    try:
                        with Image.open(out_path) as im:
                            im.load()
                        log(f"[PARTIAL] OK -> {out_path.name} (korunan {keep_len}/{total} bayt)", color="green")
                        results.append(out_path)
                    except Exception as e:
                        log(f"[PARTIAL] Deneme başarısız ({keep_len}B, v{variant_index}): {e}", color="orange")
                        try:
                            out_path.unlink()
                        except Exception:
                            pass
                except Exception as e:
                    log(f"[PARTIAL] Yazma hatası: {e}", color="red")

                keep_len -= step

        if results:
            log(f"[PARTIAL] {len(results)} başarılı varyant üretildi.", color="darkgreen")
        else:
            log("[PARTIAL] Uygun bir üst kısım kurtarılamadı.", color="orange")
    except Exception as e:
        log(f"[PARTIAL] HATA {input_path.name} -> {e}", color="red")
    return results


# -------------------------------------------------------
# Header kütüphanesi
# -------------------------------------------------------

def extract_header_profile(header_bytes: bytes) -> Dict[str, Any]:
    prof: Dict[str, Any] = {"ok": False, "components": None, "width": None, "height": None}
    try:
        segs = parse_jpeg_segments(header_bytes)
        sof = None
        for s in segs:
            if s.marker in (SOF0, SOF2):
                sof = s
                break
        if not sof:
            return prof
        info = extract_sof_sampling(sof)
        if not info:
            return prof
        dims = get_sof_dimensions(sof)
        prof["ok"] = True
        prof["components"] = info["components"]
        if dims:
            prof["width"], prof["height"] = dims
        return prof
    except Exception:
        return prof


def select_best_header_for_image(
    original_data: bytes,
    header_library: List[bytes],
    log: Callable[..., None],
) -> Optional[bytes]:
    """
    Header kütüphanesi içinden, orijinal görsele en çok uyan header'ı seçer.
    Skor kriterleri:
        - Component sayısı aynı mı?
        - Sampling (h,v) ne kadar benzer?
        - Boyut (width/height) ne kadar yakın?
    """
    try:
        orig_segs = parse_jpeg_segments(original_data)
        orig_sof = None
        for s in orig_segs:
            if s.marker in (SOF0, SOF2):
                orig_sof = s
                break

        orig_info = extract_sof_sampling(orig_sof) if orig_sof else None
        orig_dims = get_sof_dimensions(orig_sof) if orig_sof else None
        if not orig_info:
            log("[HEADER-LIB] Orijinal SOF bilgisi okunamadı, kütüphane seçiminde eşleştirme zor.", color="orange")

        def score_header(hdr: bytes) -> int:
            prof = extract_header_profile(hdr)
            if not prof["ok"] or not orig_info:
                return 0

            ocomps = orig_info["components"]
            hcomps = prof["components"]
            if len(ocomps) != len(hcomps):
                return 0

            score = 0
            for oi, hi in zip(ocomps, hcomps):
                if (oi["h"], oi["v"]) == (hi["h"], hi["v"]):
                    score += 2
                else:
                    score += 1 if abs(oi["h"] - hi["h"]) + abs(oi["v"] - hi["v"]) <= 1 else 0

            if orig_dims and prof["width"] and prof["height"]:
                ow, oh = orig_dims
                hw, hh = prof["width"], prof["height"]
                diff = abs(ow - hw) + abs(oh - hh)
                if diff == 0:
                    score += 4
                elif diff < 256:
                    score += 2
                elif diff < 1024:
                    score += 1

            return score

        best = None
        best_score = -1
        for hdr in header_library:
            s = score_header(hdr)
            if s > best_score:
                best = hdr
                best_score = s

        if best is not None and best_score > 0:
            log(f"[HEADER-LIB] En iyi header skor: {best_score}", color="darkgreen")
            return best
        else:
            log("[HEADER-LIB] Uygun header bulunamadı, mevcut referansı kullan.", color="orange")
            return None
    except Exception as e:
        log(f"[HEADER-LIB] Seçim hatası: {e}", color="red")
        return None


def build_header_library_from_folder(folder: Path, read_size: int = 32 * 1024) -> List[bytes]:
    headers: List[bytes] = []
    try:
        for f in folder.iterdir():
            if not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg"):
                continue
            try:
                with open(f, "rb") as rf:
                    b = rf.read(read_size)
                if b.startswith(b"\xff\xd8"):
                    headers.append(b)
            except Exception:
                continue
    except Exception:
        pass
    return headers


# -------------------------------------------------------
# EXIF Thumbnail Kurtarma
# -------------------------------------------------------

def fix_with_exif_thumbnail(
    input_path: Path,
    output_dir: Path,
    log: Callable[..., None],
    upscale: bool = False,
) -> Optional[Path]:
    try:
        if input_path.suffix.lower() not in [".jpg", ".jpeg"]:
            return None
        if not HAS_PIEXIF:
            log("[EXIF-THUMB] piexif modülü yüklü değil (pip install piexif).", color="orange")
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        name = input_path.stem
        out_path = output_dir / f"{name}_exif_thumb.jpg"

        exif_dict = piexif.load(str(input_path))
        thumb = exif_dict.get("thumbnail")
        if thumb:
            with open(out_path, "wb") as f:
                f.write(thumb)
            try:
                with Image.open(out_path) as im:
                    im.load()
            except Exception:
                pass
            log(f"[EXIF-THUMB] OK {input_path.name} -> {out_path.name}", color="green")

            if upscale:
                try:
                    up_path = output_dir / f"{name}_exif_thumb_upscaled.jpg"
                    with Image.open(io.BytesIO(thumb)) as imt:
                        imt.load()
                        w, h = imt.size
                        target_min = 512
                        scale = max(target_min / max(w, h), 1.0)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        up_im = imt.resize((new_w, new_h), Image.LANCZOS)
                        up_im.save(up_path, quality=95, optimize=True)
                    log(f"[EXIF-THUMB] UPSCALE -> {up_path.name} ({new_w}x{new_h})", color="darkgreen")
                    return up_path
                except Exception as e:
                    log(f"[EXIF-THUMB] Upscale başarısız: {e}", color="orange")
                    return out_path

            return out_path
        else:
            log(f"[EXIF-THUMB] {input_path.name} içinde thumbnail bulunamadı.", color="orange")
            return None
    except Exception as e:
        log(f"[EXIF-THUMB] HATA {input_path.name} -> {e}", color="red")
        return None


# -------------------------------------------------------
# Marker temizleme
# -------------------------------------------------------

def fix_with_jpeg_markers(input_path: Path, output_dir: Path, log: Callable[..., None]) -> Optional[Path]:
    ext = input_path.suffix.lower()
    if ext not in [".jpg", ".jpeg"]:
        return None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        name = input_path.stem
        out_path = output_dir / f"{name}_fixed_markers{ext}"

        with open(input_path, "rb") as f:
            data = f.read()

        soi = data.find(b"\xff\xd8")
        eoi = data.rfind(b"\xff\xd9")
        if soi == -1 or eoi == -1 or eoi <= soi:
            log(f"[MARKERS] {input_path.name} -> Geçerli SOI/EOI bulunamadı.", color="orange")
            return None

        new_data = data[soi:eoi + 2]
        with open(out_path, "wb") as f:
            f.write(new_data)

        try:
            with Image.open(out_path) as test_im:
                test_im.load()
        except Exception as e:
            log(f"[MARKERS] UYARI {out_path.name} yükleme başarısız -> {e}", color="red")

        log(f"[MARKERS] OK  {input_path.name} -> {out_path.name}", color="green")
        return out_path
    except Exception as e:
        log(f"[MARKERS] HATA {input_path.name} -> {e}", color="red")
        return None


# -------------------------------------------------------
# Smart Header V3 wrapper (header kütüphanesi + fallback)
# -------------------------------------------------------

def fix_with_smart_header_v3(
    input_path: Path,
    output_dir: Path,
    ref_header_bytes: Optional[bytes],
    log: Callable[..., None],
    header_size: int,
    keep_apps: bool = True,
    keep_com: bool = True,
    header_library: Optional[List[bytes]] = None,
) -> Optional[Path]:
    """
    Smart Header V3 onarımı.

    Strateji:
      1) Eğer header kütüphanesi varsa:
            - En uyumlu header'ı seç
            - Bununla parser tabanlı onarımı dene
            - Başarısız olursa, varsa ref_header_bytes ile tekrar dene
      2) Header kütüphanesi yoksa:
            - ref_header_bytes ile parser tabanlı onarımı dene
      3) Parser onarımı başarısız olursa:
            - Fallback: sabit header kopyalama (header_size'a göre)
    """
    ext = input_path.suffix.lower()
    if ext not in [".jpg", ".jpeg"]:
        return None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        name = input_path.stem
        out_path = output_dir / f"{name}_fixed_smart_header{ext}"

        with open(input_path, "rb") as f:
            data = f.read()

        # 1) Header seçim stratejisi: önce kütüphane, sonra referans
        primary_header: Optional[bytes] = None
        fallback_header: Optional[bytes] = None

        if header_library:
            best_hdr = select_best_header_for_image(data, header_library, log)
            if best_hdr:
                primary_header = best_hdr
                fallback_header = ref_header_bytes
            else:
                primary_header = ref_header_bytes
        else:
            primary_header = ref_header_bytes

        if not primary_header:
            log("[SMART-HEADER V3] Referans header yok, yöntem atlandı.", color="orange")
            return None

        # Tekrarlı deneme yapabilmek için küçük bir yardımcı fonksiyon
        def try_parser_repair(header_bytes: bytes, label: str) -> Optional[bytes]:
            rebuilt = rebuild_header_with_tables_v2(
                data,
                header_bytes,
                log,
                keep_apps=keep_apps,
                keep_com=keep_com,
            )
            if not rebuilt:
                return None

            new_data, meta = rebuilt
            # Geçerli mi diye hızlı bir Pillow kontrolü yap
            try:
                with Image.open(io.BytesIO(new_data)) as test_im:
                    test_im.load()
            except Exception as e:
                log(
                    f"[SMART-HEADER V3] {label} parser onarım çıktı yüklenemedi -> {e} "
                    f"(DQT={meta['dqt_count']}, DHT={meta['dht_count']})",
                    color="orange",
                )
                return None

            # Buraya geldiysek onarım başarılı
            with open(out_path, "wb") as f_out:
                f_out.write(new_data)

            log(
                f"[SMART-HEADER V3] {label} OK  {input_path.name} -> {out_path.name} "
                f"(DQT={meta['dqt_count']}, DHT={meta['dht_count']})",
                color="green",
            )
            return new_data

        # 2) Önce primary header ile parser onarım dene
        parser_data = try_parser_repair(primary_header, "PRIMARY")
        if not parser_data and fallback_header and fallback_header is not primary_header:
            # Primary başarısızsa, varsa fallback header ile bir kere daha dene
            log("[SMART-HEADER V3] PRIMARY başarısız, fallback header ile tekrar deneniyor.", color="orange")
            parser_data = try_parser_repair(fallback_header, "FALLBACK")

        if parser_data is not None:
            # Parser tabanlı onarım başarılı oldu
            return out_path

        # 3) Buraya düşerse parser-based onarım tamamen başarısız. Fallback stratejisine geç.
        log("[SMART-HEADER V3] Parser tabanlı onarım başarısız; sabit header fallback deneniyor.", color="orange")

        chosen_header = primary_header or fallback_header
        if not chosen_header:
            log("[SMART-HEADER V3] Fallback için bile header yok.", color="red")
            return None

        soi_index = data.find(b"\xff\xd8")
        ffda_index = data.find(b"\xff\xda", soi_index) if soi_index != -1 else -1
        if soi_index == -1:
            log("[SMART-HEADER V3] SOI bulunamadı, fallback atlandı.", color="orange")
            return None

        if ffda_index == -1:
            log(
                "[SMART-HEADER V3] FFDA bulunamadı. Fallback: sabit kopya "
                f"({header_size}B) ile denenecek.",
                color="orange",
            )
            replacement_size = header_size
        else:
            replacement_size = ffda_index + 2 - soi_index

        ref_copy = chosen_header[:replacement_size]
        if not ref_copy.startswith(b"\xff\xd8"):
            log("[SMART-HEADER V3] Referans header geçersiz, fallback başarısız.", color="red")
            return None

        original_scan_data_start = soi_index + (
            ffda_index + 2 - soi_index if ffda_index != -1 else replacement_size
        )
        scan_data = data[original_scan_data_start:]
        new_data = ref_copy + scan_data

        with open(out_path, "wb") as f:
            f.write(new_data)

        try:
            with Image.open(out_path) as test_im:
                test_im.load()
        except Exception as e:
            log(f"[SMART-HEADER V3] Fallback yükleme uyarısı -> {e}", color="orange")

        log(f"[SMART-HEADER V3] Fallback OK  {input_path.name} -> {out_path.name}", color="green")
        return out_path
    except Exception as e:
        log(f"[SMART-HEADER V3] HATA {input_path.name} -> {e}", color="red")
        return None
