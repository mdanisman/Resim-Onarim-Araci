from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# -------------------------------------------------------
# JPEG marker sabitleri
# -------------------------------------------------------

SOI = 0xD8
EOI = 0xD9
SOS = 0xDA
DQT = 0xDB
DHT = 0xC4
COM = 0xFE
APP0 = 0xE0
APP15 = 0xEF
SOF0 = 0xC0
SOF2 = 0xC2
RST0 = 0xD0
RST7 = 0xD7
TEM = 0x01

# Uzunluk alanı taşımayan marker'lar
NO_LENGTH_MARKERS = set([SOI, EOI, TEM] + list(range(RST0, RST7 + 1)))


# -------------------------------------------------------
# Veri yapısı
# -------------------------------------------------------

@dataclass
class JpegSegment:
    marker: int      # 0xC0, 0xDA vs. (FF sonrası byte)
    start: int       # data içindeki başlangıç offset'i (0xFF'in pozisyonu)
    length: int      # segmentin toplam byte uzunluğu (start'tan itibaren)
    data: bytes      # segmentin ham verisi (0xFF + marker + ... + payload)
    has_length: bool # marker uzunluk alanına sahip mi (SOI/RST/TEM vs. hariç)


# -------------------------------------------------------
# Yardımcı
# -------------------------------------------------------

def _safe_get_u16_be(b: bytes, pos: int, default: int = 0) -> int:
    try:
        return (b[pos] << 8) | b[pos + 1]
    except Exception:
        return default


# -------------------------------------------------------
# Segment parser
# -------------------------------------------------------

def parse_jpeg_segments(data: bytes) -> List[JpegSegment]:
    """
    JPEG segment parser'ı.

    Verilen byte dizisi içindeki JPEG segmentlerini, ilk marker'dan başlayarak
    SOS (0xFFDA) segmentine kadar parse eder. Genelde:
        - Tüm dosya verildiğinde: SOI'den SOS'e kadar header segmentleri
        - Bir slice verildiğinde  : Slice içindeki ilk marker'dan itibaren

    Not:
      - SOS görüldüğünde döngü sonlandırılır (scan verisi parse edilmez).
      - Hatalı uzunluk durumlarında, EOI'ye kadar tek bir segment halinde
        toparlamaya çalışır.
    """
    segments: List[JpegSegment] = []
    n = len(data)
    if n < 2:
        return segments

    # ÖNEMLİ DEĞİŞİKLİK:
    # Daha önce burada data.find(b"\xff\xd8") kullanıyordun.
    # Artık buffer'ın başından itibaren ilk marker'ı arıyoruz ki
    # original_data[sos_index:] gibi slice'larda da çalışabilsin.
    i = 0

    while i < n - 1:
        # 0xFF olmayan byte'ları atla
        if data[i] != 0xFF:
            i += 1
            continue

        # Birden fazla 0xFF flood'u varsa atla (00 stuffing vb.)
        j = i + 1
        while j < n and data[j] == 0xFF:
            j += 1
        if j >= n:
            break

        marker = data[j]

        # Uzunluk alanı olmayan marker'lar (SOI, EOI, RSTx, TEM)
        if marker in NO_LENGTH_MARKERS:
            seg_data = data[i:j + 1]
            segments.append(
                JpegSegment(
                    marker=marker,
                    start=i,
                    length=len(seg_data),
                    data=seg_data,
                    has_length=False,
                )
            )
            i = j + 1

            if marker == EOI:
                # Dosya sonu
                break
            continue

        # Uzunluk alanı olan marker'lar
        len_pos = j + 1
        if len_pos + 1 >= n:
            # Uzunluk alanı yarım kalmış, parse edemeyiz
            break

        seg_len = _safe_get_u16_be(data, len_pos, default=0)
        # JPEG standardına göre length alanı, length byte'larını da içerir.
        # Toplam segment uzunluğu (0xFF'den itibaren) = 2 (FF+marker) + seg_len
        if seg_len <= 0:
            break

        end = len_pos + seg_len  # DOĞRUSU: len_pos + seg_len (önceden +2 fazlaydı)

        if end > n:
            # Uzunluk alanı bozuk, EOI arayıp oraya kadar alalım
            eoi_pos = data.find(b"\xff\xd9", j)
            if eoi_pos != -1:
                end = eoi_pos + 2
            else:
                end = n

        seg_data = data[i:end]
        segments.append(
            JpegSegment(
                marker=marker,
                start=i,
                length=end - i,
                data=seg_data,
                has_length=True,
            )
        )
        i = end

        # SOS'e kadar header parse ediyoruz, SOS görüldü mü dur
        if marker == SOS:
            break

    return segments


# -------------------------------------------------------
# SOF parsing (sampling & boyut)
# -------------------------------------------------------

def extract_sof_sampling(seg: JpegSegment) -> Optional[Dict[str, Any]]:
    """
    SOF0/SOF2 segmentinden component (sampling) bilgilerini çıkarır.

    Dönen sözlük örneği:
        {
          "components": [
             {"id": 1, "h": 2, "v": 2, "q": 0},
             {"id": 2, "h": 1, "v": 1, "q": 1},
             {"id": 3, "h": 1, "v": 1, "q": 1},
          ]
        }
    """
    try:
        if seg.marker not in (SOF0, SOF2) or not seg.has_length:
            return None

        d = seg.data
        # Segment yapısı (SOF0/2):
        # 0: 0xFF
        # 1: marker
        # 2-3: length
        # 4: sample precision
        # 5-6: height
        # 7-8: width
        # 9: component count (Nf)
        comp_offset = 2 + 2 + 1 + 2 + 2  # = 9
        if len(d) < comp_offset + 1:
            return None

        components = d[comp_offset]
        base = comp_offset + 1
        items = []

        for k in range(components):
            p = base + k * 3
            if p + 2 >= len(d):
                return None
            cid = d[p]
            samp = d[p + 1]
            qid = d[p + 2]
            h = (samp >> 4) & 0xF
            v = samp & 0xF
            items.append({"id": cid, "h": h, "v": v, "q": qid})

        return {"components": items}
    except Exception:
        return None


def get_sof_dimensions(seg: JpegSegment) -> Optional[Tuple[int, int]]:
    """
    SOF segmentinden (en / boy) değerlerini okur.

    Dönen:
        (width, height) veya None
    """
    try:
        d = seg.data
        # Yukarıdaki açıklamaya göre en az 9 byte olmalı
        if len(d) < 9:
            return None

        height = _safe_get_u16_be(d, 5, default=0)
        width = _safe_get_u16_be(d, 7, default=0)
        if width <= 0 or height <= 0:
            return None

        return (width, height)
    except Exception:
        return None