from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Callable, Optional

from PIL import Image


LogFunc = Callable[..., None]


def _is_png_chunk_critical(chunk_type: bytes) -> bool:
    """
    PNG chunk tipinin "kritik" olup olmadığını döndürür.
    PNG standardına göre:
      - İlk bit 0 ise kritik, 1 ise ancillary.
    """
    try:
        return len(chunk_type) == 4 and (chunk_type[0] & 0x20) == 0
    except Exception:
        # Şüpheli durumlarda chunk'ı kritik varsaymak daha güvenli
        return True


def _rebuild_png_with_crc(
    data: bytes,
    log: LogFunc,
    skip_ancillary_on_crc_error: bool,
    drop_bad_critical: bool,
    label: str,
) -> bytes:
    """
    PNG verisini okuyup chunk'ların CRC'lerini yeniden hesaplar.

    Parametreler:
        skip_ancillary_on_crc_error:
            True ise, CRC hatalı ancillary chunk'lar tamamen atlanır.
            False ise, ancillary chunk'ların CRC'si düzeltilerek korunur.
        drop_bad_critical:
            True ise, CRC hatalı kritik chunk görüldüğünde:
                - Chunk atlanır ve sonrasında gelen chunk'lar işlenmez (erken kes).
            False ise, kritik chunk'ların CRC'si düzeltilerek korunur.

    Geri dönüş:
        Yeni oluşturulmuş PNG byte dizisi. Hatalı yapı varsa eldeki en iyi versiyon.
    """
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        # Çağıran zaten kontrol etmeli ama extra güvenlik
        return data

    pos = 8  # PNG signature sonrası
    length_data = len(data)
    new_data = bytearray()
    new_data.extend(data[:8])

    while pos + 8 <= length_data:
        # Kalan veri chunk başlığı için yeterli mi?
        if pos + 8 > length_data:
            break

        # Chunk başlık
        try:
            length = struct.unpack(">I", data[pos:pos + 4])[0]
        except struct.error:
            break

        chunk_type = data[pos + 4:pos + 8]

        # Chunk verisi ve CRC alanı için yeterli veri var mı?
        if pos + 8 + length + 4 > length_data:
            # Eksik chunk, artık ilerlemek mantıklı değil
            break

        chunk_data = data[pos + 8:pos + 8 + length]
        crc_read = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]

        # Yeni CRC hesapla
        crc_calc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF

        if crc_calc != crc_read:
            is_critical = _is_png_chunk_critical(chunk_type)
            chunk_name = chunk_type.decode("latin-1", errors="replace")

            if not is_critical:
                # Ancillary chunk
                if skip_ancillary_on_crc_error:
                    log(
                        f"[PNG-CRC][{label}] CRC hatalı ancillary chunk {chunk_name} -> SKIP edildi.",
                        color="orange",
                    )
                    pos += 12 + length
                    continue
                else:
                    log(
                        f"[PNG-CRC][{label}] CRC hatalı ancillary chunk {chunk_name} -> CRC düzeltildi.",
                        color="orange",
                    )
            else:
                # Kritik chunk
                if drop_bad_critical:
                    log(
                        f"[PNG-CRC][{label}] CRC hatalı KRİTİK chunk {chunk_name} -> chunk atlandı, sonrasına devam edilmeyecek.",
                        color="red",
                    )
                    # Bu chunk'ı da eklemeden işlemi sonlandır
                    break
                else:
                    log(
                        f"[PNG-CRC][{label}] CRC hatalı KRİTİK chunk {chunk_name} -> CRC düzeltildi.",
                        color="orange",
                    )

        # Bu noktada chunk'ı (gerekirse düzeltilmiş CRC ile) yeni PNG'ye ekle
        new_data.extend(struct.pack(">I", length))
        new_data.extend(chunk_type)
        new_data.extend(chunk_data)
        new_data.extend(struct.pack(">I", crc_calc))

        pos += 12 + length

        # IEND'e geldiysek artık bitirebiliriz
        if chunk_type == b"IEND":
            break

    return bytes(new_data)


def fix_with_png_crc(
    input_path: Path,
    output_dir: Path,
    log: LogFunc,
    skip_ancillary_on_crc_error: bool = False,
) -> Optional[Path]:
    """
    PNG dosyasındaki CRC hatalarını düzeltir ve dosyayı yeniden kaydeder.

    Strateji:
      1) "Normal" mod:
            - CRC hatalı chunk'ların CRC'si düzeltilir.
            - İstenirse ancillary chunk'lar hatalı ise atlanabilir.
            - Çıktı Pillow ile test edilir.
      2) Eğer 1. adım başarısız olursa:
            - "Agresif" mod:
                * Tüm CRC hatalı ancillary chunk'lar atlanır.
                * CRC hatalı kritik chunk görüldüğünde akış kesilir.
            - Yeni çıktı tekrar Pillow ile test edilir.

    Başarılı olursa çıktı dosyasının Path'i döner, aksi halde None.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(input_path, "rb") as f:
            data = f.read()

        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            log(f"[PNG-CRC] {input_path.name} geçerli PNG değil.", color="orange")
            return None

        name = input_path.stem
        out_path = output_dir / f"{name}_fixed_pngcrc.png"

        # 1) Normal mod
        new_data = _rebuild_png_with_crc(
            data,
            log=log,
            skip_ancillary_on_crc_error=skip_ancillary_on_crc_error,
            drop_bad_critical=False,
            label="NORMAL",
        )

        with open(out_path, "wb") as f:
            f.write(new_data)

        try:
            with Image.open(out_path) as im:
                im.load()
            log(f"[PNG-CRC] OK (NORMAL) {input_path.name} -> {out_path.name}", color="green")
            return out_path
        except Exception as e:
            log(
                f"[PNG-CRC] NORMAL mod çıktısı yüklenemedi -> {e}. Agresif mod denenecek.",
                color="orange",
            )

        # 2) Agresif mod: ancillary'leri atla, kritik hatalarda akışı kes
        aggr_path = output_dir / f"{name}_fixed_pngcrc_aggr.png"
        new_data_aggr = _rebuild_png_with_crc(
            data,
            log=log,
            skip_ancillary_on_crc_error=True,
            drop_bad_critical=True,
            label="AGGR",
        )

        with open(aggr_path, "wb") as f:
            f.write(new_data_aggr)

        try:
            with Image.open(aggr_path) as im:
                im.load()
            log(f"[PNG-CRC] OK (AGGR) {input_path.name} -> {aggr_path.name}", color="darkgreen")
            return aggr_path
        except Exception as e:
            log(f"[PNG-CRC] AGGR mod çıktısı yüklenemedi -> {e}", color="red")
            return None

    except Exception as e:
        log(f"[PNG-CRC] HATA {input_path.name} -> {e}", color="red")
        return None