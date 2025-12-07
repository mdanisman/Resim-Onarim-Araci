from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from core.jpeg_repair import build_header_library_from_folder
from core.repair_engine import pick_best_output, repair_image_all_methods
from gui import run_app
from utils import DEST_SUBFOLDER_NAME, detect_ffmpeg, is_image_file


# -------------------------------------------------------
# Basit CLI log fonksiyonu
# -------------------------------------------------------
def _cli_log(message: str, color: str | None = None) -> None:
    """
    Basit CLI loglayıcısı.
    color parametresi şimdilik sadece API uyumu için var, terminalde kullanılmıyor.
    """
    print(message)


# -------------------------------------------------------
# Yöntem listesi parse
# -------------------------------------------------------
def _parse_methods(methods: Optional[str], ffmpeg_available: bool) -> Dict[str, bool]:
    """
    Virgülle ayrılmış yöntem listesini bool bayraklarına çevirir.

    Örnek:
      "pillow,ffmpeg,header"
      "all"
    """

    default_flags: Dict[str, bool] = {
        "pillow": True,
        "png_roundtrip": True,
        "header": False,
        "marker": True,
        "ffmpeg": ffmpeg_available,
        "embed_scan": True,
        "partial_top": True,
        "exif_thumb": True,
        "png_crc": True,
    }

    if not methods:
        return default_flags

    parsed = {m.strip().lower() for m in methods.split(",") if m.strip()}
    if "all" in parsed:
        parsed = set(default_flags.keys())

    flags: Dict[str, bool] = {}
    for name, default_val in default_flags.items():
        if name not in parsed:
            flags[name] = False
        else:
            if name == "ffmpeg":
                flags[name] = ffmpeg_available
            else:
                flags[name] = True

    return flags


# -------------------------------------------------------
# Girdi dosyalarını toplama
# -------------------------------------------------------
def _collect_input_files(input_path: Path) -> List[Path]:
    """
    Girdi tek dosya ise onu, klasör ise içindeki tüm uygun görselleri döner.
    """

    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"Girdi dosyası desteklenen bir görsel değil: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Girdi yolu bulunamadı: {input_path}")

    files: List[Path] = []
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and is_image_file(path):
            files.append(path)
    return files


# -------------------------------------------------------
# Çıktı klasörü hesaplama
# -------------------------------------------------------
def _determine_output_root(input_path: Path, output: Optional[Path]) -> Path:
    """
    Kullanıcı özel çıktı klasörü vermişse onu kullanır,
    yoksa girdi konumuna 'repaired_images' alt klasörü ekler.
    """
    if output:
        return output

    base = input_path.parent if input_path.is_file() else input_path
    return base / DEST_SUBFOLDER_NAME


def _output_dir_for_file(base_output: Path, input_root: Path, file_path: Path) -> Path:
    """
    Klasör girdi senaryosunda alt klasör yapısını koruyarak çıktı klasörü üretir.
    """
    if input_root.is_dir():
        try:
            rel_parent = file_path.parent.relative_to(input_root)
        except ValueError:
            rel_parent = Path()
        return base_output / rel_parent
    return base_output


# -------------------------------------------------------
# CLI Çalıştırma
# -------------------------------------------------------
def run_cli(args: argparse.Namespace) -> int:
    input_path = Path(args.input).resolve()

    ffmpeg_cmd = detect_ffmpeg()
    ffmpeg_available = bool(ffmpeg_cmd)
    method_flags = _parse_methods(args.methods, ffmpeg_available=ffmpeg_available)

    header_bytes: Optional[bytes] = None
    if args.header_file:
        p = Path(args.header_file)
        if not p.is_file():
            raise ValueError(f"Header dosyası bulunamadı: {p}")
        header_bytes = p.read_bytes()[: args.header_size_kb * 1024]

    header_library = None
    if args.header_library:
        lib_path = Path(args.header_library)
        if not lib_path.is_dir():
            raise ValueError(f"Header kütüphanesi klasörü bulunamadı: {lib_path}")
        header_library = build_header_library_from_folder(lib_path)

    if method_flags["ffmpeg"] and not ffmpeg_cmd:
        _cli_log(
            "[WARN] FFmpeg isteniyor ancak sistemde bulunamadı, devre dışı bırakılıyor.",
            color="orange",
        )
        method_flags["ffmpeg"] = False

    files = _collect_input_files(input_path)
    if not files:
        _cli_log("İşlenecek görsel bulunamadı.", color="red")
        return 1

    base_output = _determine_output_root(input_path, args.output)
    base_output.mkdir(parents=True, exist_ok=True)

    q_map = {
        "hizli": [6, 5],
        "yuksek": [3, 4, 5],
        "normal": [4, 5],
    }
    q_list = q_map.get(args.ffmpeg_quality.lower(), [4, 5])

    _cli_log(f"Toplam {len(files)} dosya işlenecek.")
    _cli_log(f"Çıktı kök klasörü: {base_output}")

    success_count = 0

    for idx, file_path in enumerate(files, start=1):
        out_dir = _output_dir_for_file(
            base_output,
            input_path if input_path.is_dir() else input_path.parent,
            file_path,
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        _cli_log(f"[{idx}/{len(files)}] İşleniyor: {file_path}")
        outputs = repair_image_all_methods(
            input_path=file_path,
            base_output_dir=out_dir,
            ref_header_bytes=header_bytes,
            ffmpeg_cmd=ffmpeg_cmd,
            use_pillow=method_flags["pillow"],
            use_png_roundtrip=method_flags["png_roundtrip"],
            use_header=method_flags["header"],
            use_marker=method_flags["marker"],
            use_ffmpeg=method_flags["ffmpeg"],
            ffmpeg_qscale_list=q_list,
            stop_on_first_success=args.stop_on_first_success,
            header_size=args.header_size_kb * 1024,
            log=_cli_log,
            keep_apps=True,
            keep_com=True,
            header_library=header_library,
            use_embed_scan=method_flags["embed_scan"],
            use_partial_top=method_flags["partial_top"],
            use_exif_thumb=method_flags["exif_thumb"],
            use_png_crc=method_flags["png_crc"],
            exif_thumb_upscale=args.exif_thumb_upscale,
            png_crc_skip_ancillary=args.png_crc_skip_ancillary,
            # NOT: strategy_mode parametresi core.repair_engine.repair_image_all_methods
            # imzanızda yoksa bu satırı kaldırmanız gerekir.
            strategy_mode=args.strategy_mode.upper(),
        )

        best = pick_best_output(outputs) if outputs else None
        if best:
            success_count += 1
            _cli_log(f"[OK] En iyi çıktı: {best}", color="green")
        else:
            _cli_log("[FAIL] Başarılı çıktı üretilemedi.", color="red")

    _cli_log(f"\nTamamlandı. Başarılı: {success_count} / {len(files)}")
    return 0 if success_count else 2


# -------------------------------------------------------
# Argüman ayrıştırıcı
# -------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resim Onarım Aracı - GUI veya CLI modunda çalıştırılabilir.",
    )

    parser.add_argument(
        "-i",
        "--input",
        help="Onarılacak dosya veya klasör yolu. Boş bırakılırsa GUI başlatılır.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Çıktı klasörü. Varsayılan: girdiyle aynı konumda 'repaired_images'.",
    )
    parser.add_argument(
        "-m",
        "--methods",
        help=(
            "Virgülle ayrılmış yöntem listesi: "
            "pillow, png_roundtrip, header, marker, ffmpeg, "
            "embed_scan, partial_top, exif_thumb, png_crc, all"
        ),
    )
    parser.add_argument(
        "--header-file",
        dest="header_file",
        help="Smart Header için referans JPEG dosyası (opsiyonel).",
    )
    parser.add_argument(
        "--header-library",
        dest="header_library",
        help="Header kütüphanesi klasörü (opsiyonel).",
    )
    parser.add_argument(
        "--header-size-kb",
        dest="header_size_kb",
        type=int,
        default=16,
        help="Header denemeleri için kullanılacak bayt miktarı (KB). Varsayılan: 16",
    )
    parser.add_argument(
        "--ffmpeg-quality",
        dest="ffmpeg_quality",
        default="normal",
        choices=["hizli", "normal", "yuksek"],
        help="FFmpeg denemeleri için kalite profili.",
    )
    parser.add_argument(
        "--stop-on-first-success",
        action="store_true",
        help="İlk başarılı çıktıdan sonra diğer yöntemleri durdur.",
    )
    parser.add_argument(
        "--exif-thumb-upscale",
        action="store_true",
        help="EXIF thumbnail onarımında büyütme denemesi yap.",
    )
    parser.add_argument(
        "--png-crc-skip-ancillary",
        action="store_true",
        help="PNG CRC onarımında yardımcı (ancillary) chunk'ları atla.",
    )
    parser.add_argument(
        "--strategy-mode",
        default="NORMAL",
        choices=["SAFE", "NORMAL", "AGGRESSIVE"],
        help="Onarım strateji modu (core.repair_engine ile uyumlu olmalı).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Argümanlara bakılmaksızın GUI'yi başlat.",
    )

    return parser


# -------------------------------------------------------
# main
# -------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # GUI moduna geçiş: --gui verilmişse veya input yoksa
    if args.gui or not args.input:
        run_app()
        return 0

    # CLI modunda çalıştır
    try:
        return run_cli(args)
    except Exception as exc:
        _cli_log(f"[HATA] {exc}", color="red")
        return 1


if __name__ == "__main__":
    sys.exit(main())
