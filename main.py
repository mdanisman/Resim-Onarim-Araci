from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

from core.jpeg_repair import build_header_library_from_folder
from core.repair_engine import pick_best_output, repair_image_all_methods
from core.ai_patch import apply_ai_reconstruction_to_outputs
from gui import run_app
from logging_utils import create_logger
from utils import DEST_SUBFOLDER_NAME, detect_ffmpeg, is_image_file


def _write_cli_error_log(exc: Exception) -> Path:
    """Beklenmeyen hataları dosyaya dökerek kullanıcıya yol gösterir."""
    log_path = Path.cwd() / "resim_onarim_cli_error.log"
    try:
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(f"Exception: {exc.__class__.__name__}: {exc}\n\n")
            traceback.print_exc(file=fh)
        return log_path
    except Exception:
        return log_path

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
    operation_id = f"CLI-{uuid4()}"
    logger = create_logger(operation_id=operation_id, step="cli")

    def log(message: str, color: str | None = None, extra: Optional[dict] = None) -> None:
        level = logging.INFO
        if color == "red":
            level = logging.ERROR
        elif color == "orange":
            level = logging.WARNING
        logger.log(level, message, extra=extra or {})

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
        log(
            "[WARN] FFmpeg isteniyor ancak sistemde bulunamadı, devre dışı bırakılıyor.",
            color="orange",
            extra={"step": "ffmpeg-check", "method": "ffmpeg", "result": "skipped"},
        )
        method_flags["ffmpeg"] = False

    files = _collect_input_files(input_path)
    if not files:
        log("İşlenecek görsel bulunamadı.", color="red", extra={"step": "input", "result": "failed"})
        return 1

    base_output = _determine_output_root(input_path, args.output)
    base_output.mkdir(parents=True, exist_ok=True)

    q_map = {
        "hizli": [6, 5],
        "yuksek": [3, 4, 5],
        "normal": [4, 5],
    }
    q_list = q_map.get(args.ffmpeg_quality.lower(), [4, 5])

    # AI seçenekleri
    ai_enabled = bool(args.ai_patch)
    ai_use_realesrgan = not args.ai_no_realesrgan
    ai_use_gfpgan = not args.ai_no_gfpgan
    ai_use_inpaint = not args.ai_no_inpaint
    ai_damage_threshold = float(args.ai_damage_threshold)

    log(
        f"Toplam {len(files)} dosya işlenecek.",
        extra={"step": "process-start", "result": "running"},
    )
    log(
        f"Çıktı kök klasörü: {base_output}",
        extra={"step": "output", "result": "ready"},
    )
    if ai_enabled:
        log(
            f"AI JPEG Patch Reconstruction: AKTİF (Real-ESRGAN={ai_use_realesrgan}, "
            f"GFPGAN={ai_use_gfpgan}, Inpaint={ai_use_inpaint}, "
            f"threshold={ai_damage_threshold})",
            extra={"step": "ai-config", "method": "ai_patch", "result": "enabled"},
        )
    else:
        log(
            "AI JPEG Patch Reconstruction: PASİF",
            extra={"step": "ai-config", "method": "ai_patch", "result": "disabled"},
        )

    success_count = 0
    process_started = time.perf_counter()

    for idx, file_path in enumerate(files, start=1):
        out_dir = _output_dir_for_file(
            base_output,
            input_path if input_path.is_dir() else input_path.parent,
            file_path,
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        log(
            f"[{idx}/{len(files)}] İşleniyor: {file_path}",
            extra={"step": "file-start", "file": str(file_path), "result": "processing"},
        )
        file_started = time.perf_counter()
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
            log=log,
            keep_apps=True,
            keep_com=True,
            header_library=header_library,
            use_embed_scan=method_flags["embed_scan"],
            use_partial_top=method_flags["partial_top"],
            use_exif_thumb=method_flags["exif_thumb"],
            use_png_crc=method_flags["png_crc"],
            exif_thumb_upscale=args.exif_thumb_upscale,
            png_crc_skip_ancillary=args.png_crc_skip_ancillary,
            strategy_mode=args.strategy_mode.upper(),
        )

        # AI JPEG Patch Reconstruction
        if ai_enabled and outputs and file_path.suffix.lower() in (".jpg", ".jpeg"):
            ai_started = time.perf_counter()
            outputs = apply_ai_reconstruction_to_outputs(
                input_path=file_path,
                outputs=outputs,
                output_dir=out_dir,
                use_realesrgan=ai_use_realesrgan,
                use_gfpgan=ai_use_gfpgan,
                use_inpaint=ai_use_inpaint,
                damage_threshold=ai_damage_threshold,
                strategy_mode=args.strategy_mode.upper(),
                log=log,
            )
            log(
                "AI JPEG Patch Reconstruction tamamlandı.",
                extra={
                    "step": "ai-patch",
                    "file": str(file_path),
                    "method": "ai_patch",
                    "result": "success",
                    "duration_ms": int((time.perf_counter() - ai_started) * 1000),
                },
            )
        elif ai_enabled and file_path.suffix.lower() not in (".jpg", ".jpeg"):
            log(
                "AI patch JPEG olmadığı için atlandı.",
                color="orange",
                extra={"step": "ai-patch", "file": str(file_path), "result": "skipped"},
            )
        elif ai_enabled and not outputs:
            log(
                "AI patch atlandı çünkü klasik onarım çıktı üretmedi.",
                color="orange",
                extra={"step": "ai-patch", "file": str(file_path), "result": "skipped"},
            )

        best = pick_best_output(outputs, strategy_mode=args.strategy_mode.upper()) if outputs else None
        if best:
            success_count += 1
            log(
                f"[OK] En iyi çıktı: {best}",
                color="green",
                extra={
                    "step": "file-end",
                    "file": str(file_path),
                    "result": "success",
                    "duration_ms": int((time.perf_counter() - file_started) * 1000),
                },
            )
        else:
            log(
                "[FAIL] Başarılı çıktı üretilemedi.",
                color="red",
                extra={
                    "step": "file-end",
                    "file": str(file_path),
                    "result": "failed",
                    "duration_ms": int((time.perf_counter() - file_started) * 1000),
                },
            )

    total_duration_ms = int((time.perf_counter() - process_started) * 1000)
    log(
        f"\nTamamlandı. Başarılı: {success_count} / {len(files)}",
        extra={"step": "process-end", "result": "finished", "duration_ms": total_duration_ms},
    )
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

    # ---------- AI JPEG PATCH RECONSTRUCTION SEÇENEKLERİ ----------
    parser.add_argument(
        "--ai-patch",
        action="store_true",
        help="Klasik onarım çıktısı üzerine AI JPEG Patch Reconstruction uygula.",
    )
    parser.add_argument(
        "--ai-no-realesrgan",
        action="store_true",
        help="AI patch sırasında Real-ESRGAN adımını devre dışı bırak.",
    )
    parser.add_argument(
        "--ai-no-gfpgan",
        action="store_true",
        help="AI patch sırasında GFPGAN yüz onarımını devre dışı bırak.",
    )
    parser.add_argument(
        "--ai-no-inpaint",
        action="store_true",
        help="AI patch sırasında Stable Diffusion inpainting adımını devre dışı bırak.",
    )
    parser.add_argument(
        "--ai-damage-threshold",
        type=float,
        default=0.7,
        help="Hasar ısı haritası için eşik (0.0-1.0). Varsayılan: 0.7",
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
        logger = create_logger(operation_id="CLI-ERROR", step="cli-error")
        logger.exception(
            "CLI sırasında beklenmeyen hata oluştu.",
            extra={"step": "cli-error", "result": "failed"},
        )
        log_path = _write_cli_error_log(exc)
        logger.error(
            f"Detaylı hata kaydı: {log_path}",
            extra={"step": "cli-error", "result": "failed"},
        )
        return 1  

if __name__ == "__main__":
    sys.exit(main())