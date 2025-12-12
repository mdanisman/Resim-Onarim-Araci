from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from core.repair_engine import repair_image_all_methods
from core.ai_patch import apply_ai_reconstruction_to_outputs  # YENİ


@dataclass
class ProcessingOptions:
    # Mevcut alanlar
    use_pillow: bool
    use_png_roundtrip: bool
    use_header: bool
    use_marker: bool
    use_ffmpeg: bool
    stop_on_first_success: bool
    use_embed_scan: bool
    use_partial_top: bool
    use_exif_thumb: bool
    use_png_crc: bool
    exif_thumb_upscale: bool
    png_crc_skip_ancillary: bool
    header_size: int
    ffmpeg_cmd: Optional[str]
    ffmpeg_quality: str
    ref_header_bytes: Optional[bytes]
    header_library: Optional[List[bytes]]
    keep_apps: bool
    keep_com: bool
    resolve_output_dir: Callable[[Path], Optional[Path]]
    log: Callable[[str, str], None]

    # YENİ: AI JPEG Patch Reconstruction seçenekleri
    use_ai_patch: bool = False
    ai_damage_threshold: float = 0.7
    ai_use_realesrgan: bool = True
    ai_use_gfpgan: bool = False
    ai_use_inpaint: bool = False
    ai_external_cmd: Optional[str] = None


class RepairService:
    def __init__(
        self,
        on_file_processed: Callable[[Path, List[Path]], None],
        on_finished: Callable[[], None],
        on_error: Callable[[Exception], None],
    ):
        self.on_file_processed = on_file_processed
        self.on_finished = on_finished
        self.on_error = on_error
        self.processing_thread: Optional[threading.Thread] = None
        self.cancel_requested = False
        self.successful_outputs: Dict[Path, List[Path]] = {}
        self.last_processed_file: Optional[Path] = None

    def start(self, files: List[Path], options: ProcessingOptions) -> None:
        self.cancel_requested = False
        self.successful_outputs.clear()
        self.last_processed_file = None

        def worker() -> None:
            try:
                for f in files:
                    if self.cancel_requested:
                        options.log("İşlem kullanıcı tarafından iptal edildi.", color="red")
                        break

                    out_dir = options.resolve_output_dir(f)
                    if out_dir is None:
                        options.log(
                            f"Çıktı klasörü oluşturulamadığı için atlandı: {f}",
                            color="red",
                        )
                        continue

                    quality_label = options.ffmpeg_quality
                    if quality_label == "Hızlı":
                        q_list = [6, 5]
                    elif quality_label == "Yüksek":
                        q_list = [3, 4, 5]
                    else:
                        q_list = [4, 5]

                    options.log(f"İşleniyor: {f}", color="black")

                    # 1) Klasik yöntemlerle onarım
                    outputs = repair_image_all_methods(
                        input_path=f,
                        base_output_dir=out_dir,
                        ref_header_bytes=options.ref_header_bytes,
                        ffmpeg_cmd=options.ffmpeg_cmd,
                        use_pillow=options.use_pillow,
                        use_png_roundtrip=options.use_png_roundtrip,
                        use_header=options.use_header,
                        use_marker=options.use_marker,
                        use_ffmpeg=options.use_ffmpeg,
                        ffmpeg_qscale_list=q_list,
                        stop_on_first_success=options.stop_on_first_success,
                        header_size=options.header_size,
                        log=options.log,
                        keep_apps=options.keep_apps,
                        keep_com=options.keep_com,
                        header_library=options.header_library,
                        use_embed_scan=options.use_embed_scan,
                        use_partial_top=options.use_partial_top,
                        use_exif_thumb=options.use_exif_thumb,
                        use_png_crc=options.use_png_crc,
                        exif_thumb_upscale=options.exif_thumb_upscale,
                        png_crc_skip_ancillary=options.png_crc_skip_ancillary,
                    )

                    # 2) AI JPEG Patch Reconstruction (sadece JPEG’lerde)
                    try:
                        if (
                            options.use_ai_patch
                            and outputs
                            and f.suffix.lower() in (".jpg", ".jpeg")
                        ):
                            ai_outputs = apply_ai_reconstruction_to_outputs(
                                input_path=f,
                                outputs=outputs,
                                output_dir=out_dir,
                                log=options.log,
                                damage_threshold=options.ai_damage_threshold,
                                use_realesrgan=options.ai_use_realesrgan,
                                use_gfpgan=options.ai_use_gfpgan,
                                use_inpaint=options.ai_use_inpaint,
                            )
                            # AI ile üretilen ek çıktıları da listeye ekle
                            outputs.extend(ai_outputs)
                    except Exception as exc:  # AI tarafı asla tüm işlemi çökertmesin
                        options.log(f"[AI-PATCH] Çalıştırma sırasında hata: {exc}", color="orange")

                    self.successful_outputs[f] = outputs
                    self.last_processed_file = f
                    self.on_file_processed(f, outputs)

                self.on_finished()
            except Exception as exc:  # pragma: no cover - defensive
                options.log(f"İşleme sırasında beklenmeyen hata: {exc}", color="red")
                self.on_error(exc)

        self.processing_thread = threading.Thread(target=worker, daemon=True)
        self.processing_thread.start()

    def cancel(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            self.cancel_requested = True
