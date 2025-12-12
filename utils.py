from __future__ import annotations

"""
Genel yardımcı fonksiyonlar.

Bu modül, proje genelinde tekrar kullanılabilecek küçük araçları içerir:

- Görsel dosya algılama (uzantı + içerik doğrulama)
- FFmpeg ikili dosyasını tespit etme (Windows / Linux / macOS)
- Ortak sabitler (DEST_SUBFOLDER_NAME vb.)

Uzun vadeli düşünülerek tip ipuçları ile yazılmıştır; böylece IDE desteği,
refactor ve hata yakalama daha sağlıklı hale gelir.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageFile

# Bozuk / eksik son byte'ları olan görsellerde de yüklenebilmeyi dene
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Onarım çıktılarının atıldığı alt klasör adı
DEST_SUBFOLDER_NAME: str = "repaired_images"

# Desteklenen görsel uzantıları (gerekirse genişletilebilir)
# Not: Bazı formatlar (ör. HEIC) için Pillow tarafında ek kütüphane gerekebilir.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
    # ".heic",  # Pillow kurulumu uygunsa açılabiliyorsa aktif edilebilir.
}


PathLike = Union[str, Path]


def normalize_path(path: PathLike) -> Path:
    """
    str veya Path tipindeki bir yolu kesin olarak Path nesnesine çevirir.

    Bu, dışarıdan hem "C:/foo/bar.jpg" hem de Path(...) gelmesi durumunda
    tek tip kullanım sağlar.
    """
    return path if isinstance(path, Path) else Path(path)


def is_image_file(path: PathLike, check_content: bool = False) -> bool:
    """
    Verilen yolun bir resim dosyası olup olmadığını belirler.

    İki katmanlı kontrol:
      1) Uzantı kontrolü (IMAGE_EXTENSIONS set'ine göre)
      2) (Opsiyonel) check_content=True ise Pillow ile dosyayı açmayı deneyerek
         gerçekten resim olup olmadığını doğrular.

    Args:
        path: Kontrol edilecek dosya yolu (str veya Path).
        check_content: True ise, uzantıdan bağımsız olarak dosya içeriğini
                       Pillow ile açmayı dener (yavaş ama daha güvenilir).

    Returns:
        bool: Dosya bir resim olarak kabul ediliyorsa True, değilse False.
    """
    p = normalize_path(path)

    # Önce uzantıdan hızlı bir tahmin
    ext = p.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        # İçerik kontrolü istenmiyorsa direkt kabul
        if not check_content:
            return True

    if check_content:
        # Uzantı uyumlu olmasa bile, içerikten resim olup olmadığını test et
        try:
            with Image.open(p) as im:
                im.verify()  # Yalnızca bütünlük kontrolü, decode etmez
            return True
        except Exception:
            return False

    return False


def get_script_dir() -> Path:
    """
    Çalışan script'in (veya PyInstaller exe'sinin) bulunduğu klasörü döndürür.

    - PyInstaller ile paketlenmiş exe'de sys.executable kullanılır.
    - Normal çalışmada sys.argv[0] üzerinden yol çözülür.

    Herhangi bir hata durumda, mevcut çalışma dizini (cwd) geri dönüş olarak kullanılır.
    """
    try:
        if getattr(sys, "frozen", False):
            # PyInstaller / benzeri paketlenmiş exe
            script_path = Path(sys.executable).resolve()
        else:
            # Normal Python betiği
            script_path = Path(sys.argv[0]).resolve()
        return script_path.parent
    except Exception:
        return Path.cwd()


def detect_ffmpeg() -> Optional[str]:
    """
    Sistemde kullanılabilir bir FFmpeg ikili dosyasını bulmaya çalışır.

    Arama sırası:
      1) PATH üzerinde `ffmpeg` (veya Windows'ta ffmpeg.exe) var mı? -> shutil.which
      2) Script / exe klasöründe yerel `ffmpeg` veya `ffmpeg.exe` var mı?
         (Özellikle Windows için, exe ile aynı klasörde ffmpeg dağıtıldığında)
      3) Bulunamazsa None döner.

    Returns:
        Kullanılabilir FFmpeg komutunun/ikili yolunun string gösterimi veya None.
        Gui tarafında:
            - None ise "FFmpeg bulunamadı" olarak bilgilendirme yapılır.
            - Değilse bu değer subprocess çağrılarında kullanılabilir.
    """
    # 1) PATH üzerinde ara
    in_path = shutil.which("ffmpeg")
    if in_path:
        # Tam yolu döndürmek, daha belirgin log'lar ve hatalar için faydalı
        return in_path

    # 2) Script / exe klasöründe ara
    script_dir = get_script_dir()

    candidates = []

    if os.name == "nt":
        # Windows
        candidates.extend(
            [
                script_dir / "ffmpeg.exe",
                script_dir / "ffmpeg" / "bin" / "ffmpeg.exe",
            ]
        )
    else:
        # Linux / macOS gibi POSIX sistemler
        candidates.extend(
            [
                script_dir / "ffmpeg",
                script_dir / "bin" / "ffmpeg",
            ]
        )

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    # 3) Hiçbir yerde bulunamadı
    return None
