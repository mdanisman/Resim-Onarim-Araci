🇹🇷 Türkçe Bilgilendirme

Bozuk, açılmayan veya kısmen hasarlı JPEG/PNG görselleri; marker düzeltme, Smart Header, EXIF thumbnail, PNG CRC tamiri, FFmpeg yeniden encode ve akıllı skorlamayla onarır. Tek resim veya klasör üzerinden toplu işlem, önizleme ve ayrıntılı log imkanı sunar.

## Özellikler

JPEG Onarımı

Marker onarımı (SOI / EOI)

Smart Header V3 (DQT / DHT yenileme)

Header kütüphanesi kullanarak otomatik en uygun header seçme

Partial Top Recovery (dosyanın üst kısmından farklı oranlarla kurtarma)

EXIF thumbnail’den kurtarma (isteğe bağlı büyütme – upscale)

Dosya içindeki gömülü JPEG’leri tarayıp çıkarma

PNG Onarımı

PNG CRC düzeltme (NORMAL + AGGR mod)

Hatalı ancillary chunk’ları isteğe bağlı atlama

Genel Dönüştürme Yöntemleri

Pillow ile yeniden kaydetme

PNG roundtrip (PNG’ye çevir → tekrar orijinal formata döndür)

FFmpeg ile yeniden encode (JPEG/PNG için farklı kalite preset’leri)

Akıllı Skorlama

Detay/entropi analizi

Keskinlik tahmini

Gri oranı (tek ton / bozulmuş görüntü tespiti)

Truncation / kırpılmış veri tespiti

Çözünürlük ve dosya boyutunu birlikte değerlendirerek en iyi çıktıyı otomatik seçme

Arayüz & Kullanılabilirlik

Tek dosya ve klasör bazlı toplu onarım

Orijinal / En iyi onarım yan yana hızlı önizleme

Ayrı pencerede büyük önizleme

TXT / CSV log kaydı

Otomatik çıktı klasörü (örneğin repaired_images alt klasörü)

Gereksinimler

Bu uygulamayı çalıştırmak için:

İşletim sistemi:

Windows 10 / 11 (tavsiye edilen)

veya macOS / Linux (Python yüklü ise çalışır)

Yazılım gereksinimleri:

Python 3.10 veya üzeri

Python paketleri (Pillow vb.) – requirements.txt ile kurulacak

(İsteğe bağlı fakat tavsiye edilir) FFmpeg

JPEG/PNG yeniden encode kalitesini ve başarı oranını artırır.

Python hiç yüklü olmayan bir kullanıcı için aşağıda tüm adımlar detaylı anlatılıyor.

Kurulum (Adım Adım – Python Bilmeyenler İçin)
1. Python Kurulumu (Windows)

https://www.python.org adresine gidin.

Üst menüden Downloads > Windows kısmına tıklayın.

En güncel Python 3.x sürümünü indirin.

Kurulum sırasında:

“Add Python to PATH” kutusunu mutlaka işaretleyin.

Sonra Install Now ile kurulumu tamamlayın.

Kurulum bittikten sonra:

Başlat > cmd yazarak Komut İstemi’ni açın.

Aşağıdaki komutu yazıp Enter’a basın:

python --version


Örneğin Python 3.11.2 gibi bir çıktı görüyorsanız Python hazır demektir.

Eğer python komutu tanınmıyor hatası alırsanız, kurulumu tekrar yaparken “Add Python to PATH” işaretli olduğundan emin olun veya bilgisayarı yeniden başlatın.

2. FFmpeg Kurulumu (Tavsiye Edilir)

FFmpeg olmadan da program çalışır, ancak bazı onarım yöntemleri (özellikle yeniden encode) devre dışı kalır.

https://ffmpeg.org adresine gidin.

Windows için derlenmiş FFmpeg paketini indirin (çoğu zaman “gpl” veya “release full” paketleri).

İndirilen ZIP’i örneğin C:\ffmpeg klasörüne çıkarın.

C:\ffmpeg\bin klasörünü PATH değişkenine ekleyin:

Başlat menüsünde “Environment Variables” veya “Ortam Değişkenleri” aratın.

System variables bölümünde Path satırını seçip Edit deyin.

New diyerek C:\ffmpeg\bin yolunu ekleyin ve kaydedin.

Komut İstemi’nde test edin:

ffmpeg -version


Sürüm bilgisi görüyorsanız FFmpeg başarıyla eklenmiş demektir.

3. Projeyi İndirme

Bu GitHub deposunu açın.

Sağ üstten Code > Download ZIP diyerek proje dosyalarını indirin.

ZIP dosyasını örneğin C:\image-repair-engine klasörüne çıkarın.

Klasör yapısı kabaca şöyle olacaktır:

image-repair-engine/
│
├── gui.py
├── main.py
├── utils.py
├── requirements.txt
├── core/
│   ├── repair_engine.py
│   ├── jpeg_repair.py
│   ├── jpeg_parser.py
│   └── png_repair.py
└── screenshots/   (isteğe bağlı, ekran görüntüleri için)


Not: Bazı versiyonlarda core klasörü yerine bu dosyalar kök dizinde olabilir. README’deki isimler genel referanstır.

4. Gerekli Python Paketlerini Kurma

Komut İstemi’ni açın.

Proje klasörüne gidin:

cd C:\image-repair-engine


(İsteğe bağlı ama tavsiye edilen) sanal ortam oluşturun:

python -m venv venv
venv\Scripts\activate


Komut satırının başında (venv) görürseniz aktif olmuş demektir.

Gerekli paketleri yükleyin:

pip install -r requirements.txt


Bu adımda Pillow, varsa piexif ve projenin ihtiyaç duyduğu diğer kütüphaneler kurulacaktır.

5. Uygulamayı Çalıştırma

Hâlâ proje klasöründeyken (ve sanal ortam açıkken):

python gui.py


Bir pencere açılacak ve Image Repair Engine Pro arayüzü gelecektir.

Eğer pencere açılmıyorsa veya hata alırsanız, Komut İstemi’ndeki hatayı kopyalayıp issue olarak paylaşabilirsiniz.

Ayarlar

Arayüz içinde (tasarımına göre değişebilir ama genel olarak):

Çıktı Klasörü Ayarları

Onarılan dosyaların:

Aynı klasör altında repaired_images gibi bir alt klasöre

veya sizin belirlediğiniz özel bir klasöre kaydedilmesini seçebilirsiniz.

Onarım Yöntemleri

JPEG için:

Marker onarımı

Smart Header V3

Partial Top Recovery

Gömülü JPEG arama

EXIF thumbnail’den geri kazanım (+ büyütme)

Genel:

Pillow ile yeniden kaydet

PNG roundtrip

FFmpeg yeniden encode (Hızlı / Normal / Yüksek)

Smart Header & Header Library

Referans JPEG dosyası seçip onun header’ını kullanabilirsiniz.

Bir klasörden header kütüphanesi oluşturup bozuk görsele en uygun header’ın otomatik seçilmesini sağlayabilirsiniz.

APP (EXIF, IPTC vb.) ve COM segmentlerini koruyup korumamayı seçebilirsiniz.

PNG Onarım Ayarları

PNG CRC onarımını açıp kapatabilirsiniz.

Hatalı ancillary chunk’ların tamamen atlanmasını tercih edebilirsiniz.

Örnek Onarım Senaryosu

Programı çalıştırın:

python gui.py


Çıktı ayarlarından:

“Girdi klasörüne alt klasör oluştur” veya

“Özel çıktı klasörü kullan” seçeneklerinden birini seçin.

Ayarlar sekmesinden:

JPEG için Smart Header, Marker, EXIF thumbnail gibi yöntemleri aktif edin.

FFmpeg kurulmuşsa FFmpeg ile Yeniden Encode özelliğini açın.

Tek Resim veya Klasör modunu seçin:

“Tek Resim Seç ve Onar” ile tek bir bozuk JPG deneyin.

“Klasör Seç ve Tümünü Onar” ile tüm bozuk görselleri taratın.

İşlem devam ederken:

İlerleme çubuğu ve istatistiklerden süreci takip edin.

Log penceresinden her adımın ne yaptığını görebilirsiniz.

Onarım tamamlandığında:

Hızlı önizleme alanında Orijinal ve En İyi Onarım yan yana gösterilir.

Çıktıları inceleyip repaired_images klasöründen kullanabilirsiniz.

🇬🇧 English İnformation
Overview

Summary:
A desktop application that recovers damaged, corrupted, or unreadable JPEG/PNG images using Smart Header reconstruction, marker fixing, EXIF thumbnail recovery, PNG CRC repair, FFmpeg re-encode and a multi-factor intelligent scoring system. Supports both single-file and batch folder processing with inline preview and detailed logging.

Features

JPEG Repair

Marker repair (SOI / EOI)

Smart Header V3 (rebuild DQT / DHT tables)

Automatic header selection from header library

Partial Top Recovery with multiple ratios

EXIF thumbnail–based recovery (optional upscale)

Embedded JPEG extraction from corrupted binary

PNG Repair

PNG CRC fixing (NORMAL + AGGR modes)

Optionally drop corrupted ancillary chunks

General Processing

Resave using Pillow

PNG roundtrip

FFmpeg re-encode for JPEG/PNG (multiple quality presets)

Intelligent Scoring

Entropy / detail analysis

Sharpness estimation

Gray ratio (monotone / broken image detection)

Truncation level

Resolution + file size balance
→ Automatically selects the best output.

UI & Usability

Single-image or folder-based batch processing

Side-by-side preview (Original vs Best Output)

Dedicated preview window

TXT / CSV log export

Automatic output folder management (e.g. repaired_images)

Requirements

Operating System:

Windows 10 / 11 (recommended)

Or macOS / Linux with Python installed

Software:

Python 3.10+

Python packages from requirements.txt

(Optional but recommended) FFmpeg for better re-encoding quality

Installation (Step by Step)
1. Install Python

Go to https://www.python.org.

Download and install the latest Python 3.x for your OS.

On Windows, during setup:

Make sure “Add Python to PATH” is checked.

Verify in a terminal / command prompt:

python --version

2. Install FFmpeg (Optional but Recommended)

Download FFmpeg from https://ffmpeg.org.

Extract it to e.g. C:\ffmpeg (on Windows).

Add C:\ffmpeg\bin to your PATH environment variable.

Verify:

ffmpeg -version

3. Download the Project

Download this repository as ZIP from GitHub.

Extract it to a folder, e.g. C:\image-repair-engine.

Example structure:

image-repair-engine/
├── gui.py
├── main.py
├── utils.py
├── requirements.txt
└── core/
    ├── repair_engine.py
    ├── jpeg_repair.py
    ├── jpeg_parser.py
    └── png_repair.py

4. Install Python Dependencies
cd C:\image-repair-engine

# Optional but recommended: create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

5. Run the Application
python gui.py


A window should appear with the main UI of Image Repair Engine Pro.

Settings

Inside the UI you can configure:

Output Folder:

Store repaired images in a repaired_images subfolder next to the input, or

Use a global custom output folder.

Repair Methods:

Enable/disable JPEG repair strategies:

Marker repair

Smart Header V3

Partial Top Recovery

Embedded JPEG scan

EXIF thumbnail recovery (+ upscale)

General:

Pillow resave

PNG roundtrip

FFmpeg re-encode with different quality levels

Smart Header & Header Library:

Choose a reference JPEG file whose header will be used.

Build a header library from a folder of similar images and let the engine pick the best matching header.

Keep or drop APP (EXIF/IPTC) and COM segments.

PNG Repair:

Enable PNG CRC repair (Normal + Aggressive)

Optionally drop broken ancillary chunks.

Example Repair Workflow

Run:

python gui.py


Configure output mode:

Subfolder near input images, or

Global custom output directory.

Open settings and:

Enable Smart Header V3, marker repair, EXIF thumbnail recovery, etc.

Enable FFmpeg re-encode if FFmpeg is installed.

Choose processing mode:

Single image – “Select and Repair Single Image”

Folder – “Select Folder and Repair All”

Monitor:

Progress bar and statistics

Logs showing each repair attempt and outcome

After completion:

Compare Original vs Best Output in the preview area.

Access repaired files from the output folder.

License

This project is licensed under the MIT License.
See the LICENSE file for details.
