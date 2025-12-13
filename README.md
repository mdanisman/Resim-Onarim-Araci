Bozuk veya kısmen hasar görmüş JPEG ve PNG dosyalarını onaran, profesyonel algoritmalarla güçlendirilmiş masaüstü uygulaması. Tek bir akıcı arayüzde birden fazla onarım tekniğini ve gelişmiş skorlama sistemini bir araya getirir.

## 🗺️ İçindekiler
- [Temel Özellikler](#temel-ozellikler)
- [Akıllı Skorlama Sistemi](#akilli-skorlama-sistemi)
- [Arayüz ve Kullanılabilirlik](#arayuz-ve-kullanilabilirlik)
- [Kurulum ve Çalıştırma](#kurulum-ve-calistirma)
- [FFmpeg Kurulumu](#ffmpeg-kurulumu)
- [Geliştirici Rehberi](#gelistirici-rehberi)
- [Katkıda Bulunma](#katkida-bulunma)

## ✨ Temel Özellikler <a id="temel-ozellikler"></a>
**JPEG Onarımı**
- Marker Onarımı: Yanlış yerleştirilmiş SOI (Start of Image) ve EOI (End of Image) işaretleyicilerini düzeltir.
- Smart Header V3: Bozuk JPEG dosyalarında DQT / DHT tablolarını referans bir header veya dinamik bir Header Kütüphanesi kullanarak yeniden inşa eder.
- Partial Top Recovery: Dosyanın üst kısmındaki veri kayıplarını farklı oranlarda deneyerek kurtarır.
- EXIF Thumbnail'den Kurtarma: Dosya içindeki küçük EXIF önizleme resmini çıkarır, isteğe bağlı olarak büyütür (Upscale).
- Gömülü JPEG Taraması: Gizli veya gömülü JPEG verilerini tarayıp çıkarır.

**PNG Onarımı**
- PNG CRC Düzeltme: CRC (Cyclic Redundancy Check) hatalarını normal veya agresif (AGGR) modda düzeltir.
- Ancillary Chunk Atlama: Hatalı ek (ancillary) veri bloklarını isteğe bağlı atlayarak onarım sağlar.

**Genel Yöntemler ve Dönüştürme**
- Pillow ile Yeniden Kaydetme: Basit format hatalarını düzeltmek için dosyayı yeniden yazar.
- PNG Roundtrip: Dosyayı geçici olarak PNG'ye çevirip geri döndürerek bozulmaları temizler.
- FFmpeg ile Yeniden Encode: FFmpeg kuruluysa JPEG/PNG için farklı kalite ön ayarlarıyla yeniden kodlar.

## 💡 Akıllı Skorlama Sistemi <a id="akilli-skorlama-sistemi"></a>
Onarılan her çıktı, en iyi sonucu önermek için otomatik analizden geçer:
- Detay / Entropi Analizi
- Keskinlik Tahmini
- Gri Oranı (tek tonlu veya bozulmuş görüntü tespiti)
- Kırpılmış Veri İhtimali (truncation)
- Çözünürlük ve dosya boyutu dengesi

## ⚙️ Arayüz ve Kullanılabilirlik <a id="arayuz-ve-kullanilabilirlik"></a>
Tüm ayarlar tek pencerede toplanır:
- Toplu işlem: Tek dosya veya klasör bazında onarım
- Hızlı önizleme: Orijinal ve en iyi onarım çıktısını yan yana gösterme
- Log yönetimi: Kayıtları anlık görüntüleme, TXT/CSV'ye aktarma
- Çıktı klasörü: Onarılan dosyaları otomatik `repaired_images` alt klasörüne veya özel klasöre kaydetme

## ⬇️ Kurulum ve Çalıştırma <a id="kurulum-ve-calistirma"></a>
Python yüklü olmasa bile aşağıdaki adımları izleyerek başlayabilirsiniz.

### 1. 📦 Program Klasörünü Hazırlama
- İndirdiğiniz ZIP dosyasını açın.
- Klasörü sabit bir konuma (ör. Masaüstü veya `C:\Resim Onarım Aracı`) çıkarın.
- `gereksinimler.txt` ve `Kurulum.bat` dosyalarının klasörde olduğundan emin olun.

### 2. 🛠 Tek Tıkla Kurulumu Başlatma
- `Kurulum.bat` dosyasına çift tıklayın ve çalıştırın.
- Komut dosyası Python'u indirir, sistem PATH'ine ekler ve `gereksinimler.txt` içindeki Python kütüphanelerini kurar.
- Kurulum tamamlandığında komut penceresi kapanır.

### 3. ▶️ Programı Çalıştırma
- Program klasörüne dönüp `Baslat.cmd` dosyasına çift tıklayın.
- Resim Onarım Aracı penceresi açılacaktır.

## 🛠 FFmpeg Kurulumu (İsteğe Bağlı ama Önerilir) <a id="ffmpeg-kurulumu"></a>
- Yeniden encode yöntemleri için FFmpeg gereklidir; yoksa uygulama uyarı gösterebilir.
- ffmpeg.org gibi güvenilir bir kaynaktan Windows sürümünü indirip `ffmpeg.exe` dosyasını çıkarın.
- `ffmpeg.exe` dosyasını programın ana klasörüne (ör. `Baslat.cmd` ile aynı konum) kopyalayın.
- Programı yeniden çalıştırdığınızda arayüzde "FFmpeg bulundu" ibaresini göreceksiniz.

## 🧑‍💻 Geliştirici Rehberi <a id="gelistirici-rehberi"></a>
Depodaki dokümantasyon (`README.md`, `CONTRIBUTING.md`, pip uyumlu `gereksinimler.txt`) güncel tutulmalıdır. Aşağıdaki adımlar, CONTRIBUTING.md'deki düzenle tutarlı olacak şekilde özetlenmiştir.

### 1) Sanal ortam oluşturma
- **macOS / Linux**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **Windows (PowerShell)**
  ```powershell
  py -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- Ortam aktifken `python` ve `pip` komutlarını doğrudan kullanabilirsiniz.

### 2) Bağımlılıkları yükleme (pip)
- Pip'i güncelleyin ve gereksinimleri kurun:
  ```bash
  python -m pip install --upgrade pip
  python -m pip install -r gereksinimler.txt
  ```
- Dosya adı Türkçe olsa da `-r gereksinimler.txt` parametresiyle doğrudan kullanılabilir.
- **Windows:** Proje klasöründe `py -m pip install -r gereksinimler.txt`
- **macOS / Linux:** Proje klasöründe `python3 -m pip install -r gereksinimler.txt`

### 3) Conda ile çalışma
- Pip içeren bir ortam açın ve aynı `gereksinimler.txt` dosyasını kullanın:
  ```bash
  conda create -n resim-onarim python=3.10 pip
  conda activate resim-onarim
  pip install -r gereksinimler.txt
  ```

### 4) Uygulamayı çalıştırma ve test etme
- Arayüzü başlatmak için: `python main.py`
- GUI'yi modül olarak çalıştırmak için: `python -m gui`
- Otomatik testleri çalıştırmak için: `python -m unittest discover`
- `tests/` dizininde kapsamlı test yoksa komut kısa sürede tamamlanır.

## 🤝 Katkıda Bulunma <a id="katkida-bulunma"></a>
- Geri bildirim, hata raporu veya yeni özellik önerilerinizi Issue açarak veya Pull Request göndererek paylaşabilirsiniz.
- Sürece dair detaylar için [CONTRIBUTING.md](CONTRIBUTING.md) dosyasına bakın.
