Gelişmiş algoritmalarla bozuk, açılmayan veya hatalı JPEG/PNG görüntüleri kurtaran profesyonel bir masaüstü uygulaması.
Bu yazılım; marker düzeltme, Smart Header V3, EXIF thumbnail kurtarma, PNG CRC onarımı, FFmpeg yeniden encode ve çok katmanlı skor mekanizması gibi modern teknikleri bir arada sunar.
🚀 Özellikler
🔧 Temel Onarım Özellikleri

JPEG marker tamiri (SOI/EOI düzeltme)

Smart Header V3 — DQT / DHT yeniden inşa

Partial Top Recovery (farklı oranlarla üst kısım kurtarma)

Gömülü JPEG tarama (dosya içinde saklı mini JPG çıkarma)

EXIF thumbnail tabanlı kurtarma (+ isteğe bağlı upscale)

JPEG / PNG FFmpeg yeniden encode

Pillow tabanlı yeniden kaydetme

PNG roundtrip (PNG → Orijinal format)

Gelişmiş PNG CRC tamiri (AGGR mod desteği)

Header Library otomatik seçimi (ortam analizli)

🧠 Akıllı Değerlendirme & Skorlama

Detay/entropi analizi

Keskinlik ölçümü

Truncation tespiti

Gri oranı analizi

Çözünürlük skoru

Boyut + içerik denetimi

Tüm çıktılar otomatik olarak puanlanır ve en iyi sonuç otomatik seçilir.

⚙️ Strateji Modları

SAFE: En hafif ve güvenli teknikler

NORMAL: Dengeli tamir

AGGRESSIVE: En güçlü ve riskli tamir kombinasyonları

🖼️ Yüksek Kalite Önizleme

Orijinal + En iyi onarım karşılaştırmalı inline preview

Ayrı pencerede tam ekran önizleme

📁 Toplu İşlem

Tek dosya

Klasör tarama

İçerik analizi ile gerçek resim dosyalarını bulma

💾 Log & Çıkış

Zaman damgalı günlük kaydı

TXT / CSV log export

Otomatik çıktı klasörü oluşturma

📦 Kurulum
✔ Gereksinimler

Python 3.10+

Pillow

FFmpeg (opsiyonel, kalite artırır)

✔ Kurulum Komutları
pip install -r requirements.txt

✔ Çalıştırma
python gui.py

📂 Proje Yapısı
project/
│
├── gui.py                # Tkinter arayüzü
├── main.py               # Giriş noktası
├── utils.py              # Yardımcı fonksiyonlar
├── core/
│   ├── repair_engine.py  # Ana onarım motoru
│   ├── jpeg_repair.py    # JPEG özel onarım fonksiyonları
│   ├── jpeg_parser.py    # JPEG segment analizi
│   └── png_repair.py     # PNG CRC onarımı
│
└── README.md

📝 Lisans

Bu proje MIT Lisansı ile lisanslanmıştır.
Tam metin için LICENSE dosyasına bakabilirsiniz.

💬 İletişim

Geliştirici: Muharrem Danışman / +90 545 670 36 62 / mdanisman3@gmail.com
Geliştirme / destek / öneriler için issue açabilirsiniz.

⭐ Desteklemek istersen

Proje hoşuna gittiyse GitHub repo’da ⭐ vermen çok değerli olur!
