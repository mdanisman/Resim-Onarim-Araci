## 📸 Gelişmiş Resim Onarım Aracı
Bu araç, bozuk veya kısmen hasar görmüş JPEG ve PNG dosyalarını onarmak için geliştirilmiş, profesyonel seviyede algoritmalar kullanan bir masaüstü uygulamasıdır. Tek bir akıcı arayüzde birden fazla onarım tekniğini ve gelişmiş bir çıktı skorlama sistemini birleştirir.

✨ ## Temel Özellikler
JPEG Onarımı
Marker Onarımı: Yanlış yerleştirilmiş SOI (Start of Image) ve EOI (End of Image) işaretleyicilerini düzeltir.

Smart Header V3: Bozuk JPEG dosyalarında DQT / DHT tablolarını (görüntü kalitesini ve renk haritasını belirleyen kritik yapılar) referans bir header veya dinamik bir Header Kütüphanesi kullanarak yeniden inşa eder.

Partial Top Recovery: Dosyanın üst kısmındaki veri kayıplarını farklı oranlarda deneyerek kurtarmaya çalışır.

EXIF Thumbnail'den Kurtarma: Dosya içinde mevcut olan küçük EXIF önizleme resmini çıkarır ve isteğe bağlı olarak büyütür (Upscale).

Gömülü JPEG Taraması: Dosya içindeki gizli veya gömülü JPEG verilerini tarayarak çıkarır.

PNG Onarımı
PNG CRC Düzeltme: CRC (Cyclic Redundancy Check) hatalarını hem normal hem de agresif (AGGR) modda düzelterek veri bütünlüğünü sağlar.

Ancillary Chunk Atlama: Hatalı ek (ancillary) veri bloklarının isteğe bağlı olarak atlanmasıyla onarımı mümkün kılar.

Genel Yöntemler ve Dönüştürme
Pillow ile Yeniden Kaydetme: Basit format hatalarını düzeltmek için popüler Python kütüphanesi Pillow kullanılarak dosyayı yeniden yazar.

PNG Roundtrip: Dosyayı geçici olarak PNG formatına çevirip tekrar orijinal formata döndürerek bozulmaları temizler.

FFmpeg ile Yeniden Encode: Kurulumluysa, güçlü FFmpeg aracını kullanarak dosyayı yeniden kodlar (JPEG/PNG için farklı kalite ön ayarları mevcuttur).

💡 ## Akıllı Skorlama Sistemi
Programın en güçlü özelliği, onarılan her bir çıktıyı analiz eden ve puanlayan Akıllı Skorlama mekanizmasıdır. Bu skorlar sayesinde en iyi onarım sonucu kullanıcıya sunulur.

Detay / Entropi Analizi

Keskinlik Tahmini

Gri Oranı (Tek tonlu veya bozulmuş görüntü tespiti)

Kırpılmış Veri İhtimali (Truncation)

Çözünürlük ve Dosya Boyutu Dengesi

⚙️ Arayüz ve Kullanılabilirlik
Uygulama, tüm ayarları tek bir akıcı pencerede sunar:

Toplu İşlem: Tek bir dosyayı veya komple bir klasörü ve içindeki tüm resimleri onarma.

Hızlı Önizleme: Orijinal dosya ile Akıllı Skorlama ile seçilen En İyi Onarım çıktısını yan yana gösterir.

Log Yönetimi: Tüm işlem kayıtları ve hatalar anlık olarak log penceresinde gösterilir ve TXT / CSV formatında dışa aktarılabilir.

Çıktı Klasörü: Onarılan dosyaların otomatik olarak girdi klasöründe oluşturulan repaired_images alt klasörüne veya özel bir klasöre kaydedilme seçeneği.

⬇️ ## Kurulum ve Çalıştırma



🤝 Katkıda Bulunma
Geri bildirimleriniz, hata raporlarınız ve yeni özellik önerileriniz değerlidir! Lütfen bir Issue açarak veya bir Pull Request göndererek katkıda bulunun.
