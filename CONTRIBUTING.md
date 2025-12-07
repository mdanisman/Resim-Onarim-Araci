# Katkı Rehberi

Projeye katkıda bulunmak için aşağıdaki adımları takip edebilirsiniz.

## Dal (branch) stratejisi
- `main` dalı her zaman yayınlanabilir durumda tutulur.
- Her değişiklik için konuya uygun bir dal açın (ör. `feature/arayuz-iyilestirme`, `bugfix/jpeg-header`).
- Dalınızı güncel tutmak için düzenli olarak `main` ile rebase edin veya günceli çekin.

## Kod stili
- Python kodu için PEP 8'e uyun; mümkünse `black` ve `isort` ile formatlayın.
- Fonksiyon ve sınıf isimlerinde açıklayıcı Türkçe/İngilizce kullanımı tercih edin; yorum satırlarını kısa ve işlevsel tutun.
- Yeni bağımlılık ekliyorsanız `gereksinimler.txt` dosyasını güncelleyin ve yeni sürümlerin proje ile uyumlu olduğundan emin olun.

## Test ve doğrulama
- GUI'yi yerel olarak kontrol etmek için `python main.py` komutuyla uygulamayı çalıştırın.
- Otomatik testler eklediyseniz veya mevcutsa `python -m unittest discover` ile çalıştırarak PR açmadan önce doğrulayın.
- Büyük değişiklikler için ilgili fonksiyonellikleri manuel olarak kontrol edip PR açıklamasında özetleyin.

## Pull Request açarken
- Yaptığınız değişikliği, motivasyonunu ve test sonuçlarını kısaca özetleyin.
- Kod incelemesini kolaylaştırmak için küçük ve odaklı PR'lar açmaya özen gösterin.
- README.md, CONTRIBUTING.md veya gereksinimler.txt gibi dokümantasyon dosyalarını güncellediyseniz PR açıklamasında kısaca
  belirtin.
