@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: --- Yapılandırma ---
SET PYTHON_VERSION=3.12.3
SET PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe
SET INSTALLER_NAME=python_installer.exe
SET REQUIREMENTS_FILE=gereksinimler.txt

TITLE Resim Onarim Araci - Kurulum Sihirbazi

:: ------------------------------------------------------------------
:: GİRİŞ EKRANI (SETUP STİLİ)
:: ------------------------------------------------------------------
cls
echo.
echo  ==================================================================
echo   RESIM ONARIM ARACI - KURULUM SIHIRBAZI
echo  ==================================================================
echo.
echo   Gelistirici :  Muharrem DANISMAN
echo   Iletisim    :  mdanisman3@gmail.com
echo   Telif Hakki :  ^(C^) 2025  Muharrem DANISMAN - Tum haklari saklidir.
echo.
echo  Bu sihirbaz, Resim Onarim Araci'nin calisabilmesi icin gereken:
echo    - Python 3 (Kurulu degilse)
echo    - Gerekli Python kutuphaneleri (Pillow, vb.)
echo    - Baslat.cmd dosyasinin olusturulmasi
echo  adimlarini gerceklestirecek.
echo.
echo  Devam etmek icin bir tusa basin...
echo  (Kurulumu iptal etmek icin pencereyi kapatabilirsiniz.)
echo  ------------------------------------------------------------------
pause >nul

:: ------------------------------------------------------------------
:: 1) Python var mi? (Adim 1/3)
:: ------------------------------------------------------------------
cls
echo.
echo  ==================================================================
echo   ADIM 1/3 - PYTHON DENETIMI
echo  ==================================================================
echo.
echo  Sisteminizde Python komutunun varligi kontrol ediliyor...
echo.

set "PYTHON_EXE="

for /f "delims=" %%I in ('where python 2^>nul') do (
    set "PYTHON_EXE=%%I"
    goto :FOUND_PY
)

:NOT_FOUND_PY
echo  Python bulunamadi. Bu programin calismasi icin Python gerekli.
echo.
echo  Python %PYTHON_VERSION% surumu indirilecek ve sessizce kurulacak.
echo.
echo  Devam etmek icin bir tusa basin...
pause >nul

echo.
echo  Internet baglantisi kontrol ediliyor...
ping www.python.org -n 1 -w 1000 >nul
if %errorlevel% neq 0 (
    echo.
    echo  HATA: Internet baglantisi kurulamadi.
    echo  Lutfen baglantiyi kontrol edip tekrar deneyin.
    echo.
    pause
    EXIT /B 1
)

echo.
echo  Python yukleyicisi indiriliyor...
echo  Kaynak: %PYTHON_INSTALLER_URL%
echo.

bitsadmin /transfer "DownloadPython" /priority HIGH %PYTHON_INSTALLER_URL% "%TEMP%\%INSTALLER_NAME%"
if %errorlevel% neq 0 (
    echo.
    echo  HATA: Python yukleyicisi indirilemedi.
    echo  URL'yi veya baglantiyi kontrol edin.
    echo.
    pause
    EXIT /B 1
)

echo.
echo  Python kurulumu baslatiliyor. Bu islem birkac dakika surebilir...
start /wait "%TEMP%\%INSTALLER_NAME%" /quiet InstallAllUsers=1 PrependPath=1
if %errorlevel% neq 0 (
    echo.
    echo  HATA: Python kurulumu basarisiz oldu veya iptal edildi.
    echo.
    pause
    EXIT /B 1
)

echo.
echo  Python kuruldu. PATH guncellenmesi icin kisa bir sure bekleniyor...
timeout /t 5 >nul

for /f "delims=" %%I in ('where python 2^>nul') do (
    set "PYTHON_EXE=%%I"
    goto :FOUND_PY
)

echo.
echo  HATA: Python kuruldu ancak 'python' komutu hala bulunamiyor.
echo  Lutfen sistemi yeniden baslattiktan sonra bu kurulumu tekrar calistirin.
echo.
pause
EXIT /B 1

:FOUND_PY
echo  Python bulundu: "!PYTHON_EXE!"
echo.
echo  Adim 1/3 basariyla tamamlandi.
echo.
echo  Devam etmek icin bir tusa basin...
pause >nul

:: ------------------------------------------------------------------
:: 2) Gerekli kutuphaneleri yukle (Adim 2/3)
:: ------------------------------------------------------------------
cls
echo.
echo  ==================================================================
echo   ADIM 2/3 - PYTHON KUTUPHANELERININ KURULUMU
echo  ==================================================================
echo.
echo  Bu adimda, programin calmasi icin gereken Python paketleri
echo  (Pillow ve digerleri) yuklenecektir.
echo.

if not exist "%~dp0%REQUIREMENTS_FILE%" (
    echo  HATA: "%REQUIREMENTS_FILE%" dosyasi bu klasorde bulunamadi:
    echo    %~dp0
    echo  Lutfen gereksinimler.txt dosyasinin dogru yerde oldugundan emin olun.
    echo.
    pause
    EXIT /B 1
)

echo  pip guncelleniyor...
"%PYTHON_EXE%" -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo  UYARI: pip guncellenemedi. Yine de devam edilmeye calisiliyor...
)

echo.
echo  Gerekli paketler yukleniyor. Bu islem internet hizina bagli olarak
echo  birkac dakika surebilir...
echo.
"%PYTHON_EXE%" -m pip install -r "%~dp0%REQUIREMENTS_FILE%"
if %errorlevel% neq 0 (
    echo.
    echo  HATA: Paket kurulumu tamamlanamadi.
    echo  Lutfen yukaridaki hata mesajlarini kontrol edin.
    echo.
    pause
    EXIT /B 1
)

echo.
echo  Adim 2/3 basariyla tamamlandi.
echo.
echo  Devam etmek icin bir tusa basin...
pause >nul

:: ------------------------------------------------------------------
:: 3) Baslat.cmd olustur (Adim 3/3)
:: ------------------------------------------------------------------
cls
echo.
echo  ==================================================================
echo   ADIM 3/3 - BASLAT.CMD OLUSTURMA
echo  ==================================================================
echo.
echo  Bu adimda, programi kolayca baslatmaniz icin 'Baslat.cmd'
echo  dosyasi olusturulacaktir.
echo.

(
    echo @echo off
    echo REM Resim Onarim Araci - Baslatma Dosyasi
    echo REM Gelistirici : Muharrem DANIŞMAN ^(mdanisman3@gmail.com^)
    echo REM ^(C^) 2025 - Tum haklari saklidir.
    echo cd /d "%%~dp0"
    echo title Resim Onarim Araci
    echo "%PYTHON_EXE%" "gui.py"
    echo echo.
    echo echo Programi kapatmak icin bir tusa basin...
    echo pause ^>nul
) > "%~dp0Baslat.cmd"

if %errorlevel% neq 0 (
    echo.
    echo  HATA: Baslat.cmd dosyasi olusturulamadi.
    echo.
    pause
    EXIT /B 1
)

echo.
echo  Adim 3/3 basariyla tamamlandi.
echo.
echo  ==================================================================
echo                 KURULUM BASARIYLA TAMAMLANDI!
echo  ==================================================================
echo.
echo   Artik programi baslatmak icin:
echo      -> Bu klasorde 'Baslat.cmd' dosyasina cift tiklayabilirsiniz.
echo.
echo   FFmpeg destegi icin:
echo      -> ffmpeg.exe dosyasini da ayni klasore kopyalayin.
echo         (Varligi otomatik olarak algilanacaktir.)
echo.
echo   Gelistirici : Muharrem DANISMAN
echo   Iletisim    : mdanisman3@gmail.com
echo   Telif Hakki : ^(C^) 2025 - Tum haklari saklidir.
echo.
pause

ENDLOCAL
EXIT /B 0