@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: ============================================================================
:: AYARLAR
:: ============================================================================
set PY_VERSION=3.10.11
set PY_INSTALLER_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.exe
set PY_INSTALLER=python310_setup.exe
set REQUIREMENTS_FILE=gereksinimler.txt

set MODEL_DIR=models
set GFPGAN_DRIVE=https://drive.google.com/uc?export^=download^&id=1plO391KI_tFMAudOlHhQooR_LGjPF_jW
set ESRGAN_DRIVE=https://drive.google.com/uc?export^=download^&id=1o0gpcvfVbnD2gdM5qkVcd04YO-Nlkm6e
set ESRNET_DRIVE=https://drive.google.com/uc?export^=download^&id=15njh9lkvdBgBuHx24LrsTDT7YyFKUtGm

mode con: cols=78 lines=32
color 1F
title Resim Onarim Araci - Kurulum Sihirbazi ^| Muharrem DANISMAN

cls
echo.
echo  ===========================================================================
echo                         RESIM ONARIM ARACI - KURULUM SIHIRBAZI
echo  ===========================================================================
echo   Bu sihirbaz aracin calismasi icin gerekli tum bagimliliklari yukler.
echo.
echo   - Python 3.10 kontrolu / kurulumu (AI MODEL UYUMU ICIN ZORUNLU)
echo   - Temel kutuphaneler: Pillow, numpy, piexif, opencv-python (gereksinimler.txt)
echo   - AI kutuphaneleri: torch, torchvision, basicsr, facexlib, gfpgan, realesrgan
echo   - Stable Diffusion: diffusers, transformers
echo.
echo   Python zaten varsa tekrar kurulmaz.
echo  ===========================================================================
echo.
pause

:: ============================================================================
:: ADIM 1 - PYTHON 3.10 KONTROLU
:: ============================================================================
cls
echo.
echo  ===========================================================================
echo                      ADIM 1 / 4 - PYTHON KONTROLU
echo  ===========================================================================

set PY310_EXE=

if exist "C:\Program Files\Python310\python.exe" set PY310_EXE=C:\Program Files\Python310\python.exe
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set PY310_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe

if defined PY310_EXE (
    echo  Python 3.10 bulundu: %PY310_EXE%
    pause
    goto STEP2
)

echo  Python 3.10 bulunamadi → indiriliyor...
powershell -Command "Invoke-WebRequest '%PY_INSTALLER_URL%' -OutFile '%PY_INSTALLER%'" 2>nul

if %errorlevel% neq 0 (
    echo HATA: Python indirilemedi!
    pause
    exit /b 1
)

echo  Python kuruluyor...
%PY_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

if exist "C:\Program Files\Python310\python.exe" set PY310_EXE=C:\Program Files\Python310\python.exe
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set PY310_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe

if not defined PY310_EXE (
    echo HATA: Python kuruldu fakat bulunamadi!
    pause
    exit /b 1
)

echo  Python 3.10 kuruldu.
pause



:: ============================================================================
:: ADIM 2 - BAGIMLILIK YUKLEME
:: ============================================================================
:STEP2
cls
echo.
echo  ===========================================================================
echo                ADIM 2 / 4 - BAGIMLILIKLAR YUKLENIYOR
echo  ===========================================================================
echo   Yuklenecek kutuphaneler:
echo   Pillow, numpy, piexif, opencv-python
echo   torch + torchvision (CPU)
echo   basicsr, facexlib, gfpgan, realesrgan
echo   diffusers, transformers
echo  ===========================================================================

echo.
echo  pip guncelleniyor...
"%PY310_EXE%" -m pip install --upgrade pip

echo.
echo  Torch (CPU) yukleniyor...
"%PY310_EXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-warn-script-location
if %errorlevel% neq 0 (
    echo HATA: Torch yuklenemedi!
    pause
    exit /b 1
)

echo.
echo  Diger gereksinimler yukleniyor...
"%PY310_EXE%" -m pip install -r "%REQUIREMENTS_FILE%" --no-warn-script-location

if %errorlevel% neq 0 (
    echo HATA: gereksinimler.txt yuklenirken hata olustu!
    pause
    exit /b 1
)

echo.
echo  => Tum bagimliliklar yuklendi.
pause
goto STEP3



:: ============================================================================
:: ADIM 3 - GOOGLE DRIVE MODEL DOSYASI INDIRME
:: ============================================================================
:STEP3
cls
echo.
echo  ===========================================================================
echo                 ADIM 3 / 4 - MODEL DOSYALARI INDIRILIYOR
echo  ===========================================================================

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

:: ---- Google Drive Doğrudan İndirme Linkleri ----
set GFPGAN_DRIVE=https://drive.google.com/uc?export^=download^&id=1plO391KI_tFMAudOlHhQooR_LGjPF_jW
set ESRGAN_DRIVE=https://drive.google.com/uc?export^=download^&id=1o0gpcvfVbnD2gdM5qkVcd04YO-Nlkm6e
set ESRNET_DRIVE=https://drive.google.com/uc?export^=download^&id=15njh9lkvdBgBuHx24LrsTDT7YyFKUtGm

echo.
echo  GFPGAN indiriliyor...
powershell -Command "Invoke-WebRequest '%GFPGAN_DRIVE%' -OutFile '%MODEL_DIR%\GFPGANv1.3.pth'" 2>nul
if %errorlevel% neq 0 (
    echo  HATA: GFPGAN indirilemedi!
    pause
    exit /b 1
)

echo  RealESRGAN_x4plus indiriliyor...
powershell -Command "Invoke-WebRequest '%ESRGAN_DRIVE%' -OutFile '%MODEL_DIR%\RealESRGAN_x4plus.pth'" 2>nul
if %errorlevel% neq 0 (
    echo  HATA: RealESRGAN indirilemedi!
    pause
    exit /b 1
)

echo  RealESRNet_x4plus indiriliyor...
powershell -Command "Invoke-WebRequest '%ESRNET_DRIVE%' -OutFile '%MODEL_DIR%\RealESRNet_x4plus.pth'" 2>nul
if %errorlevel% neq 0 (
    echo  HATA: RealESRNet indirilemedi!
    pause
    exit /b 1
)

echo.
echo  => Tum modeller Drive'dan basariyla indirildi.
pause
goto STEP4


:: ============================================================================
:: ADIM 4 - BASLAT.CMD OLUSTURMA
:: ============================================================================
:STEP4
cls
echo.
echo  Baslat.cmd olusturuluyor...

(
echo @echo off
echo cd /d "%%~dp0"
echo "%PY310_EXE%" "main.py"
echo pause
) > Baslat.cmd

echo.
echo  ===========================================================================
echo                   KURULUM BASARIYLA TAMAMLANDI!
echo  ===========================================================================
echo  Programi calistirmak icin: Baslat.cmd
echo.
pause

ENDLOCAL
exit /b 0
