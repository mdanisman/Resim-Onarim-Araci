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
echo                       RESIM ONARIM ARACI
echo                         KURULUM SIHIRBAZI
echo  ===========================================================================
echo.
echo   Bu sihirbaz asagidaki islemleri otomatik yapar:
echo.
echo     * Python 3.10 kontrolu / kurulumu
echo     * Gerekli Python kutuphaneleri
echo     * AI model dosyalari (GFPGAN / RealESRGAN)
echo.
echo   Not: Python sistemde varsa tekrar kurulmaz.
echo.
echo  ===========================================================================
echo.
pause

:: ============================================================================
:: ADIM 1 - PYTHON 3.10 KONTROLU
:: ============================================================================
cls
echo.
echo  ===========================================================================
echo                     ADIM 1 / 4  -  PYTHON KONTROLU
echo  ===========================================================================

set PY310_EXE=

if exist "C:\Program Files\Python310\python.exe" set PY310_EXE=C:\Program Files\Python310\python.exe
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set PY310_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe

if defined PY310_EXE (
    echo  [OK] Python 3.10 bulundu:
    echo       %PY310_EXE%
    echo.
    pause
    goto STEP2
)

echo  [!] Python 3.10 bulunamadi
echo      Indirme islemi baslatiliyor...
echo.

powershell -Command "Invoke-WebRequest '%PY_INSTALLER_URL%' -OutFile '%PY_INSTALLER%'" 2>nul

if %errorlevel% neq 0 (
    echo.
    echo  [HATA] Python indirilemedi!
    pause
    exit /b 1
)

echo  Python kuruluyor...
%PY_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

if exist "C:\Program Files\Python310\python.exe" set PY310_EXE=C:\Program Files\Python310\python.exe
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set PY310_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe

if not defined PY310_EXE (
    echo.
    echo  [HATA] Python kuruldu fakat bulunamadi!
    pause
    exit /b 1
)

echo.
echo  [OK] Python 3.10 kurulumu tamamlandi.
pause


:: ============================================================================
:: ADIM 2 - BAGIMLILIK YUKLEME
:: ============================================================================
:STEP2
cls
echo.
echo  ===========================================================================
echo                  ADIM 2 / 4  -  BAGIMLILIKLAR
echo  ===========================================================================

echo  pip guncelleniyor...
"%PY310_EXE%" -m pip install --upgrade pip

echo.
echo  Torch (CPU) yukleniyor...
"%PY310_EXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-warn-script-location
if %errorlevel% neq 0 (
    echo.
    echo  [HATA] Torch yuklenemedi!
    pause
    exit /b 1
)

echo.
echo  Diger gereksinimler yukleniyor...
"%PY310_EXE%" -m pip install -r "%REQUIREMENTS_FILE%" --no-warn-script-location
if %errorlevel% neq 0 (
    echo.
    echo  [HATA] gereksinimler yuklenirken hata olustu!
    pause
    exit /b 1
)

echo.
echo  [OK] Tum bagimliliklar yuklendi.
pause
goto STEP3


:: ============================================================================
:: ADIM 3 - MODEL DOSYALARI
:: ============================================================================
:STEP3
cls
echo.
echo  ===========================================================================
echo                  ADIM 3 / 4  -  MODEL KONTROLU
echo  ===========================================================================

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

set FIRST_INSTALL=0
if not exist "%MODEL_DIR%\versiyon.txt" set FIRST_INSTALL=1

echo.
echo  Versiyon bilgisi aliniyor...
"%PY310_EXE%" -m gdown --fuzzy "https://drive.google.com/file/d/1q8XQB3hhTo-fIObHAOXniQw1hhTCR5nf/view" -O "%MODEL_DIR%\versiyon_tmp.txt"

for %%M in (GFPGAN REALESRGAN REALRESNET) do (

    set REMOTE_VER=
    set LOCAL_VER=

    for /f "tokens=2 delims==" %%R in ('findstr %%M "%MODEL_DIR%\versiyon_tmp.txt"') do set REMOTE_VER=%%R
    if "!FIRST_INSTALL!"=="0" (
        for /f "tokens=2 delims==" %%L in ('findstr %%M "%MODEL_DIR%\versiyon.txt"') do set LOCAL_VER=%%L
    )

    if "!FIRST_INSTALL!"=="1" (
        echo.
        echo  [ILK KURULUM] %%M indiriliyor...
        set DO_DOWNLOAD=1
    ) else if not "!REMOTE_VER!"=="!LOCAL_VER!" (
        echo.
        echo  [GUNCELLEME] %%M yeni surum bulundu, indiriliyor...
        set DO_DOWNLOAD=1
    ) else (
        echo.
        echo  [OK] %%M zaten guncel.
        set DO_DOWNLOAD=0
    )

    if "!DO_DOWNLOAD!"=="1" (
        if "%%M"=="GFPGAN" "%PY310_EXE%" -m gdown 1plO391KI_tFMAudOlHhQooR_LGjPF_jW -O "%MODEL_DIR%\GFPGANv1.3.pth"
        if "%%M"=="REALESRGAN" "%PY310_EXE%" -m gdown 1o0gpcvfVbnD2gdM5qkVcd04YO-Nlkm6e -O "%MODEL_DIR%\RealESRGAN_x4plus.pth"
        if "%%M"=="REALRESNET" "%PY310_EXE%" -m gdown 15njh9lkvdBgBuHx24LrsTDT7YyFKUtGm -O "%MODEL_DIR%\RealESRNet_x4plus.pth"
    )
)

copy "%MODEL_DIR%\versiyon_tmp.txt" "%MODEL_DIR%\versiyon.txt" >nul
del "%MODEL_DIR%\versiyon_tmp.txt" >nul 2>&1

echo.
echo  [OK] Model islemleri tamamlandi.
pause
goto STEP4


:: ============================================================================
:: ADIM 4 - BASLAT.CMD
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
echo                    KURULUM BASARIYLA TAMAMLANDI
echo  ===========================================================================
echo.
echo   Programi baslatmak icin:
echo   -> Baslat.cmd
echo.
pause

ENDLOCAL
exit /b 0