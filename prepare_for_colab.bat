@echo off
echo ========================================
echo PREPARING PROJECT FOR GOOGLE COLAB
echo ========================================
echo.
echo NOTE: Excluding heavy image folders for faster upload
echo Images will be downloaded separately in Colab
echo.

echo [1/3] Creating temporary folder structure...
if exist temp_colab mkdir temp_colab /s /q
mkdir temp_colab\data\processed
mkdir temp_colab\src

echo [2/3] Copying lightweight files...
xcopy /E /I /Y src temp_colab\src
xcopy /Y data\processed\*.pkl temp_colab\data\processed\
xcopy /Y data\artemis_dataset_release_v0.csv temp_colab\data\
xcopy /Y requirements.txt temp_colab\
xcopy /Y colab_training.ipynb temp_colab\
xcopy /Y COLAB_GUIDE.txt temp_colab\

echo.
echo [3/3] Creating lightweight zip (excluding images)...
echo This should be much faster now...
powershell -Command "Compress-Archive -Path temp_colab\* -DestinationPath iml-a3.zip -Force"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [4/4] Cleaning up...
    rmdir /s /q temp_colab
    
    echo.
    echo ========================================
    echo SUCCESS! Lightweight zip created
    echo ========================================
    echo.
    echo File size:
    dir iml-a3.zip
    echo.
    echo ========================================
    echo NEXT STEPS:
    echo ========================================
    echo 1. Upload iml-a3.zip to Google Colab
    echo 2. Open colab_training.ipynb in Colab
    echo 3. Runtime -^> Change runtime type -^> T4 GPU
    echo 4. Run all cells
    echo.
    echo IMPORTANT: The zip excludes images to save space.
    echo You'll need to upload images separately OR mount Drive.
    echo See colab_training.ipynb for image setup options.
    echo ========================================
) else (
    echo.
    echo ERROR: Failed to create zip file
    rmdir /s /q temp_colab
)

echo.
pause

