@echo off
setlocal
cd /d "%~dp0"

set "CheckpointsDir=models"
mkdir "%CheckpointsDir%\musetalk" 2>nul
mkdir "%CheckpointsDir%\musetalkV15" 2>nul
mkdir "%CheckpointsDir%\syncnet" 2>nul
mkdir "%CheckpointsDir%\face-parse-bisent" 2>nul
mkdir "%CheckpointsDir%\sd-vae" 2>nul
mkdir "%CheckpointsDir%\whisper" 2>nul

set "HF_ENDPOINT=https://hf-mirror.com"

hf download TMElyralab/MuseTalk --local-dir "%CheckpointsDir%" --include "musetalk/*" "musetalkV15/*"
if errorlevel 1 exit /b 1

hf download stabilityai/sd-vae-ft-mse --local-dir "%CheckpointsDir%\sd-vae" --include "config.json" "diffusion_pytorch_model.bin"
if errorlevel 1 exit /b 1

hf download openai/whisper-tiny --local-dir "%CheckpointsDir%\whisper" --include "config.json" "pytorch_model.bin" "preprocessor_config.json"
if errorlevel 1 exit /b 1

curl.exe -L --fail --retry 3 "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -o "%CheckpointsDir%\face_landmarker.task"
if errorlevel 1 exit /b 1

hf download ByteDance/LatentSync --local-dir "%CheckpointsDir%\syncnet" --include "latentsync_syncnet.pt"
if errorlevel 1 exit /b 1

hf download ManyOtherFunctions/face-parse-bisent --local-dir "%CheckpointsDir%\face-parse-bisent" --include "79999_iter.pth"
if errorlevel 1 exit /b 1

curl.exe -L --fail --retry 3 "https://download.pytorch.org/models/resnet18-5c106cde.pth" -o "%CheckpointsDir%\face-parse-bisent\resnet18-5c106cde.pth"
if errorlevel 1 exit /b 1

echo All required MuseTalk weights have been downloaded successfully.
endlocal
