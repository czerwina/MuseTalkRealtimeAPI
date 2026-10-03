#!/bin/bash
set -e

# Set the checkpoints directory
CheckpointsDir="models"

# Create necessary directories
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/face-parse-bisent models/sd-vae models/whisper

# Set HuggingFace mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com

# Download MuseTalk V1.0 weights
hf download TMElyralab/MuseTalk \
  --local-dir "$CheckpointsDir" \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# Download MuseTalk V1.5 weights (unet.pth)
hf download TMElyralab/MuseTalk \
  --local-dir "$CheckpointsDir" \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# Download SD VAE weights
hf download stabilityai/sd-vae-ft-mse \
  --local-dir "$CheckpointsDir/sd-vae" \
  --include "config.json" "diffusion_pytorch_model.bin"

# Download Whisper weights
hf download openai/whisper-tiny \
  --local-dir "$CheckpointsDir/whisper" \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Download MediaPipe Face Landmarker task model
curl -L --fail --retry 3 \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  -o "$CheckpointsDir/face_landmarker.task"

# Download SyncNet weights
hf download ByteDance/LatentSync \
  --local-dir "$CheckpointsDir/syncnet" \
  --include "latentsync_syncnet.pt"

# Download Face Parse Bisent weights
hf download ManyOtherFunctions/face-parse-bisent \
  --local-dir "$CheckpointsDir/face-parse-bisent" \
  --include "79999_iter.pth" "resnet18-5c106cde.pth"
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o "$CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth"

echo "✅ All weights have been downloaded successfully!" 
