#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
AUDIO_DIR="$SCRIPT_DIR/test_audio"

mkdir -p "$MODELS_DIR" "$AUDIO_DIR"

# ── SenseVoice-Small ──
SENSEVOICE_DIR="$MODELS_DIR/sensevoice"
if [ -f "$SENSEVOICE_DIR/model.onnx" ]; then
	echo "[skip] SenseVoice model already exists"
else
	echo "[download] SenseVoice-Small ONNX model..."
	cd "$MODELS_DIR"
	wget -q --show-progress \
		https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
	tar xf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
	mv sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 sensevoice
	rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
	echo "[done] SenseVoice model -> $SENSEVOICE_DIR"
fi

# ── Silero VAD ──
if [ -f "$MODELS_DIR/silero_vad.onnx" ]; then
	echo "[skip] Silero VAD model already exists"
else
	echo "[download] Silero VAD ONNX model..."
	wget -q --show-progress -O "$MODELS_DIR/silero_vad.onnx" \
		https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
	echo "[done] Silero VAD model -> $MODELS_DIR/silero_vad.onnx"
fi

# ── Test audio: LibriSpeech (English, ~10s) ──
if [ -f "$AUDIO_DIR/en_long.wav" ]; then
	echo "[skip] English test audio already exists"
else
	echo "[download] LibriSpeech test-clean sample..."
	cd /tmp
	wget -q --show-progress \
		https://www.openslr.org/resources/12/test-clean.tar.gz
	tar xf test-clean.tar.gz --strip-components=4 \
		"LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"
	# Convert to 16kHz mono WAV
	if command -v ffmpeg &>/dev/null; then
		ffmpeg -y -i 1089-134686-0000.flac -ar 16000 -ac 1 "$AUDIO_DIR/en_long.wav" 2>/dev/null
	elif command -v sox &>/dev/null; then
		sox 1089-134686-0000.flac -r 16000 -c 1 "$AUDIO_DIR/en_long.wav"
	else
		echo "[warn] Neither ffmpeg nor sox found, trying python..."
		python3 -c "
import soundfile as sf
import numpy as np
data, sr = sf.read('1089-134686-0000.flac')
# Simple resample by decimation if needed
if sr != 16000:
    ratio = sr / 16000
    indices = np.round(np.arange(0, len(data), ratio)).astype(int)
    indices = indices[indices < len(data)]
    data = data[indices]
sf.write('$AUDIO_DIR/en_long.wav', data, 16000, subtype='PCM_16')
"
	fi
	rm -f 1089-134686-0000.flac test-clean.tar.gz
	rm -rf LibriSpeech
	echo "[done] English audio -> $AUDIO_DIR/en_long.wav"
fi

# ── Test audio: AISHELL-1 (Chinese) ──
# AISHELL-1 full dataset is too large (~15GB). Use the model's bundled test wav instead,
# and concatenate to make a longer sample.
if [ -f "$AUDIO_DIR/zh_long.wav" ]; then
	echo "[skip] Chinese test audio already exists"
else
	echo "[create] Chinese test audio from SenseVoice bundled test_wavs..."
	if [ -f "$SENSEVOICE_DIR/test_wavs/zh.wav" ]; then
		# Repeat the bundled Chinese wav multiple times to create a longer sample
		if command -v ffmpeg &>/dev/null; then
			# Create a concat file
			for i in $(seq 1 5); do
				echo "file '$SENSEVOICE_DIR/test_wavs/zh.wav'"
			done >/tmp/concat_zh.txt
			ffmpeg -y -f concat -safe 0 -i /tmp/concat_zh.txt \
				-ar 16000 -ac 1 "$AUDIO_DIR/zh_long.wav" 2>/dev/null
			rm /tmp/concat_zh.txt
		else
			# Fallback: just copy the original
			cp "$SENSEVOICE_DIR/test_wavs/zh.wav" "$AUDIO_DIR/zh_long.wav"
		fi
		echo "[done] Chinese audio -> $AUDIO_DIR/zh_long.wav"
	else
		echo "[warn] No Chinese test wav found. Run this after SenseVoice download."
	fi
fi

# ── Summary ──
echo ""
echo "=== Models ==="
ls -lh "$MODELS_DIR/sensevoice/model.onnx" 2>/dev/null || echo "  SenseVoice: MISSING"
ls -lh "$MODELS_DIR/silero_vad.onnx" 2>/dev/null || echo "  Silero VAD: MISSING"
echo ""
echo "=== Test Audio ==="
ls -lh "$AUDIO_DIR"/*.wav 2>/dev/null || echo "  No test audio found"
