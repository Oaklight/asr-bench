"""Benchmark sherpa-onnx: SenseVoice ASR + Silero VAD + combined pipeline.

Run in the asr-bench-sherpa conda environment:
    conda activate asr-bench-sherpa
    python bench_sherpa.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

import sherpa_onnx

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
AUDIO_DIR = SCRIPT_DIR / "test_audio"
RESULTS_DIR = SCRIPT_DIR / "results"

SENSEVOICE_MODEL = MODELS_DIR / "sensevoice" / "model.int8.onnx"
SENSEVOICE_TOKENS = MODELS_DIR / "sensevoice" / "tokens.txt"
SILERO_VAD_MODEL = MODELS_DIR / "silero_vad.onnx"

WARMUP_RUNS = 10
BENCH_RUNS = 50
SAMPLE_RATE = 16000
NUM_THREADS = 4


def load_audio(path: str) -> np.ndarray:
    """Load audio file, convert to mono 16kHz float32."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # mono
    if sr != SAMPLE_RATE:
        # Simple resampling
        ratio = sr / SAMPLE_RATE
        indices = np.round(np.arange(0, len(audio), ratio)).astype(int)
        indices = indices[indices < len(audio)]
        audio = audio[indices]
    return audio


def get_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_audio_files() -> list[tuple[str, str]]:
    """Return list of (label, path) for test audio."""
    files = []
    # Model bundled test wavs
    test_wavs = MODELS_DIR / "sensevoice" / "test_wavs"
    if test_wavs.exists():
        for wav in sorted(test_wavs.glob("*.wav")):
            files.append((f"bundled_{wav.stem}", str(wav)))
    # Longer test audio
    for wav in sorted(AUDIO_DIR.glob("*.wav")):
        files.append((wav.stem, str(wav)))
    return files


def bench_asr(audio_files: list[tuple[str, str]]) -> dict:
    """Benchmark SenseVoice ASR via sherpa-onnx."""
    results = {}

    # Load model
    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(SENSEVOICE_MODEL),
        tokens=str(SENSEVOICE_TOKENS),
        use_itn=True,
        debug=False,
        num_threads=NUM_THREADS,
    )
    load_time_ms = (time.perf_counter() - t0) * 1000
    mem_after_load = get_memory_mb()

    results["model_load_ms"] = load_time_ms
    results["model_load_mem_mb"] = mem_after_load - mem_before
    results["files"] = {}

    for label, path in audio_files:
        audio = load_audio(path)
        duration_s = len(audio) / SAMPLE_RATE

        # Warmup
        for _ in range(WARMUP_RUNS):
            s = recognizer.create_stream()
            s.accept_waveform(SAMPLE_RATE, audio)
            recognizer.decode_stream(s)

        # Benchmark
        latencies = []
        cpu_samples = []
        for _ in range(BENCH_RUNS):
            psutil.cpu_percent(interval=None)  # reset counter
            t0 = time.perf_counter()
            s = recognizer.create_stream()
            s.accept_waveform(SAMPLE_RATE, audio)
            recognizer.decode_stream(s)
            elapsed = (time.perf_counter() - t0) * 1000
            cpu_after = psutil.cpu_percent(interval=None)
            latencies.append(elapsed)
            cpu_samples.append(cpu_after)

        text = s.result.text.strip()
        mem_peak = get_memory_mb()

        lat = np.array(latencies)
        results["files"][label] = {
            "duration_s": round(duration_s, 2),
            "text": text,
            "latency_mean_ms": round(float(np.mean(lat)), 2),
            "latency_median_ms": round(float(np.median(lat)), 2),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(lat, 99)), 2),
            "latency_min_ms": round(float(np.min(lat)), 2),
            "latency_max_ms": round(float(np.max(lat)), 2),
            "rtf": round(float(np.mean(lat)) / 1000 / duration_s, 4),
            "mem_peak_mb": round(mem_peak, 1),
            "cpu_mean_pct": round(float(np.mean(cpu_samples)), 1),
        }
        print(
            f"  ASR [{label}]: {np.mean(lat):.1f}ms (RTF={results['files'][label]['rtf']:.4f})"
        )

    return results


def bench_vad(audio_files: list[tuple[str, str]]) -> dict:
    """Benchmark Silero VAD via sherpa-onnx."""
    results = {}

    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(SILERO_VAD_MODEL)
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.25
    config.silero_vad.min_speech_duration = 0.25
    config.sample_rate = SAMPLE_RATE
    window_size = config.silero_vad.window_size
    load_time_ms = (time.perf_counter() - t0) * 1000
    mem_after_load = get_memory_mb()

    results["model_load_ms"] = load_time_ms
    results["model_load_mem_mb"] = mem_after_load - mem_before
    results["window_size"] = window_size
    results["files"] = {}

    for label, path in audio_files:
        audio = load_audio(path)
        duration_s = len(audio) / SAMPLE_RATE

        def run_vad(audio_data):
            vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)
            offset = 0
            segments = []
            while offset + window_size <= len(audio_data):
                vad.accept_waveform(audio_data[offset : offset + window_size])
                offset += window_size
                while not vad.empty():
                    segments.append(len(vad.front.samples))
                    vad.pop()
            vad.flush()
            while not vad.empty():
                segments.append(len(vad.front.samples))
                vad.pop()
            return segments

        # Warmup
        for _ in range(WARMUP_RUNS):
            run_vad(audio)

        # Benchmark
        latencies = []
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            segments = run_vad(audio)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

        mem_peak = get_memory_mb()
        lat = np.array(latencies)
        results["files"][label] = {
            "duration_s": round(duration_s, 2),
            "num_segments": len(segments),
            "latency_mean_ms": round(float(np.mean(lat)), 2),
            "latency_median_ms": round(float(np.median(lat)), 2),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
            "latency_min_ms": round(float(np.min(lat)), 2),
            "mem_peak_mb": round(mem_peak, 1),
        }
        print(f"  VAD [{label}]: {np.mean(lat):.1f}ms, {len(segments)} segments")

    return results


def bench_combined(audio_files: list[tuple[str, str]]) -> dict:
    """Benchmark VAD -> ASR combined pipeline."""
    results = {}

    # Load both models
    mem_before = get_memory_mb()
    t0 = time.perf_counter()

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(SENSEVOICE_MODEL),
        tokens=str(SENSEVOICE_TOKENS),
        use_itn=True,
        debug=False,
        num_threads=NUM_THREADS,
    )

    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = str(SILERO_VAD_MODEL)
    vad_config.silero_vad.threshold = 0.5
    vad_config.silero_vad.min_silence_duration = 0.25
    vad_config.silero_vad.min_speech_duration = 0.25
    vad_config.sample_rate = SAMPLE_RATE
    window_size = vad_config.silero_vad.window_size

    load_time_ms = (time.perf_counter() - t0) * 1000
    mem_after_load = get_memory_mb()

    results["total_load_ms"] = load_time_ms
    results["total_load_mem_mb"] = mem_after_load - mem_before
    results["files"] = {}

    for label, path in audio_files:
        audio = load_audio(path)
        duration_s = len(audio) / SAMPLE_RATE

        def run_pipeline(audio_data):
            vad = sherpa_onnx.VoiceActivityDetector(
                vad_config, buffer_size_in_seconds=60
            )
            texts = []
            offset = 0
            while offset + window_size <= len(audio_data):
                vad.accept_waveform(audio_data[offset : offset + window_size])
                offset += window_size
                while not vad.empty():
                    stream = recognizer.create_stream()
                    stream.accept_waveform(SAMPLE_RATE, vad.front.samples)
                    recognizer.decode_stream(stream)
                    texts.append(stream.result.text.strip())
                    vad.pop()
            vad.flush()
            while not vad.empty():
                stream = recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, vad.front.samples)
                recognizer.decode_stream(stream)
                texts.append(stream.result.text.strip())
                vad.pop()
            return texts

        # Warmup
        for _ in range(WARMUP_RUNS):
            run_pipeline(audio)

        # Benchmark
        latencies = []
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            texts = run_pipeline(audio)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

        mem_peak = get_memory_mb()
        lat = np.array(latencies)
        results["files"][label] = {
            "duration_s": round(duration_s, 2),
            "texts": texts,
            "latency_mean_ms": round(float(np.mean(lat)), 2),
            "latency_median_ms": round(float(np.median(lat)), 2),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
            "rtf": round(float(np.mean(lat)) / 1000 / duration_s, 4),
            "mem_peak_mb": round(mem_peak, 1),
        }
        print(
            f"  Pipeline [{label}]: {np.mean(lat):.1f}ms "
            f"(RTF={results['files'][label]['rtf']:.4f})"
        )

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    audio_files = get_audio_files()

    if not audio_files:
        print("No test audio found. Run download_models.sh first.")
        return

    print(f"Test audio: {[f[0] for f in audio_files]}")
    print(f"Threads: {NUM_THREADS}")
    print()

    all_results = {
        "engine": "sherpa-onnx",
        "num_threads": NUM_THREADS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print("=== ASR Benchmark ===")
    all_results["asr"] = bench_asr(audio_files)

    print("\n=== VAD Benchmark ===")
    all_results["vad"] = bench_vad(audio_files)

    print("\n=== Combined VAD+ASR Pipeline ===")
    all_results["combined"] = bench_combined(audio_files)

    out_path = RESULTS_DIR / "sherpa_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
