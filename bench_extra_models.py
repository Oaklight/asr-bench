"""Benchmark additional ASR models via sherpa-onnx:
- Qwen3-ASR-0.6B INT8
- Qwen3-ASR-1.7B INT8
- FireRedASR2-CTC INT8

Run in the asr-bench-sherpa conda environment:
    conda activate asr-bench-sherpa
    python bench_extra_models.py
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
RESULTS_DIR = SCRIPT_DIR / "results"

SILERO_VAD_MODEL = MODELS_DIR / "silero_vad.onnx"

WARMUP_RUNS = 2
BENCH_RUNS = 10
SAMPLE_RATE = 16000
NUM_THREADS = 4


def load_audio(path: str) -> np.ndarray:
    """Load audio file, convert to mono 16kHz float32."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio[:, 0]
    if sr != SAMPLE_RATE:
        ratio = sr / SAMPLE_RATE
        indices = np.round(np.arange(0, len(audio), ratio)).astype(int)
        indices = indices[indices < len(audio)]
        audio = audio[indices]
    return audio


def get_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_test_audio() -> list[tuple[str, str]]:
    """Collect test audio files — only SenseVoice bundled wavs + longer test audio."""
    files = []
    # SenseVoice bundled test wavs (zh/en/ja/ko/yue, ~5-7s each)
    sv_wavs = MODELS_DIR / "sensevoice" / "test_wavs"
    if sv_wavs.exists():
        for wav in sorted(sv_wavs.glob("*.wav")):
            files.append((f"sv_{wav.stem}", str(wav)))

    # Longer test audio
    audio_dir = SCRIPT_DIR / "test_audio"
    if audio_dir.exists():
        for wav in sorted(audio_dir.glob("*.wav")):
            files.append((wav.stem, str(wav)))

    return files


def bench_model(
    name: str,
    recognizer: sherpa_onnx.OfflineRecognizer,
    audio_files: list[tuple[str, str]],
    load_time_ms: float,
    mem_after_load: float,
    mem_before: float,
) -> dict:
    """Run benchmark for a single model across all audio files."""
    result = {
        "model_load_ms": round(load_time_ms, 2),
        "model_load_mem_mb": round(mem_after_load - mem_before, 1),
        "files": {},
    }

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
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            s = recognizer.create_stream()
            s.accept_waveform(SAMPLE_RATE, audio)
            recognizer.decode_stream(s)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

        text = s.result.text.strip()
        mem_peak = get_memory_mb()
        lat = np.array(latencies)

        rtf = float(np.mean(lat)) / 1000 / duration_s
        result["files"][label] = {
            "duration_s": round(duration_s, 2),
            "text": text,
            "latency_mean_ms": round(float(np.mean(lat)), 2),
            "latency_median_ms": round(float(np.median(lat)), 2),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
            "latency_min_ms": round(float(np.min(lat)), 2),
            "rtf": round(rtf, 4),
            "mem_peak_mb": round(mem_peak, 1),
        }
        print(f"  [{label}] {np.mean(lat):.0f}ms (RTF={rtf:.4f}) | {text[:60]}")

    return result


def load_sensevoice() -> tuple[
    sherpa_onnx.OfflineRecognizer | None, float, float, float
]:
    """Load SenseVoice-Small as baseline."""
    model = MODELS_DIR / "sensevoice" / "model.int8.onnx"
    tokens = MODELS_DIR / "sensevoice" / "tokens.txt"
    if not model.exists():
        return None, 0, 0, 0

    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model),
        tokens=str(tokens),
        use_itn=True,
        num_threads=NUM_THREADS,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    mem_after = get_memory_mb()
    return recognizer, load_ms, mem_after, mem_before


def load_qwen3_asr(
    version: str,
) -> tuple[sherpa_onnx.OfflineRecognizer | None, float, float, float]:
    """Load Qwen3-ASR model."""
    model_dir = MODELS_DIR / f"sherpa-onnx-qwen3-asr-{version}-int8-2026-03-25"
    if not model_dir.exists():
        return None, 0, 0, 0

    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=str(model_dir / "conv_frontend.onnx"),
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        tokenizer=str(model_dir / "tokenizer"),
        num_threads=NUM_THREADS,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    mem_after = get_memory_mb()
    return recognizer, load_ms, mem_after, mem_before


def load_firered_ctc() -> tuple[
    sherpa_onnx.OfflineRecognizer | None, float, float, float
]:
    """Load FireRedASR2-CTC model."""
    model_dir = MODELS_DIR / "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25"
    if not model_dir.exists():
        return None, 0, 0, 0

    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(
        model=str(model_dir / "model.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=NUM_THREADS,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    mem_after = get_memory_mb()
    return recognizer, load_ms, mem_after, mem_before


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    audio_files = get_test_audio()

    if not audio_files:
        print("No test audio found. Run download_models.sh first.")
        return

    print(f"Test audio: {[f[0] for f in audio_files]}")
    print(f"Threads: {NUM_THREADS}, Warmup: {WARMUP_RUNS}, Runs: {BENCH_RUNS}")
    print()

    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_threads": NUM_THREADS,
        "warmup_runs": WARMUP_RUNS,
        "bench_runs": BENCH_RUNS,
        "models": {},
    }

    # Model loaders: (name, loader_func, args)
    models = [
        ("SenseVoice-Small-INT8", load_sensevoice, ()),
        ("Qwen3-ASR-0.6B-INT8", load_qwen3_asr, ("0.6B",)),
        ("Qwen3-ASR-1.7B-INT8", load_qwen3_asr, ("1.7B",)),
        ("FireRedASR2-CTC-INT8", load_firered_ctc, ()),
    ]

    for model_name, loader, args in models:
        print(f"=== {model_name} ===")
        recognizer, load_ms, mem_after, mem_before = loader(*args)
        if recognizer is None:
            print("  [skip] Model not found\n")
            all_results["models"][model_name] = {"skipped": True}
            continue

        print(f"  Loaded in {load_ms:.0f}ms, mem +{mem_after - mem_before:.0f}MB")
        result = bench_model(
            model_name, recognizer, audio_files, load_ms, mem_after, mem_before
        )
        all_results["models"][model_name] = result

        # Free memory before loading next model
        del recognizer
        print()

    out_path = RESULTS_DIR / "extra_models_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':<28} {'Load(ms)':>9} {'Mem(MB)':>8} "
        f"{'Mean(ms)':>9} {'P95(ms)':>8} {'RTF':>7} | Sample text"
    )
    print("-" * 80)

    for model_name, data in all_results["models"].items():
        if data.get("skipped"):
            print(f"{model_name:<28} {'SKIPPED':>9}")
            continue
        # Use zh audio if available, else first file
        file_key = None
        for key in data["files"]:
            if "zh" in key:
                file_key = key
                break
        if file_key is None:
            file_key = next(iter(data["files"]))

        fd = data["files"][file_key]
        text_preview = fd["text"][:30]
        print(
            f"{model_name:<28} {data['model_load_ms']:>8.0f} "
            f"{data['model_load_mem_mb']:>8.0f} "
            f"{fd['latency_mean_ms']:>9.1f} {fd['latency_p95_ms']:>8.1f} "
            f"{fd['rtf']:>7.4f} | {text_preview}"
        )


if __name__ == "__main__":
    main()
