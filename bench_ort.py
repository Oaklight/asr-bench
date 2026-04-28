"""Benchmark onnxruntime: SenseVoice ASR + Silero VAD with multiple execution providers.

Run in the asr-bench-ort conda environment:
    conda activate asr-bench-ort
    python bench_ort.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

import onnxruntime as ort

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
AUDIO_DIR = SCRIPT_DIR / "test_audio"
RESULTS_DIR = SCRIPT_DIR / "results"

SENSEVOICE_MODEL = MODELS_DIR / "sensevoice" / "model.int8.onnx"
SILERO_VAD_MODEL = MODELS_DIR / "silero_vad.onnx"

WARMUP_RUNS = 10
BENCH_RUNS = 50
SAMPLE_RATE = 16000
NUM_THREADS = 4

# Silero VAD constants
SILERO_WINDOW_SIZE = 512  # for 16kHz


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


def get_audio_files() -> list[tuple[str, str]]:
    """Return list of (label, path) for test audio."""
    files = []
    test_wavs = MODELS_DIR / "sensevoice" / "test_wavs"
    if test_wavs.exists():
        for wav in sorted(test_wavs.glob("*.wav")):
            files.append((f"bundled_{wav.stem}", str(wav)))
    for wav in sorted(AUDIO_DIR.glob("*.wav")):
        files.append((wav.stem, str(wav)))
    return files


def get_providers() -> list[tuple[str, list]]:
    """Return available execution providers to benchmark."""
    providers = [("CPU", [("CPUExecutionProvider", {})])]

    available = ort.get_available_providers()
    if "OpenVINOExecutionProvider" in available:
        providers.append(
            (
                "OpenVINO-CPU",
                [
                    ("OpenVINOExecutionProvider", {"device_type": "CPU"}),
                    ("CPUExecutionProvider", {}),
                ],
            )
        )
        providers.append(
            (
                "OpenVINO-GPU",
                [
                    ("OpenVINOExecutionProvider", {"device_type": "GPU"}),
                    ("CPUExecutionProvider", {}),
                ],
            )
        )
    else:
        print("[warn] OpenVINOExecutionProvider not available")

    return providers


def create_session(
    model_path: str, providers: list, num_threads: int = NUM_THREADS
) -> ort.InferenceSession:
    """Create ONNX Runtime inference session."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, opts, providers=providers)


# ── Silero VAD via raw onnxruntime ──


class SileroVADOrt:
    """Silero VAD wrapper using raw onnxruntime."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_names = [inp.name for inp in session.get_inputs()]
        self.output_names = [out.name for out in session.get_outputs()]
        self._inspect_model()
        self.reset()

    def _inspect_model(self):
        """Inspect model inputs to determine version and setup."""
        self.input_map = {}
        for inp in self.session.get_inputs():
            self.input_map[inp.name] = {
                "shape": inp.shape,
                "type": inp.type,
            }
        print(f"    VAD model inputs: {list(self.input_map.keys())}")

    def reset(self):
        """Reset internal state for a new audio stream."""
        # Silero VAD v4/v5 uses h and c states (LSTM hidden/cell)
        # Shape: (2, 1, 64) for 16kHz
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.sr_tensor = np.array(SAMPLE_RATE, dtype=np.int64)

    def __call__(self, audio_chunk: np.ndarray) -> float:
        """Run VAD on a single audio chunk, return speech probability."""
        chunk = audio_chunk.astype(np.float32)
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]  # (1, window_size)

        # Build feed dict based on actual input names
        feed = {}
        for name in self.input_names:
            if name == "x" or name == "input":
                feed[name] = chunk
            elif name == "h":
                feed[name] = self.h
            elif name == "c":
                feed[name] = self.c
            elif "sr" in name:
                feed[name] = self.sr_tensor
            elif "state" in name:
                feed[name] = self.state

        outputs = self.session.run(self.output_names, feed)

        # First output is probability
        prob = float(outputs[0].squeeze())

        # Update states from remaining outputs
        output_names = self.output_names
        for i, oname in enumerate(output_names):
            if oname == "hn":
                self.h = outputs[i]
            elif oname == "cn":
                self.c = outputs[i]

        return prob

    def process_audio(
        self, audio: np.ndarray, threshold: float = 0.5
    ) -> list[tuple[int, int]]:
        """Process full audio and return speech segments as (start, end) sample indices."""
        self.reset()
        segments = []
        in_speech = False
        speech_start = 0

        for offset in range(0, len(audio) - SILERO_WINDOW_SIZE + 1, SILERO_WINDOW_SIZE):
            chunk = audio[offset : offset + SILERO_WINDOW_SIZE]
            prob = self(chunk)

            if prob >= threshold and not in_speech:
                in_speech = True
                speech_start = offset
            elif prob < threshold and in_speech:
                in_speech = False
                segments.append((speech_start, offset + SILERO_WINDOW_SIZE))

        if in_speech:
            segments.append((speech_start, len(audio)))

        return segments


# ── SenseVoice via raw onnxruntime ──
# Note: SenseVoice requires fbank feature extraction as preprocessing.
# We benchmark the raw model inference only, measuring the ONNX session.run() time.


def get_sensevoice_dummy_input(session: ort.InferenceSession) -> dict:
    """Create dummy input matching model's expected shape.

    SenseVoice inputs: x(N,T,560), x_length(N), language(N), text_norm(N)
    All 'N' dims must match (batch size), T is fbank frames (~100 per second).
    """
    batch_size = 1
    num_frames = 700  # ~7s of audio at 100 frames/s

    feed = {}
    for inp in session.get_inputs():
        shape = inp.shape
        concrete_shape = []
        for dim in shape:
            if isinstance(dim, str) or dim is None:
                if any(s in inp.name for s in ("x",)) and len(shape) == 3:
                    # For x: (N, T, 560) - use num_frames for T
                    if len(concrete_shape) == 0:
                        concrete_shape.append(batch_size)  # N
                    else:
                        concrete_shape.append(num_frames)  # T
                else:
                    concrete_shape.append(batch_size)  # N for scalar inputs
            else:
                concrete_shape.append(dim)

        if "float" in inp.type.lower():
            feed[inp.name] = np.random.randn(*concrete_shape).astype(np.float32)
        elif "int32" in inp.type.lower():
            feed[inp.name] = np.ones(concrete_shape, dtype=np.int32)
        elif "int64" in inp.type.lower():
            feed[inp.name] = np.ones(concrete_shape, dtype=np.int64)
        elif "int" in inp.type.lower():
            feed[inp.name] = np.ones(concrete_shape, dtype=np.int32)
        else:
            feed[inp.name] = np.zeros(concrete_shape, dtype=np.float32)

    # Set x_length to actual frame count
    if "x_length" in feed:
        feed["x_length"][:] = num_frames

    return feed


def bench_sensevoice(providers_list: list[tuple[str, list]]) -> dict:
    """Benchmark SenseVoice model across execution providers."""
    results = {}

    for provider_name, providers in providers_list:
        print(f"\n  Provider: {provider_name}")
        try:
            mem_before = get_memory_mb()
            t0 = time.perf_counter()
            session = create_session(str(SENSEVOICE_MODEL), providers)
            load_time_ms = (time.perf_counter() - t0) * 1000
            mem_after = get_memory_mb()

            actual_providers = session.get_providers()
            print(f"    Active: {actual_providers}")
            print(
                f"    Load: {load_time_ms:.0f}ms, Mem: +{mem_after - mem_before:.0f}MB"
            )

            # Get model input info
            print(
                f"    Inputs: {[(i.name, i.shape, i.type) for i in session.get_inputs()]}"
            )

            # Create dummy input
            feed = get_sensevoice_dummy_input(session)
            print(f"    Feed shapes: {[(k, v.shape) for k, v in feed.items()]}")

            # Warmup
            for _ in range(WARMUP_RUNS):
                session.run(None, feed)

            # Benchmark
            latencies = []
            for _ in range(BENCH_RUNS):
                t0 = time.perf_counter()
                session.run(None, feed)
                elapsed = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed)

            lat = np.array(latencies)
            mem_peak = get_memory_mb()

            results[provider_name] = {
                "success": True,
                "active_providers": actual_providers,
                "model_load_ms": round(load_time_ms, 2),
                "model_load_mem_mb": round(mem_after - mem_before, 1),
                "latency_mean_ms": round(float(np.mean(lat)), 2),
                "latency_median_ms": round(float(np.median(lat)), 2),
                "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
                "latency_p99_ms": round(float(np.percentile(lat, 99)), 2),
                "latency_min_ms": round(float(np.min(lat)), 2),
                "latency_max_ms": round(float(np.max(lat)), 2),
                "mem_peak_mb": round(mem_peak, 1),
                "input_shapes": {k: list(v.shape) for k, v in feed.items()},
            }
            print(
                f"    Inference: mean={np.mean(lat):.1f}ms, p95={np.percentile(lat, 95):.1f}ms"
            )

            del session

        except Exception as e:
            print(f"    FAILED: {e}")
            results[provider_name] = {"success": False, "error": str(e)}

    return results


def bench_vad(
    providers_list: list[tuple[str, list]], audio_files: list[tuple[str, str]]
) -> dict:
    """Benchmark Silero VAD across execution providers."""
    results = {}

    for provider_name, providers in providers_list:
        print(f"\n  Provider: {provider_name}")
        try:
            mem_before = get_memory_mb()
            t0 = time.perf_counter()
            session = create_session(str(SILERO_VAD_MODEL), providers, num_threads=1)
            load_time_ms = (time.perf_counter() - t0) * 1000
            mem_after = get_memory_mb()

            actual_providers = session.get_providers()
            print(f"    Active: {actual_providers}")

            vad = SileroVADOrt(session)

            file_results = {}
            for label, path in audio_files:
                audio = load_audio(path)
                duration_s = len(audio) / SAMPLE_RATE

                # Warmup
                for _ in range(WARMUP_RUNS):
                    vad.process_audio(audio)

                # Benchmark
                latencies = []
                for _ in range(BENCH_RUNS):
                    t0 = time.perf_counter()
                    segments = vad.process_audio(audio)
                    elapsed = (time.perf_counter() - t0) * 1000
                    latencies.append(elapsed)

                lat = np.array(latencies)
                mem_peak = get_memory_mb()

                file_results[label] = {
                    "duration_s": round(duration_s, 2),
                    "num_segments": len(segments),
                    "latency_mean_ms": round(float(np.mean(lat)), 2),
                    "latency_median_ms": round(float(np.median(lat)), 2),
                    "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
                    "latency_min_ms": round(float(np.min(lat)), 2),
                    "mem_peak_mb": round(mem_peak, 1),
                }
                print(
                    f"    VAD [{label}]: {np.mean(lat):.1f}ms, {len(segments)} segments"
                )

            results[provider_name] = {
                "success": True,
                "active_providers": actual_providers,
                "model_load_ms": round(load_time_ms, 2),
                "model_load_mem_mb": round(mem_after - mem_before, 1),
                "files": file_results,
            }

            del session, vad

        except Exception as e:
            print(f"    FAILED: {e}")
            results[provider_name] = {"success": False, "error": str(e)}

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    audio_files = get_audio_files()

    print(f"onnxruntime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Test audio: {[f[0] for f in audio_files]}")
    print(f"Threads: {NUM_THREADS}")

    providers_list = get_providers()
    print(f"Will benchmark: {[p[0] for p in providers_list]}")

    all_results = {
        "engine": "onnxruntime",
        "ort_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "num_threads": NUM_THREADS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print("\n=== SenseVoice ASR Benchmark (raw model inference) ===")
    all_results["asr"] = bench_sensevoice(providers_list)

    if audio_files:
        print("\n=== Silero VAD Benchmark ===")
        all_results["vad"] = bench_vad(providers_list, audio_files)
    else:
        print("\n[skip] No audio files for VAD benchmark")

    out_path = RESULTS_DIR / "ort_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
