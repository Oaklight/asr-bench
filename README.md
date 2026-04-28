# asr-bench

Benchmark comparison of ASR/VAD inference engines on Intel laptop hardware (no NVIDIA GPU).

## Motivation

When building a local speech recognition pipeline with **SenseVoice-Small** (ASR) and **Silero VAD**, there are multiple inference engine options:

- **sherpa-onnx** — ships its own optimized onnxruntime, with native SenseVoice + Silero VAD integration
- **onnxruntime** — standard ONNX Runtime with CPU execution provider
- **onnxruntime + OpenVINO EP** — leveraging Intel's OpenVINO for potential acceleration on Intel CPUs/iGPUs

This experiment measures real-world performance to determine which engine to use.

## Hardware

| Component | Spec |
|-----------|------|
| CPU | Intel Core Ultra 7 155H (16 threads) |
| RAM | 32 GB LPDDR5X |
| GPU | Intel Arc Xe-LPG integrated (no NVIDIA) |
| NPU | Intel AI Boost (~11 TOPS INT8) |
| Disk | 1.8 TB NVMe SSD |

## Models

| Model | Size | Purpose |
|-------|------|---------|
| [SenseVoice-Small](https://github.com/FunAudioLLM/SenseVoice) (INT8) | 229 MB | ASR — 50+ languages, zh/en/ja/ko/yue |
| [Silero VAD](https://github.com/snakers4/silero-vad) v4 | 629 KB | Voice Activity Detection |
| [Qwen3-ASR-0.6B](https://github.com/QwenLM/Qwen3-ASR) (INT8) | 938 MB | ASR — 30+ languages, autoregressive |
| [FireRedASR2-CTC](https://github.com/modelscope/FunASR) (INT8) | ~500 MB | ASR — zh+en, CTC non-autoregressive |

## Results

### SenseVoice-Small ASR

Test audio: bundled samples (~5-7s each, zh/en/ja/ko/yue).

| Engine | Load (ms) | Infer Mean (ms) | Infer P95 (ms) | Mem Peak (MB) |
|--------|-----------|-----------------|----------------|---------------|
| **sherpa-onnx** | 1,181 | **81-102** | **84-112** | **373** |
| ort CPU | 1,193 | 686 | 708 | 586 |
| ort OpenVINO-CPU | 8,142 | 1,056 | 1,220 | 5,062 |
| ort OpenVINO-GPU* | 1,097 | 733 | 742 | 684 |

\* OpenVINO GPU EP failed to initialize (Meteor Lake iGPU not recognized), fell back to CPU.

**sherpa-onnx is ~7x faster than raw onnxruntime** on the same model. OpenVINO EP is *slower* than plain CPU.

#### sherpa-onnx ASR detail (real audio, with RTF)

| Audio | Duration | Infer (ms) | RTF |
|-------|----------|------------|-----|
| en | 7.2s | 101 | 0.014 |
| ja | 7.2s | 103 | 0.014 |
| ko | 4.6s | 74 | 0.016 |
| yue | 5.1s | 84 | 0.016 |
| zh | 5.6s | 82 | 0.015 |

RTF < 0.02 across all languages — real-time capable with wide margin.

### Silero VAD

| Engine | Infer Mean (ms) | Infer P95 (ms) | Mem (MB) |
|--------|-----------------|----------------|----------|
| sherpa-onnx | 17-27 | 18-27 | **310** |
| **ort CPU** | **10-18** | **11-21** | 685 |
| ort OpenVINO-CPU | 17-28 | 17-36 | 694 |

Raw onnxruntime CPU is marginally faster for VAD alone, but uses 2x memory. OpenVINO again provides no benefit.

### Combined Pipeline (VAD + ASR, sherpa-onnx only)

| Audio | Duration | E2E (ms) | RTF | Mem (MB) |
|-------|----------|----------|-----|----------|
| en | 7.2s | 128 | 0.018 | 376 |
| ja | 7.2s | 115 | 0.016 | 386 |
| ko | 4.6s | 76 | 0.016 | 386 |
| yue | 5.1s | 80 | 0.016 | 386 |
| zh | 5.6s | 96 | 0.017 | 388 |

Full VAD→ASR pipeline runs in **76-128ms** for 5-7s audio. Memory footprint under 400MB.

## Key Findings

1. **sherpa-onnx dominates for ASR**: 7x faster than raw onnxruntime on the same INT8 model. Internal optimizations (quantized inference paths, memory layout, custom kernels) make a huge difference that no execution provider switch can match.

2. **OpenVINO EP hurts rather than helps**:
   - ASR: 54% slower than plain CPU EP (1,056ms vs 686ms)
   - VAD: 60% slower than plain CPU EP (28ms vs 17ms)
   - Adds 8s model load time and 5GB memory overhead
   - Root cause: graph compilation cost exceeds inference benefit for small models

3. **OpenVINO GPU is non-functional**: Meteor Lake integrated Arc GPU lacks necessary drivers/support for OpenVINO EP. Falls back to CPU silently.

4. **sherpa-onnx unifies VAD + ASR cleanly**: Single dependency, shared onnxruntime, native Silero VAD + SenseVoice integration, ~388MB total memory.

## Conclusion

**Use sherpa-onnx.** Do not bother with OpenVINO for these models on this hardware.

For the [asr2clip](https://github.com/oaklight/asr2clip) project, the migration path is:
- Replace numpy-based VAD with sherpa-onnx's built-in Silero VAD
- Replace remote API ASR with sherpa-onnx's local SenseVoice-Small
- Single `pip install sherpa-onnx` — no OpenVINO, no CUDA, no complexity

### Extended Model Comparison

Additional models were benchmarked via sherpa-onnx to evaluate alternatives to SenseVoice-Small:

| Model | Params | Load (ms) | Mem (MB) | Infer zh (ms) | RTF | zh/en Quality | Multilingual |
|-------|--------|-----------|----------|---------------|-----|---------------|--------------|
| **SenseVoice-Small INT8** | 234M | 2,299 | 310 | **91** | **0.016** | Good | Excellent (50+ langs) |
| Qwen3-ASR-0.6B INT8 | 600M | 4,475 | 893 | 913 | 0.163 | Better (punctuation) | Excellent (30+ langs) |
| FireRedASR2-CTC INT8 | ~1B | 1,509 | 103 | 681 | 0.122 | Good | Poor (zh+en only) |

Key observations:
- **SenseVoice-Small is 10x faster** than Qwen3-ASR and 7.5x faster than FireRedASR2-CTC
- Qwen3-ASR produces slightly higher quality Chinese output (with punctuation, traditional characters) but the autoregressive decoder makes it far too slow for real-time use on CPU
- FireRedASR2-CTC has the smallest memory footprint (103MB) but fails on non-zh/en languages (Japanese/Korean output is garbage)
- Qwen3-ASR-1.7B is not yet available as a sherpa-onnx ONNX export (only 0.6B exists)
- **SenseVoice-Small remains the best choice** for asr2clip: fastest inference, good multilingual support, reasonable memory usage

## Reproduction

```bash
# Clone and download models
git clone https://github.com/oaklight/asr-bench.git
cd asr-bench
bash download_models.sh

# Create environments (to avoid sherpa-onnx / onnxruntime conflicts)
conda create -n asr-bench-sherpa python=3.11 -y
conda activate asr-bench-sherpa
pip install sherpa-onnx numpy soundfile psutil tabulate

conda create -n asr-bench-ort python=3.11 -y
conda activate asr-bench-ort
pip install onnxruntime-openvino numpy soundfile psutil tabulate

# Run benchmarks
conda activate asr-bench-sherpa && python bench_sherpa.py
conda activate asr-bench-ort && python bench_ort.py

# Compare results
python compare.py
```

## License

MIT
