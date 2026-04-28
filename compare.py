"""Compare benchmark results from sherpa-onnx and onnxruntime.

Reads results/*.json and outputs a Markdown comparison table.
Can run in either conda environment (only needs json, tabulate).
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_results() -> tuple[dict | None, dict | None]:
    sherpa = None
    ort = None

    sherpa_path = RESULTS_DIR / "sherpa_results.json"
    ort_path = RESULTS_DIR / "ort_results.json"

    if sherpa_path.exists():
        with open(sherpa_path) as f:
            sherpa = json.load(f)
    else:
        print(f"[warn] {sherpa_path} not found")

    if ort_path.exists():
        with open(ort_path) as f:
            ort = json.load(f)
    else:
        print(f"[warn] {ort_path} not found")

    return sherpa, ort


def format_table(headers: list[str], rows: list[list]) -> str:
    """Format a simple Markdown table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def fmt_row(cells):
        return (
            "| "
            + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells))
            + " |"
        )

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * w for w in col_widths) + " |",
    ]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def compare_asr(sherpa: dict | None, ort: dict | None) -> str:
    """Compare ASR results."""
    lines = ["## SenseVoice-Small ASR Benchmark", ""]

    headers = [
        "Engine",
        "Load (ms)",
        "Infer Mean (ms)",
        "Infer P95 (ms)",
        "RTF",
        "Mem Peak (MB)",
    ]
    rows = []

    if sherpa and "asr" in sherpa:
        asr = sherpa["asr"]
        for label, data in asr.get("files", {}).items():
            rows.append(
                [
                    f"sherpa-onnx [{label}]",
                    f"{asr['model_load_ms']:.0f}",
                    f"{data['latency_mean_ms']:.1f}",
                    f"{data['latency_p95_ms']:.1f}",
                    f"{data['rtf']:.4f}",
                    f"{data['mem_peak_mb']:.0f}",
                ]
            )

    if ort and "asr" in ort:
        for provider_name, data in ort["asr"].items():
            if not data.get("success"):
                rows.append(
                    [
                        f"ort {provider_name}",
                        "FAIL",
                        data.get("error", "")[:40],
                        "",
                        "",
                        "",
                    ]
                )
                continue
            rows.append(
                [
                    f"ort {provider_name}",
                    f"{data['model_load_ms']:.0f}",
                    f"{data['latency_mean_ms']:.1f}",
                    f"{data['latency_p95_ms']:.1f}",
                    "N/A*",
                    f"{data['mem_peak_mb']:.0f}",
                ]
            )

    lines.append(format_table(headers, rows))
    lines.append("")
    lines.append(
        "*ort ASR uses dummy input (no fbank preprocessing), RTF not comparable"
    )
    return "\n".join(lines)


def compare_vad(sherpa: dict | None, ort: dict | None) -> str:
    """Compare VAD results."""
    lines = ["## Silero VAD Benchmark", ""]

    headers = [
        "Engine",
        "File",
        "Load (ms)",
        "Infer Mean (ms)",
        "Infer P95 (ms)",
        "Segments",
        "Mem (MB)",
    ]
    rows = []

    if sherpa and "vad" in sherpa:
        vad = sherpa["vad"]
        for label, data in vad.get("files", {}).items():
            rows.append(
                [
                    "sherpa-onnx",
                    label,
                    f"{vad['model_load_ms']:.0f}",
                    f"{data['latency_mean_ms']:.1f}",
                    f"{data['latency_p95_ms']:.1f}",
                    str(data["num_segments"]),
                    f"{data['mem_peak_mb']:.0f}",
                ]
            )

    if ort and "vad" in ort:
        for provider_name, pdata in ort["vad"].items():
            if not pdata.get("success"):
                continue
            for label, data in pdata.get("files", {}).items():
                rows.append(
                    [
                        f"ort {provider_name}",
                        label,
                        f"{pdata['model_load_ms']:.0f}",
                        f"{data['latency_mean_ms']:.1f}",
                        f"{data['latency_p95_ms']:.1f}",
                        str(data["num_segments"]),
                        f"{data['mem_peak_mb']:.0f}",
                    ]
                )

    lines.append(format_table(headers, rows))
    return "\n".join(lines)


def compare_combined(sherpa: dict | None) -> str:
    """Show combined VAD+ASR pipeline results (sherpa-onnx only)."""
    lines = ["## Combined VAD + ASR Pipeline (sherpa-onnx only)", ""]

    if not sherpa or "combined" not in sherpa:
        lines.append("No combined pipeline results available.")
        return "\n".join(lines)

    combined = sherpa["combined"]
    headers = ["File", "Load (ms)", "E2E Mean (ms)", "E2E P95 (ms)", "RTF", "Mem (MB)"]
    rows = []

    for label, data in combined.get("files", {}).items():
        rows.append(
            [
                label,
                f"{combined['total_load_ms']:.0f}",
                f"{data['latency_mean_ms']:.1f}",
                f"{data['latency_p95_ms']:.1f}",
                f"{data['rtf']:.4f}",
                f"{data['mem_peak_mb']:.0f}",
            ]
        )

    lines.append(format_table(headers, rows))
    return "\n".join(lines)


def main():
    sherpa, ort = load_results()

    if not sherpa and not ort:
        print("No results found. Run bench_sherpa.py and/or bench_ort.py first.")
        return

    sections = [
        "# ASR/VAD Inference Engine Benchmark Results",
        "",
    ]

    if sherpa:
        sections.append(
            f"- sherpa-onnx: {sherpa.get('timestamp', 'N/A')}, threads={sherpa.get('num_threads')}"
        )
    if ort:
        sections.append(
            f"- onnxruntime: {ort.get('ort_version', 'N/A')}, "
            f"providers={ort.get('available_providers', [])}, "
            f"threads={ort.get('num_threads')}"
        )
    sections.append("")

    sections.append(compare_asr(sherpa, ort))
    sections.append("")
    sections.append(compare_vad(sherpa, ort))
    sections.append("")
    sections.append(compare_combined(sherpa))

    report = "\n".join(sections)
    print(report)

    out_path = RESULTS_DIR / "comparison.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
