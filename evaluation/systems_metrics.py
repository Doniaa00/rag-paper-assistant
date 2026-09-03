"""
Step 9d: stage-level latency and resource-footprint measurement.

Instruments the full pipeline (dense -> sparse -> RRF fuse -> rerank ->
generate) across all 30 eval-set questions (including negatives -- refusal
responses still cost real latency) and captures, per question:
  - per-stage wall-clock latency (dense/sparse/fusion/rerank, ms)
  - generation TTFT (s) and steady-state throughput (tokens/sec), from
    Ollama's own internal timing breakdown (see generate_with_metrics)
  - total end-to-end response time
  - peak system RAM and peak GPU VRAM during that question, sampled by a
    background thread while the question runs

A throwaway warm-up query runs first and is entirely discarded (not timed,
not logged) -- per the Step 8 finding that cold model load/CUDA-context
init inflates the first call by ~2 orders of magnitude versus steady state.

Writes per-question detail to data/evaluation/systems_metrics_results.csv
and prints an aggregate summary, overall and broken down by question_type.
"""

import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "retrieval"))

import psutil  # noqa: E402

from rrf_fusion import dense_search, sparse_search, rrf_fuse  # noqa: E402
from rerank import rerank  # noqa: E402
from generation.local_ollama import OllamaBackend  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
EVAL_SET_CSV = DATA_DIR / "evaluation" / "eval_set.csv"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"
RESULTS_CSV = DATA_DIR / "evaluation" / "systems_metrics_results.csv"

DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RRF_K = 60
FUSED_TOP_N = 10
RERANK_TOP_N = 5

SAMPLER_INTERVAL_S = 0.3
WARMUP_QUERY = "How does SMOTE generate synthetic samples?"

STAGE_FIELDS = ["dense_ms", "sparse_ms", "fusion_ms", "rerank_ms"]
GEN_FIELDS = ["ttft_seconds", "tokens_per_second", "generation_seconds"]
TOTAL_FIELDS = ["total_response_time_ms", "peak_ram_gb", "peak_vram_gb"]

FIELDNAMES = (
    ["question_id", "question_type", "pillar"]
    + STAGE_FIELDS + GEN_FIELDS + ["eval_count_tokens"] + TOTAL_FIELDS
)


class ResourceSampler:
    """Background thread sampling system RAM (psutil) and GPU VRAM
    (nvidia-smi) at a fixed interval, for peak-usage measurement across a
    single question's pipeline run. Uses system-wide figures rather than
    per-process, consistent with how VRAM was measured in Steps 7/8 --
    Ollama runs as a separate OS process from this one, so only a
    system-wide reading captures both it and the in-process reranker/BGE-M3."""

    def __init__(self, interval_s: float = SAMPLER_INTERVAL_S):
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread = None
        self.ram_samples_gb = []
        self.vram_samples_gb = []

    def _sample_once(self):
        try:
            self.ram_samples_gb.append(psutil.virtual_memory().used / (1024 ** 3))
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            vram_mb = int(result.stdout.strip().splitlines()[0])
            self.vram_samples_gb.append(vram_mb / 1024)
        except Exception:
            pass

    def _loop(self):
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.interval_s)

    def start(self):
        self.ram_samples_gb = []
        self.vram_samples_gb = []
        self._stop_event.clear()
        self._sample_once()  # guarantee at least one sample even for very fast questions
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def peak_ram_gb(self):
        return max(self.ram_samples_gb) if self.ram_samples_gb else None

    def peak_vram_gb(self):
        return max(self.vram_samples_gb) if self.vram_samples_gb else None


def load_questions():
    with open(EVAL_SET_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_chunks_by_id():
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"]: json.loads(line) for line in f}


def run_instrumented_pipeline(query: str, chunks_by_id: dict, backend: OllamaBackend):
    t0 = time.perf_counter()
    dense_ids = dense_search(query, top_k=DENSE_TOP_K)
    t1 = time.perf_counter()

    sparse_ids = sparse_search(query, top_k=SPARSE_TOP_K)
    t2 = time.perf_counter()

    fused_pairs = rrf_fuse(dense_ids, sparse_ids, k=RRF_K)[:FUSED_TOP_N]
    t3 = time.perf_counter()

    fused_ids = [cid for cid, _score in fused_pairs]
    candidate_chunks = [chunks_by_id[cid] for cid in fused_ids]
    reranked_pairs = rerank(query, candidate_chunks, top_n=RERANK_TOP_N)
    t4 = time.perf_counter()

    evidence_chunks = [chunk for chunk, _score in reranked_pairs]
    gen_result = backend.generate_with_metrics(query, evidence_chunks)
    t5 = time.perf_counter()

    timings = {
        "dense_ms": (t1 - t0) * 1000,
        "sparse_ms": (t2 - t1) * 1000,
        "fusion_ms": (t3 - t2) * 1000,
        "rerank_ms": (t4 - t3) * 1000,
        "total_response_time_ms": (t5 - t0) * 1000,
    }
    return timings, gen_result


def main():
    questions = load_questions()
    chunks_by_id = load_chunks_by_id()
    backend = OllamaBackend()

    print("Running warm-up query (discarded, not timed or logged)...", file=sys.stderr)
    run_instrumented_pipeline(WARMUP_QUERY, chunks_by_id, backend)
    print("Warm-up complete -- beginning timed run.\n", file=sys.stderr)

    sampler = ResourceSampler()
    rows = []

    for i, q in enumerate(questions, start=1):
        qid = q["question_id"]
        query = q["question"]
        qtype = q["question_type"].strip()
        print(f"[{i}/{len(questions)}] {qid} ({qtype}): {query[:55]}...", file=sys.stderr)

        sampler.start()
        timings, gen_result = run_instrumented_pipeline(query, chunks_by_id, backend)
        sampler.stop()

        rows.append({
            "question_id": qid,
            "question_type": qtype,
            "pillar": q["pillar"],
            "dense_ms": round(timings["dense_ms"], 2),
            "sparse_ms": round(timings["sparse_ms"], 2),
            "fusion_ms": round(timings["fusion_ms"], 3),
            "rerank_ms": round(timings["rerank_ms"], 2),
            "ttft_seconds": round(gen_result["ttft_seconds"], 3),
            "tokens_per_second": round(gen_result["tokens_per_second"], 1),
            "generation_seconds": round(gen_result["generation_seconds"], 3),
            "eval_count_tokens": gen_result["eval_count"],
            "total_response_time_ms": round(timings["total_response_time_ms"], 2),
            "peak_ram_gb": round(sampler.peak_ram_gb(), 3) if sampler.peak_ram_gb() is not None else "",
            "peak_vram_gb": round(sampler.peak_vram_gb(), 3) if sampler.peak_vram_gb() is not None else "",
        })

    write_csv(rows)
    print_summary(rows)


def write_csv(rows: list):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _stats_line(rows: list, field: str):
    values = [r[field] for r in rows if r[field] != ""]
    if not values:
        return "avg=n/a  min=n/a  max=n/a"
    return f"avg={statistics.mean(values):>9.3f}  min={min(values):>9.3f}  max={max(values):>9.3f}"


def _print_group_table(rows: list, label: str):
    print(f"\n{label} (n={len(rows)}):")
    for field in STAGE_FIELDS:
        print(f"  {field:<28} {_stats_line(rows, field)}")
    for field in GEN_FIELDS:
        print(f"  {field:<28} {_stats_line(rows, field)}")
    print(f"  {'total_response_time_ms':<28} {_stats_line(rows, 'total_response_time_ms')}")
    print(f"  {'peak_ram_gb':<28} {_stats_line(rows, 'peak_ram_gb')}")
    print(f"  {'peak_vram_gb':<28} {_stats_line(rows, 'peak_vram_gb')}")


def print_summary(rows: list):
    print("\n" + "=" * 90)
    print("SYSTEMS METRICS -- AGGREGATE SUMMARY (Section 5.1)")
    print("=" * 90)
    print(f"\nWarm-up query excluded from all timing above (ran once, discarded before this loop started).")

    _print_group_table(rows, "OVERALL")

    for qtype in ["factual", "comparative", "negative"]:
        subset = [r for r in rows if r["question_type"] == qtype]
        if subset:
            _print_group_table(subset, qtype.upper())

    print("\n" + "-" * 90)
    print("OUTLIER FLAGS (total_response_time_ms or peak_vram_gb > mean + 1.5 * stdev)")
    print("-" * 90)

    total_times = [r["total_response_time_ms"] for r in rows]
    mean_total = statistics.mean(total_times)
    stdev_total = statistics.stdev(total_times) if len(total_times) > 1 else 0
    threshold_total = mean_total + 1.5 * stdev_total

    vram_values = [r["peak_vram_gb"] for r in rows if r["peak_vram_gb"] != ""]
    mean_vram = statistics.mean(vram_values) if vram_values else 0
    stdev_vram = statistics.stdev(vram_values) if len(vram_values) > 1 else 0
    threshold_vram = mean_vram + 1.5 * stdev_vram

    flagged = False
    for r in rows:
        reasons = []
        if r["total_response_time_ms"] > threshold_total:
            reasons.append(f"total_response_time={r['total_response_time_ms']:.0f}ms (threshold {threshold_total:.0f}ms)")
        if r["peak_vram_gb"] != "" and r["peak_vram_gb"] > threshold_vram:
            reasons.append(f"peak_vram={r['peak_vram_gb']:.2f}GB (threshold {threshold_vram:.2f}GB)")
        if reasons:
            flagged = True
            print(f"  {r['question_id']} ({r['question_type']}): " + "; ".join(reasons))

    if not flagged:
        print("  None.")

    print(f"\nFull per-question detail written to: {RESULTS_CSV}")
    print("=" * 90)


if __name__ == "__main__":
    main()
