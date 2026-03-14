"""Profile search engine: measures load time, per-query latency, memory usage, and result quality."""

import time
import tracemalloc
import os

# --- Test queries (10 expected-good, 10 expected-poor) ---
GOOD_QUERIES = [
    "cristina lopes",
    "Thornton",
    "ACM",
    "master of software engineering",
    "information retrieval",
    "AI Club",
    "software engineering",
    "data structure",
    "algorithms",
    "Donald Bren Hall"
]

POOR_QUERIES = [
    "ICS 33",
    "web crawler",
    "professor",
    "UCI admissions",
    "campus map directions",
    "machine learning",
    "computer science",
    "how to apply",
    "python programming",
    "artificial intelligence"
]

ALL_QUERIES = [(q, "good") for q in GOOD_QUERIES] + [(q, "poor") for q in POOR_QUERIES]


def main():
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[0]

    t0 = time.perf_counter()
    from search import load_index, search
    idx = load_index()
    load_time = time.perf_counter() - t0

    mem_after = tracemalloc.get_traced_memory()[0]
    mem_peak = tracemalloc.get_traced_memory()[1]

    index_files = ["postings.bin", "offsets.json", "tokens.txt", "offsets_compact.bin",
                   "doc_lengths.json", "doc_map.json", "pagerank.json"]
    index_file_size = sum(os.path.getsize(f"index/{fn}") for fn in index_files
                          if os.path.exists(f"index/{fn}"))

    print("=" * 70)
    print("SEARCH ENGINE PROFILE")
    print("=" * 70)
    print(f"Index files size:     {index_file_size / 1e6:.1f} MB (all index files)")
    print(f"Memory used:          {(mem_after - mem_before) / 1e6:.1f} MB")
    print(f"Peak memory:          {mem_peak / 1e6:.1f} MB")
    print(f"Index load time:      {load_time:.2f} s")
    print(f"Mem / index ratio:    {(mem_after - mem_before) / index_file_size:.2f}x")
    print()

    # Per-query profiling
    print(f"{'Category':<6} {'Query':<40} {'Time (ms)':>10} {'Results':>8}")
    print("-" * 70)

    timings = {"good": [], "poor": []}

    for query, category in ALL_QUERIES:
        t0 = time.perf_counter()
        results = search(query, idx, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        timings[category].append(elapsed_ms)
        n = len(results)
        print(f"{category:<6} {query:<40} {elapsed_ms:>10.2f} {n:>8}")

    print("-" * 70)
    print()

    for cat in ("good", "poor"):
        ts = timings[cat]
        print(f"{cat.upper()} queries:  avg={sum(ts)/len(ts):.2f} ms  "
              f"min={min(ts):.2f} ms  max={max(ts):.2f} ms")

    all_ts = timings["good"] + timings["poor"]
    print(f"OVERALL:          avg={sum(all_ts)/len(all_ts):.2f} ms  "
          f"min={min(all_ts):.2f} ms  max={max(all_ts):.2f} ms")
    print()

    # Detailed results
    print("=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    for query, category in ALL_QUERIES:
        results = search(query, idx, top_k=5)
        print(f"\n[{category.upper()}] '{query}'")
        if not results:
            print("  No results.")
        else:
            for i, (url, score) in enumerate(results, 1):
                print(f"  {i}. {url}  ({score:.2f})")

    idx["postings_fh"].close()


if __name__ == "__main__":
    main()
