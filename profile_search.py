"""Profile search engine: measures load time, per-query latency, memory usage, and result quality."""

import time
import tracemalloc
import os

# Good queries: topics well-represented in the ICS corpus — expected to return relevant results
GOOD_QUERIES = [
    "cristina lopes",           # known faculty member
    "Thornton",                 # known faculty member
    "ACM",                      # well-known org mentioned across many pages
    "master of software engineering",  # specific degree program
    "information retrieval",    # active research area at ICS
    "AI Club",                  # student org
    "software engineering",     # core ICS program
    "data structure",           # common CS course topic
    "algorithms",               # common CS course topic
    "Donald Bren Hall",         # ICS building
]

# Poor queries: vague, off-domain, or ambiguous — expected to return less relevant results
POOR_QUERIES = [
    "ICS 33",                   # course number with no dedicated page
    "web crawler",              # implementation detail, not a content topic
    "professor",                # too generic — matches any faculty mention
    "UCI admissions",           # mostly links to non-ICS admissions pages
    "campus map directions",    # off-domain; returns contact/directions boilerplate
    "machine learning",         # broad term, results scatter across unrelated pages
    "computer science",         # too broad — matches nearly every page
    "how to apply",             # generic phrase pulled from many unrelated pages
    "python programming",       # common term, not tied to a specific page
    "artificial intelligence",  # broad; matches news articles more than course pages
]

# Combine into a single list with category labels for unified iteration
ALL_QUERIES = [(q, "good") for q in GOOD_QUERIES] + [(q, "poor") for q in POOR_QUERIES]


def main():
    # Start memory tracking before importing/loading anything
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[0]

    # Time the full index load (reads lexicon + metadata into memory)
    t0 = time.perf_counter()
    from search import load_index, search
    idx = load_index()
    load_time = time.perf_counter() - t0

    # Capture memory after load to measure how much the index occupies in RAM
    mem_after = tracemalloc.get_traced_memory()[0]
    mem_peak = tracemalloc.get_traced_memory()[1]

    # Sum all on-disk index file sizes to compare against in-memory footprint
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
    # Ratio < 1.0 means memory footprint is smaller than the on-disk index (required by spec)
    print(f"Mem / index ratio:    {(mem_after - mem_before) / index_file_size:.2f}x")
    print()

    # Per-query latency: time each search call individually
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

    # Aggregate latency stats by category
    for cat in ("good", "poor"):
        ts = timings[cat]
        print(f"{cat.upper()} queries:  avg={sum(ts)/len(ts):.2f} ms  "
              f"min={min(ts):.2f} ms  max={max(ts):.2f} ms")

    all_ts = timings["good"] + timings["poor"]
    print(f"OVERALL:          avg={sum(all_ts)/len(all_ts):.2f} ms  "
          f"min={min(all_ts):.2f} ms  max={max(all_ts):.2f} ms")
    print()

    # Print top-5 results for each query to evaluate result quality
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

    # Close the postings file handle opened by load_index()
    idx["postings_fh"].close()


if __name__ == "__main__":
    main()
