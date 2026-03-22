"""Search engine: loads the inverted index and answers queries using BM25 + PageRank."""

import array
import json
import math
import re
import struct
import time
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def tokenize(text):
    """Extract alphanumeric tokens and apply Porter stemming (matches indexer)."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    return [stemmer.stem(t.lower()) for t in tokens]


_BM25_B = 0.75  # BM25 length normalization strength (0 = no normalization, 1 = full)


def _load_tokens_compact(path):
    """Load tokens.txt as a single bytes object + array of line-start offsets.
    Uses ~70 MB instead of ~430 MB from a Python list of strings.
    data.split() is a C-level call; we then compute cumulative byte offsets in Python.
    """
    with open(path, "rb") as f:
        data = f.read()
    parts = data.split(b"\n")
    if parts and not parts[-1]:  # drop trailing empty entry if file ends with \n
        parts.pop()
    offsets = array.array("I")  # uint32 — sufficient since tokens.txt is < 4 GB
    pos = 0
    for p in parts:
        offsets.append(pos)
        pos += len(p) + 1  # +1 for the '\n' separator
    return data, offsets


def _bisect_tokens(data, offsets, term):
    """Binary search for term in the compact bytes buffer. Returns index or -1.
    Slices data[start:end] for each midpoint — no Python string objects created.
    """
    term_b = term.encode("utf-8")
    n = len(offsets)
    lo, hi = 0, n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start = offsets[mid]
        end = offsets[mid + 1] - 1 if mid + 1 < n else len(data)  # exclude '\n'
        token = data[start:end]
        if token == term_b:
            return mid
        elif token < term_b:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def load_index():
    """Load the lightweight offset index and doc metadata into memory.
    Postings are NOT loaded — they are read from disk on demand per query.

    What lives in memory:
      - tokens_bytes / tokens_offsets : compact lexicon (~70 MB)
      - offset_vals / length_vals     : byte offsets into postings.bin (~57 MB)
      - doc_map, doc_lengths, pr_scores: per-document metadata (~small)
    """
    print("Loading index...")

    # Compact lexicon: bytes buffer + parallel uint32 offset array
    tokens_bytes, tokens_offsets = _load_tokens_compact("index/tokens.txt")

    # Binary file: [4-byte count][n × uint64 offsets][n × uint32 lengths]
    with open("index/offsets_compact.bin", "rb") as f:
        n = struct.unpack("I", f.read(4))[0]
        offset_vals = array.array("Q")   # uint64: byte offset of each posting list
        offset_vals.fromfile(f, n)
        length_vals = array.array("I")   # uint32: byte length of each posting list
        length_vals.fromfile(f, n)

    # doc_id (str) → URL
    with open("index/doc_map.json") as f:
        doc_map = json.load(f)
    # doc_id (str) → total token count (used for BM25 length normalization)
    with open("index/doc_lengths.json") as f:
        doc_lengths = json.load(f)

    # Keep postings.bin open; seek to offsets on each query
    postings_fh = open("index/postings.bin", "rb")

    # Average document length across corpus (BM25 denominator)
    avg_len = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1

    # PageRank scores (optional — search still works without this file)
    pr_scores = {}
    try:
        with open("index/pagerank.json") as f:
            pr_scores = json.load(f)
        print(f"PageRank loaded: {len(pr_scores):,} scores")
    except FileNotFoundError:
        pass

    print(f"Index loaded: {n:,} unique tokens, {len(doc_map):,} documents\n")
    return {
        "tokens_bytes": tokens_bytes,
        "tokens_offsets": tokens_offsets,
        "offset_vals": offset_vals,
        "length_vals": length_vals,
        "doc_map": doc_map,
        "N": len(doc_map),
        "postings_fh": postings_fh,
        "doc_lengths": doc_lengths,
        "avg_len": avg_len,
        "pr_scores": pr_scores,
    }


def read_postings(idx, term):
    """Locate term in the lexicon via binary search, then seek to its posting list on disk."""
    i = _bisect_tokens(idx["tokens_bytes"], idx["tokens_offsets"], term)
    if i < 0:
        return None  # term not in index
    offset = idx["offset_vals"][i]
    length = idx["length_vals"][i]
    idx["postings_fh"].seek(offset)
    data = idx["postings_fh"].read(length)
    # Each entry is [doc_id, tf, important_flag]
    return json.loads(data)


def search(query, idx, top_k=5):
    """Score documents using BM25 tf-idf with importance boost, bigram bonus, and PageRank.

    Scoring layers (additive except PageRank):
      1. BM25 tf-idf  — length-normalized term frequency × IDF
      2. Importance boost (×2) — term appears in title/heading/bold
      3. Bigram bonus (×1.5 IDF) — consecutive query terms appear together
      4. PageRank boost (×1 to ×1.5) — multiplicative, proportional to PR score
    """
    raw_terms = tokenize(query)
    if not raw_terms:
        return []

    # Deduplicate while preserving order
    query_terms = list(dict.fromkeys(raw_terms))

    # Boolean AND: fetch posting lists; return empty if any term is missing
    posting_lists = []
    for term in query_terms:
        postings = read_postings(idx, term)
        if postings is None:
            return []  # AND semantics — all terms must appear
        posting_lists.append(postings)

    # Intersect to find documents containing all query terms
    common_docs = set(p[0] for p in posting_lists[0])
    for pl in posting_lists[1:]:
        common_docs &= set(p[0] for p in pl)

    if not common_docs:
        return []

    N = idx["N"]
    doc_lengths = idx["doc_lengths"]
    avg_len = idx["avg_len"]

    # Score each term: BM25 tf-idf × importance boost
    scores = {}
    for term, postings in zip(query_terms, posting_lists):
        df = len(postings)                              # document frequency
        idf = math.log10(N / df)                        # inverse document frequency
        for doc_id, tf, important in postings:
            if doc_id not in common_docs:
                continue
            doc_len = doc_lengths.get(str(doc_id), avg_len)
            tf_norm = tf / (1 - _BM25_B + _BM25_B * doc_len / avg_len)  # BM25 normalization
            tf_log = 1 + math.log10(tf_norm) if tf_norm > 0 else 0       # log-frequency weight
            boost = 2.0 if important else 1.0           # double weight for title/heading terms
            scores[doc_id] = scores.get(doc_id, 0) + tf_log * idf * boost

    # Bigram bonus: reward documents where adjacent query terms co-occur
    if len(query_terms) > 1:
        for i in range(len(query_terms) - 1):
            bigram = f"{query_terms[i]} {query_terms[i+1]}"
            bi_postings = read_postings(idx, bigram)
            if bi_postings:
                df = len(bi_postings)
                idf = math.log10(N / df)
                for doc_id, tf, _ in bi_postings:
                    if doc_id in common_docs:
                        tf_log = 1 + math.log10(tf) if tf > 0 else 0
                        scores[doc_id] = scores.get(doc_id, 0) + tf_log * idf * 1.5

    # PageRank multiplicative boost: high-authority pages score up to 1.5× higher
    pr_scores = idx.get("pr_scores", {})
    if pr_scores:
        for doc_id in scores:
            pr = pr_scores.get(str(doc_id), 0.0)  # normalized [0, 1]
            scores[doc_id] *= (1 + 0.5 * pr)

    # Sort by score descending; deduplicate by base URL to avoid near-duplicate results
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    doc_map = idx["doc_map"]
    seen_base = set()
    results = []
    for doc_id, score in ranked:
        url = doc_map[str(doc_id)]
        url = url.get("url") if isinstance(url, dict) else url
        # Normalize URL: strip index.html and fragment so /foo/index.html == /foo/
        base_url = re.sub(r'/index\.[a-zA-Z]+$', '/', url.split("#")[0]).rstrip('/')
        if base_url not in seen_base:
            seen_base.add(base_url)
            results.append((url, score))
        if len(results) == top_k:
            break
    return results


def run_query(query, idx, top_k=5):
    """Run a single query, print results and elapsed time to stdout."""
    t0 = time.perf_counter()
    results = search(query, idx, top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Query: '{query}'  ({elapsed_ms:.1f} ms)")
    if not results:
        print("  No results found.")
    else:
        for i, (url, score) in enumerate(results, 1):
            print(f"  {i}. {url}  (score: {score:.4f})")
    print()
    return results


def main():
    """Interactive search loop — type a query, press Enter, see top-5 results."""
    idx = load_index()

    while True:
        try:
            query = input("Search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            idx["postings_fh"].close()
            return
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            idx["postings_fh"].close()
            return
        run_query(query, idx, top_k=5)


if __name__ == "__main__":
    main()
