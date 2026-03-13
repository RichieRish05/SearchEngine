import array
import bisect
import json
import math
import re
import struct
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def tokenize(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    return [stemmer.stem(t.lower()) for t in tokens]


_BM25_B = 0.75  # length normalization parameter


def load_index():
    """Load the lightweight offset index and doc metadata into memory.
    Postings are read from disk on demand per query.

    Uses a sorted token list + compact binary arrays (~56 MB) instead of
    the offsets.json dict (~190 MB) to keep memory below index file size.
    """
    print("Loading index...")
    # Compact offsets: sorted tokens + parallel binary arrays
    with open("index/tokens.txt", encoding="utf-8") as f:
        tokens_list = f.read().splitlines()
    with open("index/offsets_compact.bin", "rb") as f:
        n = struct.unpack("I", f.read(4))[0]
        offset_vals = array.array("Q")
        offset_vals.fromfile(f, n)
        length_vals = array.array("I")
        length_vals.fromfile(f, n)

    with open("index/doc_map.json") as f:
        doc_map = json.load(f)
    with open("index/doc_lengths.json") as f:
        doc_lengths = json.load(f)
    postings_fh = open("index/postings.bin", "rb")

    avg_len = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1

    print(
        f"Index loaded: {n:,} unique tokens, {len(doc_map):,} documents\n"
    )
    return {
        "tokens_list": tokens_list,
        "offset_vals": offset_vals,
        "length_vals": length_vals,
        "doc_map": doc_map,
        "N": len(doc_map),
        "postings_fh": postings_fh,
        "doc_lengths": doc_lengths,
        "avg_len": avg_len,
    }


def read_postings(idx, term):
    """Read a single posting list from disk by seeking to its offset."""
    tokens_list = idx["tokens_list"]
    i = bisect.bisect_left(tokens_list, term)
    if i >= len(tokens_list) or tokens_list[i] != term:
        return None
    offset = idx["offset_vals"][i]
    length = idx["length_vals"][i]
    idx["postings_fh"].seek(offset)
    data = idx["postings_fh"].read(length)
    # Each entry is [doc_id, tf, important_flag]
    return json.loads(data)


def search(query, idx, top_k=5):
    """Log-frequency weighted tf-idf search with importance boost."""
    raw_terms = tokenize(query)
    if not raw_terms:
        return []

    query_terms = list(dict.fromkeys(raw_terms))

    # Read posting lists from disk
    posting_lists = []
    for term in query_terms:
        postings = read_postings(idx, term)
        if postings is None:
            return []  # AND semantics: missing term → no results
        posting_lists.append(postings)

    # Intersect document sets
    common_docs = set(p[0] for p in posting_lists[0])
    for pl in posting_lists[1:]:
        common_docs &= set(p[0] for p in pl)

    if not common_docs:
        return []

    # BM25 length-normalized tf-idf with importance boost
    N = idx["N"]
    doc_lengths = idx["doc_lengths"]
    avg_len = idx["avg_len"]

    scores = {}
    for term, postings in zip(query_terms, posting_lists):
        df = len(postings)
        idf = math.log10(N / df)
        for doc_id, tf, important in postings:
            if doc_id not in common_docs:
                continue
            doc_len = doc_lengths.get(str(doc_id), avg_len)
            tf_norm = tf / (1 - _BM25_B + _BM25_B * doc_len / avg_len)
            tf_log = 1 + math.log10(tf_norm) if tf_norm > 0 else 0
            boost = 2.0 if important else 1.0
            scores[doc_id] = scores.get(doc_id, 0) + tf_log * idf * boost

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    doc_map = idx["doc_map"]
    seen_base = set()
    results = []
    for doc_id, score in ranked:
        url = doc_map[str(doc_id)]
        base_url = re.sub(r'/index\.[a-zA-Z]+$', '/', url.split("#")[0]).rstrip('/')
        if base_url not in seen_base:
            seen_base.add(base_url)
            results.append((url, score))
        if len(results) == top_k:
            break
    return results


def run_query(query, idx, top_k=5):
    results = search(query, idx, top_k)
    print(f"Query: '{query}'")
    if not results:
        print("  No results found.")
    else:
        for i, (url, score) in enumerate(results, 1):
            print(f"  {i}. {url}  (score: {score:.4f})")
    print()
    return results


def main():
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
