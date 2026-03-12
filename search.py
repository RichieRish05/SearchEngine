import json
import math
import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Common English stopwords - filtered at query time
STOPWORDS = frozenset(
    "a an the is it in on at to for of and or but not by with from as be "
    "this that was were are been has have had do does did will would shall "
    "should can could may might its his her he she they them their you your "
    "we our i me my so if no nor too also very just how what when where who "
    "which all each any some no".split()
)


def tokenize(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    
    return [stemmer.stem(t.lower()) for t in tokens]


def load_index():
    """Load the lightweight offset index and doc metadata into memory.
    Postings are read from disk on demand per query."""
    print("Loading index...")
    with open("index/offsets.json") as f:
        offsets = json.load(f)
    with open("index/doc_map.json") as f:
        doc_map = json.load(f)
    # Open postings file handle (kept open for seeks)
    postings_fh = open("index/postings.bin", "rb")

    print(f"Index loaded: {len(offsets):,} unique tokens, {len(doc_map):,} documents\n")
    return {
        "offsets": offsets,
        "doc_map": doc_map,
        "N": len(doc_map),
        "postings_fh": postings_fh,
    }


def read_postings(idx, term):
    """Read a single posting list from disk by seeking to its offset."""
    entry = idx["offsets"].get(term)
    if entry is None:
        return None
    offset, length = entry
    idx["postings_fh"].seek(offset)
    data = idx["postings_fh"].read(length)
    # Each entry is [doc_id, tf, important_flag]
    return json.loads(data)


def search(query, idx, top_k=5):
    """Log-frequency weighted tf-idf search with importance boost and stopword filtering."""
    raw_terms = tokenize(query)
    if not raw_terms:
        return []

    # Filter stopwords but keep if all terms are stopwords
    filtered = [t for t in raw_terms if t not in STOPWORDS]
    query_terms = list(dict.fromkeys(filtered if filtered else raw_terms))

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

    # Log-frequency weighting with idf and importance boost
    N = idx["N"]

    scores = {}
    for term, postings in zip(query_terms, posting_lists):
        df = len(postings)
        idf = math.log10(N / df)
        for doc_id, tf, important in postings:
            if doc_id not in common_docs:
                continue
            tf_log = 1 + math.log10(tf) if tf > 0 else 0
            boost = 1.5 if important else 1.0
            scores[doc_id] = scores.get(doc_id, 0) + tf_log * idf * boost

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    doc_map = idx["doc_map"]
    return [(doc_map[str(doc_id)], score) for doc_id, score in ranked[:top_k]]


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
