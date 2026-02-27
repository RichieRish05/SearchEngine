import json
import math
import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def tokenize(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    return [stemmer.stem(t.lower()) for t in tokens]


def load_index():
    print("Loading index...")
    with open("index/index.json") as f:
        index = json.load(f)
    with open("index/doc_map.json") as f:
        doc_map = json.load(f)
    print(f"Index loaded: {len(index):,} unique tokens, {len(doc_map):,} documents\n")
    return index, doc_map


def search(query, index, doc_map, top_k=5):
    # Stem and deduplicate tokens of serach query
    query_terms = list(dict.fromkeys(tokenize(query)))

    if not query_terms:
        return []

    # Retrieve posting lists; any missing term → empty result (AND semantics)
    posting_lists = []
    for term in query_terms:
        if term not in index:
            return []
        posting_lists.append(index[term])

    # Intersect document sets
    common_docs = set(p["doc_id"] for p in posting_lists[0])
    for pl in posting_lists[1:]:
        common_docs &= set(p["doc_id"] for p in pl)

    if not common_docs:
        return []

    # Score with tf-idf; boost terms appearing in important tags
    N = len(doc_map)
    scores = {}
    for term, postings in zip(query_terms, posting_lists):
        df = len(postings)
        idf = math.log(N / df)
        for posting in postings:
            doc_id = posting["doc_id"]
            if doc_id in common_docs:
                tf = posting["tf"]
                boost = 1.5 if posting.get("important") else 1.0
                scores[doc_id] = scores.get(doc_id, 0) + tf * idf * boost

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[str(doc_id)], score) for doc_id, score in ranked[:top_k]]


def run_query(query, index, doc_map, top_k=5):
    results = search(query, index, doc_map, top_k)
    print(f"Query: '{query}'")
    if not results:
        print("  No results found.")
    else:
        for i, (url, score) in enumerate(results, 1):
            print(f"  {i}. {url}  (score: {score:.4f})")
    print()
    return results


def main():
    index, doc_map = load_index()

    while True:
        try:
            query = input("Search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            return
        run_query(query, index, doc_map, top_k=5)


if __name__ == "__main__":
    main()
