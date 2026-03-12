"""Convert the full JSON index into a disk-friendly format:
  - index/postings.bin  : one posting list per line as compact JSON
  - index/offsets.json  : token → [byte_offset, length] into postings.bin
  - index/doc_lengths.json : doc_id → total token count (for BM25)

This lets the search engine load only the small offsets file into memory
and seek into postings.bin on demand, keeping memory far below index size.
"""

import json
import os


def build_doc_lengths(index):
    """Sum tf across all terms for each document to get document length."""
    lengths = {}
    for postings in index.values():
        for p in postings:
            doc_id = p["doc_id"]
            lengths[doc_id] = lengths.get(doc_id, 0) + p["tf"]
    return lengths


def main():
    print("Loading full index...")
    with open("index/index.json") as f:
        index = json.load(f)
    print(f"  {len(index):,} tokens loaded")

    # Build document lengths
    print("Computing document lengths...")
    doc_lengths = build_doc_lengths(index)
    with open("index/doc_lengths.json", "w") as f:
        json.dump(doc_lengths, f)
    print(f"  {len(doc_lengths):,} documents")

    # Write postings.bin and offsets
    print("Writing postings.bin and offsets.json...")
    offsets = {}
    with open("index/postings.bin", "wb") as f:
        for token in sorted(index.keys()):
            postings = index[token]
            # Compact format: list of [doc_id, tf, important_flag]
            compact = [[p["doc_id"], p["tf"], 1 if p.get("important") else 0] for p in postings]
            line = json.dumps(compact, separators=(",", ":")).encode("utf-8") + b"\n"
            offset = f.tell()
            f.write(line)
            offsets[token] = [offset, len(line)]

    with open("index/offsets.json", "w") as f:
        json.dump(offsets, f)

    # Report sizes
    for name in ["offsets.json", "postings.bin", "doc_lengths.json"]:
        path = os.path.join("index", name)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {name}: {size_mb:.1f} MB")

    print("Done.")


if __name__ == "__main__":
    main()
