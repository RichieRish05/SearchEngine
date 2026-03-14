"""Convert the full JSON index into a disk-friendly format:
  - index/postings.bin       : one posting list per line as compact JSON
  - index/offsets.json       : token → [byte_offset, length] into postings.bin
  - index/tokens.txt         : sorted token list (one per line)
  - index/offsets_compact.bin: parallel binary arrays [uint64 offsets | uint32 lengths]
  - index/doc_lengths.json   : doc_id → total token count (for BM25)

tokens.txt + offsets_compact.bin allow O(log n) bisect lookup with ~56 MB memory
(vs ~190 MB for the offsets.json dict), satisfying the memory < index size requirement.
"""

import array
import json
import os
import struct


def build_doc_lengths(index):
    """Sum tf across all unigram terms for each document to get document length.
    Bigrams (keys containing a space) are excluded to avoid inflating BM25 denominators.
    """
    lengths = {}
    for token, postings in index.items():
        if " " in token:  # skip bigrams
            continue
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

    # Write compact binary offsets for low-memory lookup
    print("Writing tokens.txt and offsets_compact.bin...")
    sorted_tokens = sorted(offsets.keys())
    offset_vals = array.array("Q", (offsets[t][0] for t in sorted_tokens))
    length_vals = array.array("I", (offsets[t][1] for t in sorted_tokens))
    with open("index/tokens.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted_tokens))
    n = len(sorted_tokens)
    with open("index/offsets_compact.bin", "wb") as f:
        f.write(struct.pack("I", n))  # 4-byte count header
        offset_vals.tofile(f)         # n × 8 bytes
        length_vals.tofile(f)         # n × 4 bytes

    # Report sizes
    for name in ["offsets.json", "postings.bin", "doc_lengths.json",
                 "tokens.txt", "offsets_compact.bin"]:
        path = os.path.join("index", name)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {name}: {size_mb:.1f} MB")

    print("Done.")

    # Compute PageRank from the link graph saved by indexer.py
    import compute_pagerank
    compute_pagerank.main()


if __name__ == "__main__":
    main()
