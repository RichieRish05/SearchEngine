"""Convert the full JSON index into a disk-friendly binary format for fast on-demand lookup:

  index/postings.bin        — all posting lists concatenated; one JSON array per line
  index/offsets.json        — token → [byte_offset, length] into postings.bin
  index/tokens.txt          — sorted token list (one per line)
  index/offsets_compact.bin — parallel binary arrays: [uint64 offsets | uint32 lengths]
  index/doc_lengths.json    — doc_id → total token count (for BM25 normalization)

tokens.txt + offsets_compact.bin allow O(log n) bisect lookup using ~70 MB RAM
(vs ~430 MB for a Python list of strings), satisfying the memory < index size requirement.

Run once after indexer.py. PageRank (compute_pagerank.py) is called automatically at the end.
"""

import array
import json
import os
import struct


def build_doc_lengths(index):
    """Sum tf across all unigram terms per document to get document length.
    Bigrams (keys with a space) are excluded to avoid inflating BM25 denominators.
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

    # Compute per-document token counts (used by BM25 in search.py)
    print("Computing document lengths...")
    doc_lengths = build_doc_lengths(index)
    with open("index/doc_lengths.json", "w") as f:
        json.dump(doc_lengths, f)
    print(f"  {len(doc_lengths):,} documents")

    # Write postings.bin: each line is a compact JSON array [[doc_id, tf, important], ...]
    # offsets.json maps each token to its [byte_offset, length] in postings.bin
    print("Writing postings.bin and offsets.json...")
    offsets = {}
    with open("index/postings.bin", "wb") as f:
        for token in sorted(index.keys()):
            postings = index[token]
            # Convert dict postings to compact [doc_id, tf, 0/1] triples
            compact = [[p["doc_id"], p["tf"], 1 if p.get("important") else 0] for p in postings]
            line = json.dumps(compact, separators=(",", ":")).encode("utf-8") + b"\n"
            offset = f.tell()
            f.write(line)
            offsets[token] = [offset, len(line)]

    with open("index/offsets.json", "w") as f:
        json.dump(offsets, f)

    # Write compact binary lexicon: sorted tokens as text + parallel binary offset arrays
    # Allows O(log n) lookup with ~70 MB RAM instead of loading the full offsets.json dict
    print("Writing tokens.txt and offsets_compact.bin...")
    sorted_tokens = sorted(offsets.keys())
    offset_vals = array.array("Q", (offsets[t][0] for t in sorted_tokens))  # uint64 byte offsets
    length_vals = array.array("I", (offsets[t][1] for t in sorted_tokens))  # uint32 byte lengths
    with open("index/tokens.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted_tokens))
    n = len(sorted_tokens)
    with open("index/offsets_compact.bin", "wb") as f:
        f.write(struct.pack("I", n))   # 4-byte count header
        offset_vals.tofile(f)          # n × 8 bytes
        length_vals.tofile(f)          # n × 4 bytes

    # Report on-disk sizes for each index file
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
