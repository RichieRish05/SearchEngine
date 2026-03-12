# How We Optimized the Index

## The Problem

Our original search engine stored the entire inverted index as a single JSON file (`index.json`, 618 MB). When the search engine started, it loaded this whole file into a Python dictionary using `json.load()`. This caused two major problems:

1. **Huge memory footprint**: The JSON file expanded to ~3 GB in memory (4.84x the file size), because Python dicts, strings, and lists have significant overhead per object.
2. **Slow startup**: Parsing 618 MB of JSON took ~27 seconds before a single query could run.

The M3 requirement states that the memory footprint must be **smaller than the index size**, so loading everything into RAM was not an option.

## The Solution: Disk-Based Index with Offset Lookup

We split the single monolithic JSON file into two files that work together:

### File 1: `postings.bin` (150 MB) — The Postings Store

This file contains every posting list, written one per line in a compact format. Each line is a JSON array of `[doc_id, tf, important_flag]` triples for a single token.

For example, the line for the token "algorithm" might look like:

```
[[204,3,1],[1087,1,0],[2301,5,1],[4892,2,0]]
```

This means:
- Document 204 has tf=3 and the term appears in an important tag
- Document 1087 has tf=1 and is not in an important tag
- ...and so on

This is more compact than the original format which used named keys (`{"doc_id": 204, "tf": 3, "important": true}`).

### File 2: `offsets.json` (32 MB) — The Lookup Table

This is a small dictionary that maps each token to its **byte position** and **byte length** inside `postings.bin`:

```json
{
  "algorithm": [0, 847],
  "comput": [847, 12304],
  "data": [13151, 9821],
  ...
}
```

So `"algorithm": [0, 847]` means: to read the posting list for "algorithm", seek to byte 0 in `postings.bin` and read 847 bytes.

## How Search Works with This Format

### At Startup (once)
1. Load `offsets.json` into memory (32 MB — the lookup table)
2. Load `doc_map.json` into memory (4.6 MB — doc_id to URL mapping)
3. Open a file handle to `postings.bin` (no data read yet)

Total memory: ~256 MB (well below the 618 MB index size).

### At Query Time (per query)
1. Tokenize and stem the query (e.g., "machine learning" becomes `["machin", "learn"]`)
2. For each query term, look up its offset in the offsets dictionary
3. **Seek** to that position in `postings.bin` and read just that one posting list
4. Intersect the posting lists (AND semantics) and score the results

This means we only read the posting lists we actually need. A typical query reads 2-3 posting lists from disk, not the entire 150 MB file.

### Visual Diagram

```
                    STARTUP (in memory)
                    +------------------+
                    | offsets.json     |  token -> [byte_offset, length]
                    | (32 MB)         |
                    +------------------+
                    | doc_map.json    |  doc_id -> URL
                    | (4.6 MB)        |
                    +------------------+

Query: "machine learning"
  |
  v
  stem -> ["machin", "learn"]
  |
  v
  offsets["machin"] = [52840, 3201]    <- lookup in memory (instant)
  offsets["learn"]  = [41200, 8734]    <- lookup in memory (instant)
  |
  v
                    ON DISK (not in memory)
                    +------------------+
  seek to 52840 -> | postings.bin     |  read 3201 bytes -> posting list for "machin"
  seek to 41200 -> | (150 MB)         |  read 8734 bytes -> posting list for "learn"
                    +------------------+
  |
  v
  Intersect, score, return top 5 results
```

## How We Built It (`optimize_index.py`)

The conversion script reads the original `index.json` once and writes the two new files:

1. Sort all tokens alphabetically
2. For each token, convert its posting list to the compact `[doc_id, tf, flag]` format
3. Write that compact JSON line to `postings.bin`, recording the byte offset before writing
4. After all tokens are written, save all the offsets to `offsets.json`

## Performance Comparison

| Metric | Before (index.json) | After (postings.bin + offsets) |
|--------|---------------------|-------------------------------|
| Memory used | 2,991 MB | 256 MB |
| Load time | 27 seconds | 2.4 seconds |
| Avg query time | 16 ms | 39 ms |
| Mem/index ratio | 4.84x | 1.40x |

Query time increased slightly because we now read from disk per query instead of memory. But 39 ms is still fast, and the tradeoff is worth it: we use **11x less memory** and start **11x faster**.

## Why This Works

The key insight is that at query time, we only need the posting lists for the 1-5 terms in the query, not all 1,093,030 terms. By keeping only the small lookup table in memory and reading individual posting lists on demand, we avoid loading the 99.99% of the index that any given query doesn't need.