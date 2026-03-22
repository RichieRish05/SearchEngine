"""Compute PageRank from the link graph saved by indexer.py.

Reads:  index/links.json   (source_url → [target_url, ...])
        index/doc_map.json  (doc_id → url)

Writes: index/pagerank.json (doc_id_str → normalized PR score in [0, 1])

Run once after indexer.py, before optimize_index.py or search.py.
"""

import json
from collections import defaultdict

DAMPING = 0.85    # probability of following a link (standard PageRank value)
ITERATIONS = 50   # number of power-iteration steps


def main():
    print("Loading link graph and doc map...")
    with open("index/links.json") as f:
        raw_links = json.load(f)
    with open("index/doc_map.json") as f:
        doc_map = json.load(f)

    # Build URL → doc_id mapping (only pages that made it into the index)
    url_to_id = {v: int(k) for k, v in doc_map.items()}
    indexed_ids = set(url_to_id.values())
    N = len(indexed_ids)
    print(f"  {N:,} indexed documents, {len(raw_links):,} source URLs in link graph")

    # Build adjacency restricted to indexed pages; ignore self-links
    out_links = defaultdict(set)  # src_id → {tgt_id, ...}
    in_links = defaultdict(set)   # tgt_id → {src_id, ...}
    for src_url, targets in raw_links.items():
        if src_url not in url_to_id:
            continue
        src = url_to_id[src_url]
        for tgt_url in targets:
            if tgt_url in url_to_id:
                tgt = url_to_id[tgt_url]
                if tgt != src:
                    out_links[src].add(tgt)
                    in_links[tgt].add(src)

    print(f"  {len(out_links):,} pages with outgoing links")

    # PageRank power iteration: PR(p) = (1-d)/N + d * sum(PR(q)/|out(q)|)
    pr = {doc_id: 1.0 / N for doc_id in indexed_ids}  # uniform initialization
    for iteration in range(ITERATIONS):
        new_pr = {}
        for doc_id in indexed_ids:
            # Sum contributions from all pages that link to this page
            rank_sum = sum(
                pr[j] / len(out_links[j])
                for j in in_links[doc_id]
                if out_links[j]
            )
            new_pr[doc_id] = (1 - DAMPING) / N + DAMPING * rank_sum
        pr = new_pr
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}/{ITERATIONS}")

    print(sum(pr.values()))  # should be ~1.0 (sanity check)

    # Normalize to [0, 1] by dividing by max PR — used as a multiplicative boost in search
    max_pr = max(pr.values())
    min_pr = min(pr.values())
    pr_norm = {str(k): v / max_pr for k, v in pr.items()}

    with open("index/pagerank.json", "w") as f:
        json.dump(pr_norm, f)

    print(f"PageRank saved for {len(pr_norm):,} documents.")
    print(f"  Max PR: {max_pr:.6f}  Min PR: {min_pr:.6f}")


if __name__ == "__main__":
    main()
