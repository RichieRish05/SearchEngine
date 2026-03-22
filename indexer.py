"""Two-pass indexer:
  Pass 1 — collect anchor text and link graph from all pages (collect_anchor_text)
  Pass 2 — build the inverted index, dumping partial indexes every 10K docs (build_index)
  Final  — merge all partial indexes into index/index.json (merge_partial_indexes)
"""

import os
import re
import json
import hashlib
import logging
import warnings
from urllib.parse import urljoin, urldefrag
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from nltk.stem import PorterStemmer
from collections import defaultdict

# --- Simhash near-duplicate detection ---
_SIMHASH_BITS = 64
_BAND_SIZE = 16          # divide 64-bit fingerprint into 4 bands of 16 bits
_NUM_BANDS = _SIMHASH_BITS // _BAND_SIZE
_NEAR_DUP_THRESHOLD = 3  # Hamming distance ≤ 3 → treat as near-duplicate
_BAND_MASK = (1 << _BAND_SIZE) - 1


def _simhash(tokens):
    """Compute a 64-bit simhash fingerprint for a token list.
    Each token is hashed; bits are accumulated as +1/-1 votes, then thresholded.
    """
    v = [0] * _SIMHASH_BITS
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16) & ((1 << _SIMHASH_BITS) - 1)
        for i in range(_SIMHASH_BITS):
            v[i] += 1 if (h >> i) & 1 else -1
    fp = 0
    for i in range(_SIMHASH_BITS):
        if v[i] > 0:
            fp |= (1 << i)
    return fp


def _get_bands(fp):
    """Split a 64-bit fingerprint into _NUM_BANDS chunks of _BAND_SIZE bits each."""
    return [(fp >> (i * _BAND_SIZE)) & _BAND_MASK for i in range(_NUM_BANDS)]


def _is_near_duplicate(fp, bands_table):
    """Return True if fp is within Hamming distance _NEAR_DUP_THRESHOLD of any seen fingerprint.
    Uses LSH banding: candidate set = all fingerprints sharing at least one band value.
    """
    candidates = set()
    for i, band_val in enumerate(_get_bands(fp)):
        candidates.update(bands_table[i].get(band_val, []))
    return any(bin(fp ^ c).count("1") <= _NEAR_DUP_THRESHOLD for c in candidates)


def _add_to_bands(fp, bands_table):
    """Register a fingerprint in the LSH bands table."""
    for i, band_val in enumerate(_get_bands(fp)):
        bands_table[i].setdefault(band_val, []).append(fp)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

CORPUS_PATH = "DEV"
stemmer = PorterStemmer()


def tokenize(text):
    """Extract alphanumeric tokens and apply Porter stemming."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    return [stemmer.stem(t.lower()) for t in tokens]


def get_important_tokens(soup):
    """Extract tokens from high-signal HTML tags (title, headings, bold).
    These tokens receive a 2× scoring boost during search.
    """
    important = set()
    for tag in soup.find_all(["title", "h1", "h2", "h3", "b", "strong"]):
        text = tag.get_text()
        tokens = tokenize(text)
        important.update(tokens)
    return important


def compute_tf(tokens):
    """Count raw term frequency for each token in the list."""
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    return tf


def dump_partial(partial_index, dump_num):
    """Write the current in-memory partial index to disk as JSON and reset."""
    os.makedirs("partial_indexes", exist_ok=True)
    path = f"partial_indexes/partial_{dump_num}.json"
    with open(path, "w") as f:
        json.dump(partial_index, f)
    logger.info(f"Dumped partial index #{dump_num} to {path}")
    return path


def load_documents(corpus_path):
    """Walk the corpus directory and yield (doc_id, url, content) for each JSON file."""
    doc_id = 0
    for domain in sorted(os.listdir(corpus_path)):
        domain_path = os.path.join(corpus_path, domain)
        if not os.path.isdir(domain_path):
            continue
        for fname in sorted(os.listdir(domain_path)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(domain_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Skipping {fpath}: {e}")
                continue
            url = data.get("url", "")
            content = data.get("content", "")
            yield doc_id, url, content
            doc_id += 1


def parse_html(content):
    """Parse raw HTML content into a BeautifulSoup object."""
    return BeautifulSoup(content, "lxml")


def collect_anchor_text(corpus_path):
    """Pass 1: collect anchor text and link graph for each target URL.

    Returns anchor_map: target_url → list of stemmed anchor tokens.
    Also writes index/links.json (source_url → [target_urls]) for PageRank.
    """
    anchor_map = defaultdict(list)
    link_graph = defaultdict(set)
    for _, url, content in load_documents(corpus_path):
        soup = parse_html(content)
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            # Skip non-page links
            if not href or href.startswith(("javascript:", "mailto:", "#")):
                continue
            try:
                target, _ = urldefrag(urljoin(url, href))
            except ValueError:
                continue
            link_graph[url].add(target)
            text = a.get_text().strip()
            if text:
                anchor_map[target].extend(tokenize(text))  # anchor text credited to target page
    os.makedirs("index", exist_ok=True)
    with open("index/links.json", "w") as f:
        json.dump({k: list(v) for k, v in link_graph.items()}, f)
    logger.info(f"Link graph: {len(link_graph):,} source URLs saved to index/links.json")
    logger.info(f"Anchor text collected for {len(anchor_map):,} target URLs")
    return anchor_map


def build_index(anchor_map=None):
    """Pass 2: build the inverted index with duplicate detection.

    - Exact duplicates: skipped via MD5 hash of page text
    - Near-duplicates: skipped via SimHash + LSH banding
    - Boilerplate: nav/footer/aside removed before tokenizing
    - Partial index dumped to disk every DUMP_EVERY documents to limit RAM usage
    """
    DUMP_EVERY = 10000
    partial_index = defaultdict(list)
    partial_files = []
    doc_map = {}
    dump_num = 0
    seen_hashes = set()                            # exact duplicate fingerprints
    bands_table = [{} for _ in range(_NUM_BANDS)]  # LSH tables for SimHash
    duplicates_skipped = 0

    for doc_id, url, content in load_documents(CORPUS_PATH):
        soup = parse_html(content)

        # Strip boilerplate: nav/footer/aside often contain link lists irrelevant to page content
        for tag in soup.find_all(["nav", "footer", "aside"]):
            tag.decompose()
        text = soup.get_text()

        # Skip exact duplicates (identical page text)
        content_hash = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        if content_hash in seen_hashes:
            duplicates_skipped += 1
            continue
        seen_hashes.add(content_hash)

        # Skip near-duplicates (Hamming distance ≤ 3 on 64-bit simhash)
        tokens = tokenize(text)
        fp = _simhash(tokens)
        if _is_near_duplicate(fp, bands_table):
            duplicates_skipped += 1
            continue
        _add_to_bands(fp, bands_table)

        doc_map[doc_id] = url

        # Tokens in title/headings/bold get importance flag for scoring boost
        important = get_important_tokens(soup)

        # Augment with incoming anchor text — credited to this page as important tokens
        if anchor_map:
            anchor_tokens = anchor_map.get(url, [])
            tokens = tokens + anchor_tokens
            important = important | set(anchor_tokens)

        tf = compute_tf(tokens)

        # Unigram postings: {doc_id, tf, important}
        for token, freq in tf.items():
            posting = {"doc_id": doc_id, "tf": freq, "important": token in important}
            partial_index[token].append(posting)

        # Bigram postings: stored as "stem1 stem2" keys — used for phrase bonus in search
        bigram_tf = defaultdict(int)
        for i in range(len(tokens) - 1):
            bigram_tf[f"{tokens[i]} {tokens[i+1]}"] += 1
        for bigram, freq in bigram_tf.items():
            partial_index[bigram].append({"doc_id": doc_id, "tf": freq, "important": False})

        # Flush partial index to disk to keep memory bounded
        if doc_id > 0 and doc_id % DUMP_EVERY == 0:
            path = dump_partial(partial_index, dump_num)
            partial_files.append(path)
            partial_index = defaultdict(list)
            dump_num += 1

        if doc_id % 100 == 0 and doc_id > 0:
            logger.info(f"Processed {doc_id} documents so far")

    # Dump any remaining postings
    path = dump_partial(partial_index, dump_num)
    partial_files.append(path)

    # Save doc_id → URL mapping for retrieval
    os.makedirs("index", exist_ok=True)
    with open("index/doc_map.json", "w") as f:
        json.dump(doc_map, f)
    logger.info(f"Saved doc map with {len(doc_map)} entries")
    logger.info(f"Duplicate pages skipped: {duplicates_skipped}")

    return partial_files


def merge_partial_indexes(partial_files):
    """Merge all partial index JSON files into a single index/index.json."""
    merged_index = defaultdict(list)

    for path in partial_files:
        with open(path, "r") as f:
            partial_index = json.load(f)
        for token, postings in partial_index.items():
            merged_index[token].extend(postings)

    os.makedirs("index", exist_ok=True)
    with open("index/index.json", "w") as f:
        json.dump(merged_index, f)

    return merged_index


if __name__ == "__main__":
    logger.info("Pass 1: collecting anchor text...")
    anchor_map = collect_anchor_text(CORPUS_PATH)
    logger.info("Pass 2: building index...")
    partial_files = build_index(anchor_map)
    merged = merge_partial_indexes(partial_files)
    logger.info(f"Partial index files: {partial_files}")
