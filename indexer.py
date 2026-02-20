import os
import re
import json
import logging
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk.stem import PorterStemmer
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

CORPUS_PATH = "DEV"
stemmer = PorterStemmer()


def tokenize(text):
    """Extract alphanumeric tokens and apply Porter stemming."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    return [stemmer.stem(t.lower()) for t in tokens]


def get_important_tokens(soup):
    important = set()
    for tag in soup.find_all(["title", "h1", "h2", "h3", "b", "strong"]):
        text = tag.get_text()
        tokens = tokenize(text)
        important.update(tokens)
    return important


def compute_tf(tokens):
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    return tf

def dump_partial(partial_index, dump_num):
    os.makedirs("partial_indexes", exist_ok=True)
    path = f"partial_indexes/partial_{dump_num}.json"
    with open(path, "w") as f:
        json.dump(partial_index, f)
    logger.info(f"Dumped partial index #{dump_num} to {path}")
    return path



def load_documents(corpus_path):
    """Walk the corpus and yield (doc_id, url, content) for each document."""
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
    """Parse HTML content and return the BeautifulSoup object."""
    return BeautifulSoup(content, "lxml")


def build_index():
    DUMP_EVERY = 10000
    partial_index = defaultdict(list)
    partial_files = []
    doc_map = {}
    dump_num = 0

    for doc_id, url, content in load_documents(CORPUS_PATH):
        doc_map[doc_id] = url

        soup = parse_html(content)
        tokens = tokenize(soup.get_text())
        important = get_important_tokens(soup)
        tf = compute_tf(tokens)

        for token, freq in tf.items():
            posting = {"doc_id": doc_id, "tf": freq, "important": token in important}
            partial_index[token].append(posting)

        if doc_id > 0 and doc_id % DUMP_EVERY == 0:
            path = dump_partial(partial_index, dump_num)
            partial_files.append(path)
            partial_index = defaultdict(list)
            dump_num += 1

        if doc_id % 100 == 0 and doc_id > 0:
            logger.info(f"Processed {doc_id} documents so far")

    # dump remaining
    path = dump_partial(partial_index, dump_num)
    partial_files.append(path)

    # save doc map
    os.makedirs("index", exist_ok=True)
    with open("index/doc_map.json", "w") as f:
        json.dump(doc_map, f)
    logger.info(f"Saved doc map with {len(doc_map)} entries")

    return partial_files

def merge_partial_indexes(partial_files):
    merged_index = defaultdict(list)

    for path in partial_files:
        with open(path, "r") as f:
            partial_index = json.load(f)
        for token, postings in partial_index.items():
            merged_index[token].extend(postings)

    # save merged index
    os.makedirs("index", exist_ok=True)
    with open("index/index.json", "w") as f:
        json.dump(merged_index, f)
        
    return merged_index


if __name__ == "__main__":
    logger.info("Starting Indexing...")
    partial_files = build_index()
    merged = merge_partial_indexes(partial_files)
    logger.info(f"Partial index files: {partial_files}")
