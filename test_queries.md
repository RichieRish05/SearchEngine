# M3 Test Queries

## Queries That Perform Well

1. cristina lopes
2. machine learning
3. ACM
4. master of software engineering
5. information retrieval
6. computer science
7. artificial intelligence
8. software engineering
9. data structure
10. algorithms

## Queries That Initially Performed Poorly

11. how to apply
12. python programming
13. the
14. campus map directions
15. deep reinforcement learning research
16. internship opportunities summer
17. ICS 33
18. professor
19. web crawler
20. graduation requirements checklist

## What Was Done to Improve Poor Queries

### Problem 1: Long documents dominated rankings
Queries like "data structure", "machine learning", "ICS 33", and "web crawler" returned irrelevant bulk text files (maven-contents.txt, DVD.txt, book1) because raw tf-idf scoring favored documents with high raw term frequency regardless of length.

**Fix:** Replaced raw tf with log-frequency weighting using the formula `1 + log10(tf)`. This dampens the effect of high term counts so that a document with 100 occurrences scores only ~3x higher than a document with 1, rather than 100x higher.

### Problem 2: Stopwords polluted results
Queries like "how to apply" and "the" were dominated by stopword matches, returning random large documents where common words appeared most often.

**Fix:** Added stopword filtering at query time. Common English stopwords are removed before searching, so "how to apply" effectively searches for "apply". If all query terms are stopwords (e.g., "the"), the filter is bypassed to still return results.

### Problem 3: Memory and load time
The original search engine loaded the entire 618 MB JSON index into memory (~3 GB in Python dicts), taking 27 seconds to start.

**Fix:** Built a disk-based index format. Only a lightweight offset map (32 MB) and document metadata are loaded into memory (261 MB total). Posting lists are read from disk on demand via file seek. Load time dropped to 2.4 seconds.
