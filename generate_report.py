import json
import os
from dataclasses import dataclass


@dataclass
class Data:
    indexed_documents: int
    unique_tokens: int
    total_bytes: int

data = Data(0,0,0)

# 1. Number of indexed documents
with open('index/doc_map.json') as f:
    doc_map = json.load(f)
    data.indexed_documents = len(doc_map)



# 2. Number of unique tokens
with open('index/index.json') as f:
    index = json.load(f)
    data.unique_tokens = len(index)

# 3. Total index size on disk
data.total_bytes = os.path.getsize('index/index.json') + os.path.getsize('index/doc_map.json')


print(f'Indexed documents : {data.indexed_documents:,}')
print(f'Unique tokens     : {data.unique_tokens:,}')
print(f'Index size on disk: {data.total_bytes / 1024:,.1f} KB')


with open("report.txt", "w") as f:
    f.write(f'Indexed documents : {data.indexed_documents:,}\n')
    f.write(f'Unique tokens     : {data.unique_tokens:,}\n')
    f.write(f'Index size on disk: {data.total_bytes / 1024:,.1f} KB\n')
