import json

with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Total chunks:", len(chunks))
for c in chunks[:10]:
    print("\n---")
    print("SOURCE:", c["source_type"])
    print("TITLE:", c["title"])
    print("WORDS:", c["word_count"])
    print("TAGS:", c["topic_tags"])
    print(c["text"][:300])