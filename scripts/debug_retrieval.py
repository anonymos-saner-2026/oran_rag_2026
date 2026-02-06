import argparse, sqlite3, json

def fetch_chunk(db, chunk_id):
    cur = db.cursor()
    cur.execute("SELECT chunk_id, doc_id, clause_id, title, page_start, page_end, text FROM chunks WHERE chunk_id=?", (chunk_id,))
    row = cur.fetchone()
    if not row:
        return None
    keys = ["chunk_id","doc_id","clause_id","title","page_start","page_end","text"]
    return dict(zip(keys,row))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/indexes/docstore.sqlite")
    ap.add_argument("--chunk_id", required=True)
    ap.add_argument("--chars", type=int, default=800)
    args = ap.parse_args()

    db = sqlite3.connect(args.db)
    c = fetch_chunk(db, args.chunk_id)
    if not c:
        print("NOT FOUND")
        return
    print(json.dumps({k:c[k] for k in c if k!="text"}, indent=2, ensure_ascii=False))
    print("\n--- TEXT (head) ---\n")
    print((c["text"] or "")[:args.chars])

if __name__ == "__main__":
    main()
