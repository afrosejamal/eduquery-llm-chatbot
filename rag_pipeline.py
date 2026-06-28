import os
import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

DATA_PATH = "data/raw"
CHROMA_PATH = "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="eduquery")


def extract_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print("⚠ Skipping broken PDF:", pdf_path)
        return ""


def chunk_text(text, size=500):
    words = text.split()
    if len(words) == 0:
        return []
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


def ingest_pdfs():
    print("📥 Ingesting PDFs...\n")

    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                print("➡", path)

                text = extract_text(path)
                if not text:
                    print("   ⚠ No readable text — skipped\n")
                    continue

                chunks = chunk_text(text)
                if not chunks:
                    print("   ⚠ No chunks — skipped\n")
                    continue

                embeddings = model.encode(chunks).tolist()

                ids = [f"{file}_{i}" for i in range(len(chunks))]
                metas = [{"source": file} for _ in chunks]

                collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metas
                )

                print(f"   ✅ Added {len(chunks)} chunks\n")

    print("🎉 INGESTION COMPLETE")
    print("📦 Total chunks in DB =", collection.count())


if __name__ == "__main__":
    ingest_pdfs()
