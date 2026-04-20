import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

KNOWLEDGE_BASE_DIR = "knowledge_base"
KB_FILES = [
    "resume.txt", "work_experience.txt", "projects.txt",
    "skills.txt", "certifications.txt", "education.txt", "career_narrative.txt"
]


def get_embedding(text: str) -> list:
    """Get embedding vector for a single text string."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embeddings = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[text],
        parameters={"input_type": "query"}
    )
    return embeddings[0]["values"]


def embed_resume_text(resume_text: str, namespace: str = "resume"):
    """
    Called from app.py when user uploads a PDF resume.
    Chunks the text and uploads to Pinecone under the given namespace.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(resume_text)
    print(f"Created {len(chunks)} chunks from uploaded resume")

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch_texts = chunks[i:i + batch_size]

        embeddings = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=batch_texts,
            parameters={"input_type": "passage"}
        )

        vectors = []
        for j, (emb, text) in enumerate(zip(embeddings, batch_texts)):
            vectors.append({
                "id": f"resume_chunk_{i+j}",
                "values": emb["values"],
                "metadata": {"text": text, "source": "uploaded_resume"}
            })

        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Uploaded batch {i // batch_size + 1}")

    print(f"Resume embedded into Pinecone namespace: '{namespace}'")


def load_documents():
    documents = []
    for filename in KB_FILES:
        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        if os.path.exists(filepath):
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            documents.extend(docs)
            print(f"Loaded: {filename}")
        else:
            print(f"Not found: {filename}")
    return documents


def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def upload_to_pinecone(chunks):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    print(f"Generating embeddings and uploading {len(chunks)} chunks...")

    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]

        embeddings = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=batch_texts,
            parameters={"input_type": "passage"}
        )

        vectors = []
        for j, (emb, meta, text) in enumerate(zip(embeddings, batch_meta, batch_texts)):
            vectors.append({
                "id": f"chunk_{i+j}",
                "values": emb["values"],
                "metadata": {**meta, "text": text}
            })

        index.upsert(vectors=vectors)
        print(f"Uploaded batch {i // batch_size + 1}")


def main():
    print("Starting embedding process...")
    documents = load_documents()
    if not documents:
        print("No documents found!")
        return
    chunks = chunk_documents(documents)
    upload_to_pinecone(chunks)
    print("Knowledge base successfully embedded into Pinecone.")


if __name__ == "__main__":
    main()
