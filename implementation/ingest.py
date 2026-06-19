import os
import glob
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE = str(BASE_DIR / "knowledge-base")

INDEX_NAME = os.getenv("PINECONE_API_KEY", "talentbridge")
print(INDEX_NAME)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)


def fetch_documents():
    documents = []

    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))

    if not folders:
        print(f"No folders found in: {KNOWLEDGE_BASE}")
        return documents

    for folder in folders:
        if not os.path.isdir(folder):
            continue

        doc_type = os.path.basename(folder)

        print(f"Loading folder: {doc_type}")

        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            silent_errors=True,
        )

        loaded = loader.load()

        if not loaded:
            continue

        for doc in loaded:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

        print(f"Loaded {len(loaded)} documents")

    print(f"Total documents loaded: {len(documents)}")
    return documents


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")
    return chunks


def create_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {INDEX_NAME}")

        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,  # text-embedding-3-large
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

    return pc


def build_vectorstore(chunks):
    if not chunks:
        print("No chunks found")
        return

    create_index()

    print("Uploading vectors to Pinecone...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    print("✅ Pinecone ingestion complete")


if __name__ == "__main__":
    docs = fetch_documents()
    chunks = create_chunks(docs)
    build_vectorstore(chunks)