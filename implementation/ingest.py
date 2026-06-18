import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent.parent 
DB_NAME = str(BASE_DIR / "vector_db")
KNOWLEDGE_BASE = str(BASE_DIR / "knowledge-base")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    """Load .md files from every sub-folder inside knowledge-base/."""
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
            folder, glob="**/*.md", loader_cls=TextLoader,loader_kwargs={"encoding": "utf-8"},
            show_progress=True, silent_errors=True,
        )

        loaded = loader.load()
        if not loaded:
            print(f"No .md files found in: {folder}")
            continue

        for doc in loaded:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

        print(f"Loaded {len(loaded)} file(s) from '{doc_type}'")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def create_chunks(documents):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created : {len(chunks)}")
    return chunks


def build_vectorstore(chunks):
    """Create or overwrite the Chroma vectorstore."""
    if not chunks:
        print("No chunks to embed. Check that your .md files exist and are not empty.")
        return None

    if os.path.exists(DB_NAME):
        print("Deleting existing vectorstore …")
        Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings,
        ).delete_collection()

    print("Building vectorstore (this may take a moment) …")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )

    count = vectorstore._collection.count()
    print(f"Vectorstore ready: {count:,} vectors stored")
    return vectorstore


if __name__ == "__main__":
    docs   = fetch_documents()
    chunks = create_chunks(docs)
    build_vectorstore(chunks)
    print("\n🚀 Ingestion complete")