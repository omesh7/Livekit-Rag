import os
import hashlib
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

# Directories
THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / "data"
STORAGE_DIR = THIS_DIR / "query-engine-storage"
HASH_FILE = STORAGE_DIR / "data_hash.txt"


def get_data_hash(directory: Path) -> str:
    """Computes a SHA256 hash of the contents of all files in a directory."""
    hasher = hashlib.sha256()
    if not directory.exists():
        return ""

    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            filepath = Path(root) / filename
            try:
                with open(filepath, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
            except IOError:
                logging.warning(f"Could not read file: {filepath}")
    return hasher.hexdigest()


def needs_reembedding() -> bool:
    """Check if data needs to be re-embedded."""
    if not (STORAGE_DIR / "docstore.json").exists():
        return True

    current_hash = get_data_hash(DATA_DIR)
    stored_hash = ""

    if HASH_FILE.exists():
        with open(HASH_FILE, "r") as f:
            stored_hash = f.read().strip()

    return current_hash != stored_hash


def embed_data():
    """Embed the data and store the index."""
    logging.info("Embedding data...")

    # Ensure storage directory exists
    STORAGE_DIR.mkdir(exist_ok=True)

    # Configure models
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = GoogleGenAI(model="gemini-2.0-flash")

    # Load and embed documents
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    if not documents:
        raise ValueError("No documents found in the data directory")

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=STORAGE_DIR)

    # Save hash
    current_hash = get_data_hash(DATA_DIR)
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)

    logging.info(f"Embedding complete. Hash: {current_hash}")


def main():
    """Main function to check for data changes and re-embed if necessary."""
    if needs_reembedding():
        embed_data()
    else:
        logging.info("Data unchanged. Skipping embedding.")


if __name__ == "__main__":
    main()
