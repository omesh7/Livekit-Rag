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
    """
    Computes a SHA256 hash of the contents of all files in a directory.
    """
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

def main():
    """
    Main function to check for data changes and re-embed if necessary.
    """
    logging.info("Starting pre-warmup embedding process...")

    # Ensure storage directory exists
    STORAGE_DIR.mkdir(exist_ok=True)

    # Calculate current hash of the data
    current_hash = get_data_hash(DATA_DIR)
    logging.info(f"Current data hash: {current_hash}")

    # Read the stored hash
    stored_hash = ""
    if HASH_FILE.exists():
        with open(HASH_FILE, "r") as f:
            stored_hash = f.read().strip()
        logging.info(f"Stored data hash: {stored_hash}")

    # Compare hashes
    if current_hash == stored_hash and (STORAGE_DIR / "docstore.json").exists():
        logging.info("Data has not changed. Loading from existing storage.")
        # Optional: Load the index to confirm it's valid
        try:
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
            load_index_from_storage(storage_context)
            logging.info("Successfully loaded index from storage. Pre-warmup complete.")
        except Exception as e:
            logging.error(f"Failed to load index from storage, re-embedding might be needed. Error: {e}")
        return

    logging.info("Data has changed or storage is new. Re-embedding...")

    # Configure models
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = GoogleGenAI(model="gemini-2.0-flash")

    # Load documents
    try:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        if not documents:
            logging.error("No documents found in the data directory. Aborting.")
            return
        logging.info(f"Loaded {len(documents)} document(s).")
    except Exception as e:
        logging.error(f"Failed to load documents: {e}")
        return

    # Create and persist the index
    try:
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        logging.info("Successfully created and stored the new index.")
    except Exception as e:
        logging.error(f"Failed to create or persist index: {e}")
        return

    # Save the new hash
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)
    logging.info(f"Updated data hash to: {current_hash}")
    logging.info("Pre-warmup and embedding process complete.")

if __name__ == "__main__":
    main()
