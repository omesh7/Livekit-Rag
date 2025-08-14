#!/usr/bin/env python3
"""
s3_vector_storage.py
Embed local files into Amazon S3 Vectors using Bedrock Titan Text Embeddings V2,
and query them reliably (no overwrites, proper chunking).
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any

import boto3
from botocore.exceptions import ClientError

from dotenv import load_dotenv

# LlamaIndex imports for reading + chunking
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

# -------- CONFIG --------
THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / "data"

BUCKET_NAME = os.getenv("S3VECTOR_BUCKET", "demo-vector-bucket-portfolio")
HASH_BUCKET = os.getenv("S3VECTOR_HASH_BUCKET", "has-general-pupose-bucket")
INDEX_NAME = os.getenv("S3VECTOR_INDEX", "portfolio-index")
REGION_NAME = os.getenv("AWS_REGION", "us-east-1")
# Titan v2 supports 256, 512 or 1024. Use 512 or 1024 for best accuracy.
INDEX_DIMENSION = int(os.getenv("INDEX_DIMENSION", 512))

# Bedrock model id for Titan Text Embeddings V2
TITAN_EMBED_MODEL = "amazon.titan-embed-text-v2:0"

# chunking config (LlamaIndex)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))  # tokens approx (512 is ok)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))  # overlap tokens

# -------- LOGGING --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- AWS clients --------
s3_client = boto3.client("s3", region_name=REGION_NAME)
s3vectors_client = boto3.client("s3vectors", region_name=REGION_NAME)
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION_NAME)


class S3VectorStorage:
    def __init__(
        self, bucket_name: str, index_name: str, index_dimension: int = INDEX_DIMENSION
    ):
        self.bucket_name = bucket_name
        self.index_name = index_name
        self.index_dimension = index_dimension
        self.s3 = s3_client
        self.s3vectors = s3vectors_client
        self.bedrock = bedrock_client

        # set LlamaIndex chunk settings so SimpleDirectoryReader yields chunks according to these
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP
        # We'll use SentenceSplitter to keep sentence boundaries where possible
        self.splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    # ---------------- Bedrock helpers ----------------
    def _invoke_bedrock_embedding(
        self, text: str, dimensions: int = None, normalize: bool = True
    ) -> List[float]:
        """Call Bedrock Titan Text Embeddings V2 and return a python list of floats.
        We use normalize=True recommended for RAG (unit vectors). See Bedrock docs."""
        dims = dimensions or self.index_dimension
        payload = {
            "inputText": text,
            "dimensions": dims,
            "normalize": normalize,
            "embeddingTypes": ["float"],
        }
        try:
            resp = self.bedrock.invoke_model(
                modelId=TITAN_EMBED_MODEL,
                contentType="application/json",
                body=json.dumps(payload),
            )
            body_bytes = resp["body"].read()
            body_text = (
                body_bytes.decode("utf-8")
                if isinstance(body_bytes, (bytes, bytearray))
                else str(body_bytes)
            )
            parsed = json.loads(body_text)
        except ClientError as e:
            logger.error(f"Bedrock invoke_model error: {e}")
            raise

        # Try likely response fields
        emb = (
            parsed.get("embedding")
            or parsed.get("embeddings")
            or parsed.get("embeddingsByType", {}).get("float")
        )
        if not emb or not isinstance(emb, list):
            raise ValueError(f"No embedding found in Bedrock response: {parsed}")

        embedding = [float(x) for x in emb]
        if len(embedding) != dims:
            raise ValueError(f"Embedding length {len(embedding)} != expected {dims}")
        return embedding

    # ---------------- Bucket & index management ----------------
    def create_vector_bucket(self):
        try:
            self.s3vectors.get_vector_bucket(vectorBucketName=self.bucket_name)
            logger.info(f"Vector bucket '{self.bucket_name}' exists.")
            return
        except self.s3vectors.exceptions.NotFoundException:
            logger.info(f"Vector bucket '{self.bucket_name}' not found. Creating...")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchVectorBucket",):
                logger.info("Vector bucket not found. Creating...")
            else:
                logger.error(f"Error checking vector bucket: {e}")
                raise

        try:
            self.s3vectors.create_vector_bucket(vectorBucketName=self.bucket_name)
            logger.info(f"Created vector bucket '{self.bucket_name}'.")
        except ClientError as e:
            logger.error(f"Failed to create vector bucket: {e}")
            raise

    def create_vector_index(self, dimension: int = None):
        dim = dimension or self.index_dimension
        try:
            info = self.s3vectors.get_index(
                vectorBucketName=self.bucket_name, indexName=self.index_name
            )
            resp_dim = info.get("dimension") or info.get("indexDimension")
            if resp_dim:
                self.index_dimension = int(resp_dim)
            logger.info(
                f"Index '{self.index_name}' exists with dimension {self.index_dimension}."
            )
            return
        except self.s3vectors.exceptions.NotFoundException:
            logger.info(
                f"Index '{self.index_name}' not found. Creating with dimension {dim}..."
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in ("NoSuchIndex",):
                logger.error(f"Error checking index: {e}")
                raise

        try:
            self.s3vectors.create_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                dataType="float32",
                dimension=dim,
                distanceMetric="cosine",
                metadataConfiguration={
                    "nonFilterableMetadataKeys": [
                        "content",
                        "source",
                        "doc_id",
                        "chunk_idx",
                    ]
                },
            )
            self.index_dimension = dim
            logger.info(f"Created index '{self.index_name}' (dim {dim}).")
        except ClientError as e:
            logger.error(f"Failed to create index: {e}")
            raise

    # ---------------- Ingest (chunk + embed + store) ----------------
    def _split_document_text(self, text: str) -> List[str]:
        """Use LlamaIndex sentence splitter to split to chunks (keeps sentences)."""
        # SentenceSplitter has method split_text which expects a string and returns list[str]
        chunks = self.splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]

    def embed_documents(self, force_reembed: bool = False):
        """Main ingestion: chunk documents, embed each chunk, store with unique key."""
        # load raw documents (SimpleDirectoryReader returns Document objects)
        docs = SimpleDirectoryReader(DATA_DIR).load_data()
        if not docs:
            raise ValueError("No files found in data directory.")

        self.create_vector_bucket()
        self.create_vector_index(self.index_dimension)

        vectors_to_put = []
        total_chunks = 0

        for doc_idx, doc in enumerate(docs):
            filename = doc.metadata.get("file_path") or f"doc_{doc_idx}"
            filename = Path(filename).name
            # If SimpleDirectoryReader already splits into many small docs, 'doc.text' may already be partial.
            # We'll still split (idempotent) to ensure consistent chunking.
            chunks = self._split_document_text(doc.text)
            if not chunks:
                continue
            for chunk_idx, chunk_text in enumerate(chunks):
                total_chunks += 1
                # unique key per chunk: filename__chunk_<n>
                key = f"{filename}__chunk_{chunk_idx}"

                # embed via Bedrock Titan v2 (normalized)
                emb = self._invoke_bedrock_embedding(
                    chunk_text, dimensions=self.index_dimension, normalize=True
                )

                # vector payload per S3 Vectors API
                v = {
                    "key": key,
                    "data": {"float32": emb},
                    "metadata": {
                        "content": chunk_text[
                            :5000
                        ],  # keep chunk text as metadata (but not too large)
                        "doc_id": filename,
                        "source": filename,
                        "chunk_idx": chunk_idx,
                    },
                }
                vectors_to_put.append(v)

        if not vectors_to_put:
            logger.warning("No vectors prepared for upload.")
            return

        # Put vectors in reasonable batch size
        BATCH = 100  # safe batch size
        for i in range(0, len(vectors_to_put), BATCH):
            batch = vectors_to_put[i : i + BATCH]
            try:
                self.s3vectors.put_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=self.index_name,
                    vectors=batch,
                )
                logger.info(
                    f"Uploaded batch {i}-{i+len(batch)} ({len(batch)} vectors)."
                )
            except ClientError as e:
                logger.error(f"put_vectors failed at batch starting {i}: {e}")
                raise

        logger.info(
            f"Finished embedding and storing {len(vectors_to_put)} chunks (from {len(docs)} files)."
        )
        # save data hash for quick checks (optional)
        self._save_embedding_hash()

    # ---------------- Query ----------------
    def query_vectors(
        self, query: str, top_k: int = 5, max_chunks_per_doc: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Query S3 Vectors and return up to `top_k` chunks.
        Allows up to `max_chunks_per_doc` chunks from a single document/source.
        """
        # 1) embed the query using Bedrock (same model & normalization as during ingest)
        q_emb = self._invoke_bedrock_embedding(
            query, dimensions=self.index_dimension, normalize=True
        )

        # 2) request more results from S3 so we can apply per-doc limits
        request_k = max(top_k * max_chunks_per_doc * 2, top_k + 5)
        try:
            resp = self.s3vectors.query_vectors(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                queryVector={"float32": q_emb},
                topK=request_k,
                returnMetadata=True,
                returnDistance=True,
            )
        except ClientError as e:
            logger.error(f"query_vectors call failed: {e}")
            return []

        raw = resp.get("vectors", []) or []

        # 3) sort by distance (ascending). For S3 Vectors distance: lower == more similar.
        raw_sorted = sorted(raw, key=lambda x: float(x.get("distance", 0.0)))

        results = []
        per_doc_count = {}

        for entry in raw_sorted:
            meta = entry.get("metadata") or {}
            doc_id = meta.get("doc_id") or meta.get("source") or entry.get("key")

            # allow up to max_chunks_per_doc per doc_id
            per_doc_count.setdefault(doc_id, 0)
            if per_doc_count[doc_id] >= max_chunks_per_doc:
                continue

            results.append(
                {
                    "key": entry.get("key"),
                    "doc_id": doc_id,
                    "chunk_idx": meta.get("chunk_idx"),
                    "content": meta.get("content", "")[:4000],
                    "distance": float(entry.get("distance", 0.0)),
                }
            )
            per_doc_count[doc_id] += 1

            if len(results) >= top_k:
                break

        logger.info(
            f"Query returned {len(raw)} raw vectors, filtered to {len(results)} results (top_k={top_k}, max_per_doc={max_chunks_per_doc})."
        )
        for r in results:
            logger.info(
                f"hit: key={r['key']} doc={r['doc_id']} chunk={r['chunk_idx']} dist={r['distance']}"
            )

        return results

    # ---------------- Utilities ----------------
    def list_vectors_debug(self, max_results: int = 20) -> List[Dict[str, Any]]:
        try:
            resp = self.s3vectors.list_vectors(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                maxResults=max_results,
            )
            return resp.get("vectors", [])
        except ClientError as e:
            logger.warning(f"list_vectors failed: {e}")
            return []

    def _get_data_hash(self) -> str:
        hasher = hashlib.sha256()
        if not DATA_DIR.exists():
            return ""
        for root, _, files in os.walk(DATA_DIR):
            for fname in sorted(files):
                path = Path(root) / fname
                try:
                    with open(path, "rb") as f:
                        while chunk := f.read(8192):
                            hasher.update(chunk)
                except IOError:
                    logger.warning(f"Could not read file: {path}")
        return hasher.hexdigest()

    def _save_embedding_hash(self):
        h = self._get_data_hash()
        if not h:
            return
        try:
            self.s3.put_object(
                Bucket=HASH_BUCKET,
                Key="metadata/data_hash.txt",
                Body=h,
                ContentType="text/plain",
            )
        except ClientError as e:
            logger.warning(f"Could not save data hash: {e}")

    def _get_stored_hash(self) -> str:
        try:
            resp = self.s3.get_object(
                Bucket=self.bucket_name, Key="metadata/data_hash.txt"
            )
            return resp["Body"].read().decode("utf-8").strip()
        except ClientError:
            return ""

    def _index_exists(self) -> bool:
        try:
            self.s3vectors.get_index(
                vectorBucketName=self.bucket_name, indexName=self.index_name
            )
            return True
        except ClientError:
            return False


# ---------------- CLI helpers ----------------
s3_vector_storage = S3VectorStorage(BUCKET_NAME, INDEX_NAME, INDEX_DIMENSION)


def embed_data_to_s3(force_reembed: bool = False):
    s3_vector_storage.embed_documents(force_reembed)


def query_s3_vectors(query: str, top_k: int = 3, max_chunks_per_doc: int = 2) -> str:
    hits = s3_vector_storage.query_vectors(query, top_k=top_k, max_chunks_per_doc=max_chunks_per_doc)
    if not hits:
        return "No relevant information found."

    parts = []
    for h in hits:
        parts.append(f"[{h['doc_id']}] (chunk {h.get('chunk_idx')}) {h['content']}\n(distance: {h['distance']:.4f})")
    return "\n\n".join(parts)



def main():
    try:
        logger.info("Embedding data (force reembed)...")
        embed_data_to_s3(force_reembed=True)

        q = "What is the send project Omesh Worked On?"
        logger.info(f"Running test query: {q}")
        print(query_s3_vectors(q, top_k=3))

        sample = s3_vector_storage.list_vectors_debug(10)
        logger.info(f"Sample stored keys: {[v.get('key') for v in sample]}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
