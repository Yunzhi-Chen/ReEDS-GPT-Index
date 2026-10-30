from __future__ import annotations

"""
ReEDS FAISS Index Builder (CI / HPC)
====================================

Builds a FAISS vector index for the ReEDS repository so an AI agent can
quickly retrieve relevant code and documentation snippets.

In CI, this script is called by the update-index workflow with Gemini
embeddings. On HPC you can still use HuggingFace or Ollama models.

Embedding providers are lazily imported so CI only needs the packages for
the chosen provider (Gemini by default).

Environment variables:
    REEDS_REPO             Path to repo root (default: cwd)
    REEDS_INDEX_DIR        Where to write index.faiss / index.pkl
    REEDS_EMBED_MODEL      gemini:gemini-embedding-001 | hf:BAAI/bge-base-en-v1.5 | ollama:nomic-embed-text
    REEDS_INCREMENTAL      1 = replace changed files in existing index (default 1)
    REEDS_SKIP_UNCHANGED   1 = skip files whose content hash matches manifest  (default 1)
    REEDS_EMBED_BATCH      Batch size for embedding calls  (default 16)
    REEDS_EMBED_DEVICE     e.g. "cuda" for HuggingFace GPU
    REEDS_EMBED_DELAY      Seconds between batches to avoid rate limits (default 1.0)
    REEDS_MAX_FILE_BYTES   Skip files larger than this size in bytes (default 5,000,000)
    REEDS_CSV_MAX_LINES    Maximum CSV lines to index (default 4000)
    REEDS_CHUNK_SIZE       Text chunk size (default 1200)
    REEDS_CHUNK_OVERLAP    Text chunk overlap (default 200)
    GOOGLE_API_KEY         Required when using gemini: embeddings
"""

import os
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Iterable, Tuple, Optional, List, Dict, Any

from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ---------------------------------------------------------------------------
# 1) Configuration
# ---------------------------------------------------------------------------

DEFAULT_EMBED_MODEL = os.environ.get(
    "REEDS_EMBED_MODEL",
    "gemini:gemini-embedding-001",
)

EMBED_BATCH_SIZE = int(os.environ.get("REEDS_EMBED_BATCH", "16"))
MAX_FILE_BYTES = int(os.environ.get("REEDS_MAX_FILE_BYTES", str(5_000_000)))

INCREMENTAL = os.environ.get("REEDS_INCREMENTAL", "1") == "1"
SKIP_UNCHANGED = os.environ.get("REEDS_SKIP_UNCHANGED", "1") == "1"

EMBED_DELAY = float(os.environ.get("REEDS_EMBED_DELAY", "1.0"))

INCLUDE_EXT = {
    ".md",
    ".rst",
    ".txt",
    ".tex",
    ".py",
    ".sh",
    ".bat",
    ".ps1",
    ".gms",
    ".inc",
    ".dat",
    ".opt",
    ".op2",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".r",
    ".jl",
    ".sql",
}

INCLUDE_FILENAMES = {
    "Makefile",
    "Dockerfile",
    "Snakefile",
    "README",
    "CHANGELOG",
    "CITATION",
    "LICENSE",
}

EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "node_modules",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "outputs",
    "output",
    "runs",
    "run",
    "lstfiles",
    "g00files",
    "ReEDS_Augur_output",
}

EXCLUDE_PATH_SUBSTR: set[str] = set()

CSV_MAX_LINES = int(os.environ.get("REEDS_CSV_MAX_LINES", "4000"))
CHUNK_SIZE = int(os.environ.get("REEDS_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("REEDS_CHUNK_OVERLAP", "200"))


# ---------------------------------------------------------------------------
# 2) Utility helpers
# ---------------------------------------------------------------------------

def get_git_commit(repo: Path) -> Optional[str]:
    """Return the current git commit hash, if repo is a git repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def sha1_bytes(b: bytes) -> str:
    """Return SHA1 hex digest for bytes."""
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()


def safe_read_text(fp: Path) -> Optional[str]:
    """Read a text file safely, ignoring decoding errors when needed."""
    try:
        raw = fp.read_bytes()

        # Skip likely binary files.
        if b"\x00" in raw[:4096]:
            return None

        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="ignore")

    except Exception:
        return None


def file_signature(fp: Path) -> Optional[Dict[str, Any]]:
    """
    Return a stable file signature based on file content.

    This intentionally uses size + sha1 instead of mtime because GitHub Actions
    runners perform fresh checkouts, where modification times can be unstable.
    """
    try:
        raw = fp.read_bytes()
        return {
            "size": len(raw),
            "sha1": sha1_bytes(raw),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3) File crawling
# ---------------------------------------------------------------------------

def iter_files(repo: Path) -> Iterable[Path]:
    """Yield files that should be included in the ReEDS index."""
    for p in repo.rglob("*"):
        if p.is_dir():
            continue

        ext = p.suffix.lower()
        name = p.name

        if ext not in INCLUDE_EXT and name not in INCLUDE_FILENAMES:
            continue

        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue

        full_str = str(p)
        if any(s in full_str for s in EXCLUDE_PATH_SUBSTR):
            continue

        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
        except Exception:
            continue

        yield p


# ---------------------------------------------------------------------------
# 4) Chunking
# ---------------------------------------------------------------------------

def make_splitter(ext: str) -> RecursiveCharacterTextSplitter:
    """Return a text splitter with simple code/document-aware separators."""
    if ext == ".py":
        return RecursiveCharacterTextSplitter(
            add_start_index=True,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
        )

    if ext == ".gms":
        return RecursiveCharacterTextSplitter(
            add_start_index=True,
            chunk_size=min(CHUNK_SIZE, 900),
            chunk_overlap=min(CHUNK_OVERLAP, 150),
            separators=[";\n", "\n\n", "\n", " ", ""],
        )

    if ext in {".md", ".rst", ".txt"}:
        return RecursiveCharacterTextSplitter(
            add_start_index=True,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        )

    if ext in {
        ".csv",
        ".tsv",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".ini",
        ".cfg",
        ".opt",
        ".op2",
    }:
        return RecursiveCharacterTextSplitter(
            add_start_index=True,
            chunk_size=min(CHUNK_SIZE, 800),
            chunk_overlap=min(CHUNK_OVERLAP, 120),
            separators=["\n", ",", " ", ""],
        )

    return RecursiveCharacterTextSplitter(
        add_start_index=True,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


# ---------------------------------------------------------------------------
# 5) Embeddings
# ---------------------------------------------------------------------------

def make_embeddings() -> Tuple[object, str]:
    """
    Create the embedding provider.

    Supported model strings:
        gemini:gemini-embedding-001
        hf:BAAI/bge-base-en-v1.5
        ollama:nomic-embed-text
    """
    model_name = os.environ.get("REEDS_EMBED_MODEL", DEFAULT_EMBED_MODEL)

    # --- Gemini ---
    if model_name.lower().startswith("gemini:"):
        gemini_model = model_name.split(":", 1)[1]

        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini embeddings.")

        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        print(f"Using Gemini embeddings: {gemini_model}")
        return (
            GoogleGenerativeAIEmbeddings(model=gemini_model),
            f"gemini:{gemini_model}",
        )

    # --- Ollama ---
    if model_name.lower().startswith("ollama:") or model_name.lower() == "ollama":
        ollama_model = (
            model_name.split(":", 1)[1]
            if ":" in model_name
            else os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        )
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        from langchain_ollama import OllamaEmbeddings

        print(f"Using Ollama embeddings: {ollama_model} (base: {base_url})")
        return (
            OllamaEmbeddings(model=ollama_model, base_url=base_url),
            f"ollama:{ollama_model}",
        )

    # --- HuggingFace ---
    if model_name.lower().startswith("hf:"):
        model_name = model_name.split(":", 1)[1]

    device = os.environ.get("REEDS_EMBED_DEVICE")
    model_kwargs = {}
    if device:
        model_kwargs["device"] = device

    encode_kwargs = {"normalize_embeddings": True}

    from langchain_community.embeddings import HuggingFaceEmbeddings

    print(f"Using HuggingFace embeddings: {model_name} | device={device or 'default'}")

    return (
        HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        ),
        f"hf:{model_name}",
    )


# ---------------------------------------------------------------------------
# 6) File to Documents
# ---------------------------------------------------------------------------

def file_to_documents(repo: Path, fp: Path) -> List[Document]:
    """Convert one source file into one LangChain Document."""
    text = safe_read_text(fp)
    if not text:
        return []

    ext = fp.suffix.lower()

    if ext == ".csv":
        lines = text.splitlines()
        if len(lines) > CSV_MAX_LINES:
            text = "\n".join(lines[:CSV_MAX_LINES])

    sig = file_signature(fp) or {"size": -1, "sha1": ""}

    rel = fp.relative_to(repo)
    rel_path = str(rel).replace("\\", "/")

    meta = {
        "path": str(fp),
        "rel_path": rel_path,
        "ext": ext,
        "size": sig["size"],
        "file_sha1": sig["sha1"],
    }

    return [Document(page_content=text, metadata=meta)]


def batched(items: List[Document], batch_size: int) -> Iterable[List[Document]]:
    """Yield batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


# ---------------------------------------------------------------------------
# 6b) Retry wrapper for rate-limited embedding APIs
# ---------------------------------------------------------------------------

def _embed_with_retry(
    vectordb: Optional[FAISS],
    batch: List[Document],
    embeddings,
    max_retries: int = 5,
) -> FAISS:
    """Embed a batch with exponential backoff on rate-limit errors."""
    delay = 30

    for attempt in range(max_retries):
        try:
            if vectordb is None:
                return FAISS.from_documents(
                    batch,
                    embeddings,
                    normalize_L2=True,
                    ids=[c.metadata["chunk_id"] for c in batch],
                )

            vectordb.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
            return vectordb

        except Exception as e:
            err_str = str(e)

            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = delay * (2 ** attempt)
                print(
                    f"\n  Rate limited "
                    f"(attempt {attempt + 1}/{max_retries}). "
                    f"Waiting {wait}s..."
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"Failed after {max_retries} retries due to rate limiting."
    )


def stable_chunk_id(doc: Document) -> str:
    """Independent of CI checkout path, run time and random UUIDs."""
    data = doc.metadata["rel_path"] + "\0" + doc.page_content
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def replace_file(vectordb, chunks, old_ids, embeddings):
    """Replace only the in-memory file revision; reuse exact-content vectors.

    Disk files are saved only after all files succeed. An embedding error raises
    and leaves the downloaded index untouched. Call on a staging copy in CI.
    """
    unique = {}
    for doc in chunks:
        doc.metadata["chunk_id"] = stable_chunk_id(doc)
        doc.metadata["content_sha256"] = hashlib.sha256(doc.page_content.encode()).hexdigest()
        unique.setdefault(doc.metadata["chunk_id"], doc)
    reusable = {}
    if vectordb is not None and old_ids:
        positions = {doc_id: position for position, doc_id in vectordb.index_to_docstore_id.items()}
        wanted = {c.page_content for c in unique.values()}
        for doc_id in old_ids:
            old = vectordb.docstore.search(doc_id)
            if old.page_content in wanted and old.page_content not in reusable:
                reusable[old.page_content] = vectordb.index.reconstruct(positions[doc_id]).copy()
        vectordb.delete(old_ids)
    reused = [c for c in unique.values() if c.page_content in reusable]
    if reused:
        vectordb.add_embeddings(
            [(c.page_content, reusable[c.page_content]) for c in reused],
            metadatas=[c.metadata for c in reused], ids=[c.metadata["chunk_id"] for c in reused],
        )
    new = [c for c in unique.values() if c.page_content not in reusable]
    for batch in batched(new, EMBED_BATCH_SIZE):
        vectordb = _embed_with_retry(vectordb, batch, embeddings)
        time.sleep(EMBED_DELAY)
    return vectordb


# ---------------------------------------------------------------------------
# 7) Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    repo = Path(os.environ.get("REEDS_REPO", ".")).expanduser().resolve()
    out_dir = Path(os.environ.get("REEDS_INDEX_DIR", "./reeds_index")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("ReEDS repo:", repo)
    print("Index dir:", out_dir)
    print("Incremental:", INCREMENTAL, "| Skip unchanged:", SKIP_UNCHANGED)

    if not repo.exists():
        raise FileNotFoundError(f"REEDS_REPO path does not exist: {repo}")

    if not repo.is_dir():
        raise NotADirectoryError(f"REEDS_REPO path is not a directory: {repo}")

    embeddings, embed_name = make_embeddings()

    vectordb: Optional[FAISS] = None
    existing_index = (
        (out_dir / "index.faiss").exists()
        and (out_dir / "index.pkl").exists()
    )
    if INCREMENTAL and ((out_dir / "index.faiss").exists() != (out_dir / "index.pkl").exists()):
        raise RuntimeError("Incomplete previous index pair; preserve it and abort")

    # Load previous manifest for skip-unchanged comparison.
    prev_file_sigs: Dict[str, Dict[str, Any]] = {}
    prev = {}
    prev_manifest_path = out_dir / "manifest.json"

    if prev_manifest_path.exists():
        try:
            prev = json.loads(prev_manifest_path.read_text(encoding="utf-8"))
            prev_file_sigs = prev.get("file_signatures", {}) or {}
        except Exception as e:
            raise RuntimeError("Previous index manifest is unreadable; preserve it and abort") from e

    # Load previous FAISS index when doing incremental update.
    if INCREMENTAL and existing_index:
        if not prev or prev.get("embed_model") != embed_name:
            raise RuntimeError("Incremental index requires a matching model manifest")
        try:
            vectordb = FAISS.load_local(
                str(out_dir),
                embeddings,
                allow_dangerous_deserialization=True,
                normalize_L2=True,
            )
            print("Loaded existing FAISS index from:", out_dir)
        except Exception as e:
            raise RuntimeError("Existing index could not load; preserve it and abort") from e

    source_ids = {}
    if vectordb is not None:
        for doc_id in vectordb.index_to_docstore_id.values():
            path = vectordb.docstore.search(doc_id).metadata.get("rel_path")
            source_ids.setdefault(path, []).append(doc_id)

    current_commit = get_git_commit(repo)

    manifest: Dict[str, object] = {
        "repo": str(repo),
        "git_commit": current_commit,
        "created_utc": int(time.time()),
        "embed_model": embed_name,
        "include_ext": sorted(INCLUDE_EXT),
        "include_filenames": sorted(INCLUDE_FILENAMES),
        "exclude_dirs": sorted(EXCLUDE_DIRS),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "incremental": INCREMENTAL,
        "skip_unchanged": SKIP_UNCHANGED,
        "file_signature_method": "size+sha1",
        "index_schema_version": 2,
        "chunk_id_method": "sha256(rel_path + NUL + full_text)",
        "file_signatures": {},
    }

    files = list(iter_files(repo))
    print("Files to consider:", len(files))

    if not files:
        raise RuntimeError("No files matched filters; refusing to publish an empty or stale index")

    total_chunks = 0
    indexed_files = 0
    skipped_files = 0
    new_file_sigs: Dict[str, Dict[str, Any]] = {}

    for fp in tqdm(files, desc="Indexing files", unit="file"):
        rel = str(fp.relative_to(repo)).replace("\\", "/")
        sig = file_signature(fp)

        if sig:
            new_file_sigs[rel] = sig

        if SKIP_UNCHANGED and INCREMENTAL and vectordb is not None and sig:
            prev_sig = prev_file_sigs.get(rel)

            if (
                prev_sig
                and prev_sig.get("size") == sig.get("size")
                and prev_sig.get("sha1") == sig.get("sha1")
            ):
                skipped_files += 1
                continue

        docs = file_to_documents(repo, fp)
        if not docs:
            vectordb = replace_file(vectordb, [], source_ids.get(rel, []), embeddings)
            continue

        splitter = make_splitter(fp.suffix.lower())
        chunks = splitter.split_documents(docs)

        if not chunks:
            vectordb = replace_file(vectordb, [], source_ids.get(rel, []), embeddings)
            continue

        for c in chunks:
            content_bytes = c.page_content.encode("utf-8", errors="ignore")
            c.metadata["chunk_sha1"] = sha1_bytes(content_bytes[:20000])

        vectordb = replace_file(vectordb, chunks, source_ids.get(rel, []), embeddings)

        total_chunks += len(chunks)
        indexed_files += 1

    if vectordb is None:
        print("No chunks were indexed.")
        return

    # Deleted/excluded official files must not survive as current search results.
    removed = [doc_id for path, ids in source_ids.items() if path not in new_file_sigs for doc_id in ids]
    if removed:
        vectordb.delete(removed)

    vectordb.save_local(str(out_dir))

    manifest["file_signatures"] = new_file_sigs

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved FAISS index to: {out_dir}")
    print(f"  embed_model: {embed_name}")
    print(f"  git_commit: {current_commit}")
    print(f"  files_indexed: {indexed_files}")

    if INCREMENTAL and existing_index and SKIP_UNCHANGED:
        print(f"  files_skipped: {skipped_files}")

    print(f"  chunks_added: {total_chunks}")


if __name__ == "__main__":
    main()
