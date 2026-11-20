"""Your own knowledge base: drop in PDFs, Word/Excel/PowerPoint files, text files and notes, then ask
the agent questions and get answers that name the source file (and page).

Search is local (BM25 keyword search, no AI model and no internet), so it costs no quota and also works
offline. Only the few matching passages are shown to the model when you ask a question.

PDF text needs the small pypdf package:  pip install pypdf
Only the extracted text is kept (in knowledge/index.json), not the original file: keep your own copy."""
import io
import json
import math
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

import uploads

KB_DIR = Path(__file__).parent / "knowledge"
INDEX_FILE = KB_DIR / "index.json"
LOCK = threading.RLock()

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_DOCS = 100
MAX_PDF_PAGES = 2000
MAX_CHUNKS_PER_DOC = 3000
MAX_TOTAL_CHUNKS = 40000
CHUNK_CHARS = 900
CHUNK_OVERLAP = 120
MAX_RESULTS = 8
SNIPPET_CHARS = 1200
MAX_QUERY_CHARS = 200
BM25_K1 = 1.4
BM25_B = 0.75

STOP = frozenset(
    "a an the and or of to in on at for from by with as is are was were be been it its this that these "
    "those i you he she we they my your our their me him her them do does did not no but if then than so "
    "can could should would will just about into over under also have has had what which who whom how "
    "when where why".split())


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------
# Storage
# ---------------------------------------------------------------
def _load_docs() -> list:
    if INDEX_FILE.exists():
        try:
            data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("docs"), list):
                return data["docs"]
        except (OSError, ValueError):
            try:
                INDEX_FILE.replace(INDEX_FILE.with_name("index.corrupt.json"))  # keep the damaged file
            except OSError:
                pass
    return []


def _save_docs(docs: list) -> None:
    KB_DIR.mkdir(exist_ok=True)
    tmp = INDEX_FILE.with_name(INDEX_FILE.name + ".tmp")
    tmp.write_text(json.dumps({"docs": docs}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(INDEX_FILE)


def _public(doc: dict) -> dict:
    pages = {c.get("page") for c in doc.get("chunks", []) if c.get("page")}
    return {
        "id": doc["id"],
        "name": doc["name"],
        "kind": doc.get("kind", ""),
        "size": int(doc.get("size", 0)),
        "added": doc.get("added", ""),
        "chunks": len(doc.get("chunks", [])),
        "pages": len(pages) or None,
        "truncated": bool(doc.get("truncated")),
    }


def list_documents() -> list:
    with LOCK:
        docs = _load_docs()
    return sorted((_public(d) for d in docs), key=lambda d: d["added"], reverse=True)


def stats() -> dict:
    with LOCK:
        docs = _load_docs()
    chars = sum(len(c.get("text", "")) for d in docs for c in d.get("chunks", []))
    return {"documents": len(docs), "chunks": sum(len(d.get("chunks", [])) for d in docs), "chars": chars}


def remove_document(doc_id: str) -> bool:
    with LOCK:
        docs = _load_docs()
        remaining = [d for d in docs if d.get("id") != doc_id]
        if len(remaining) == len(docs):
            return False
        _save_docs(remaining)
    return True


# ---------------------------------------------------------------
# Reading files into text
# ---------------------------------------------------------------
def _extract_pdf(data: bytes) -> list:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError("Reading PDF text needs the pypdf package. In Command Prompt run: pip install pypdf")
    pages = []
    try:
        reader = PdfReader(io.BytesIO(data))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("That PDF is password protected.")
        for number, page in enumerate(reader.pages, 1):
            if number > MAX_PDF_PAGES:
                break
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                pages.append((number, text))
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"That PDF could not be read ({str(e)[:100]}).")
    if sum(len(t) for _, t in pages) < 20:
        raise ValueError("No text was found in that PDF. It may be scanned pictures: this knowledge base "
                         "needs a PDF with text you can select.")
    return pages


def _extract(kind: str, ext: str, data: bytes) -> list:
    """[(page number or None, text), ...]"""
    if kind == "pdf":
        return _extract_pdf(data)
    if kind == "office":
        return [(None, uploads.office_to_text(ext, data))]
    return [(None, uploads.decode_text(data))]


def _split_long(text: str) -> list:
    pieces = []
    while len(text) > CHUNK_CHARS:
        low = CHUNK_CHARS // 2
        cut = max(text.rfind(" ", low, CHUNK_CHARS), text.rfind("\n", low, CHUNK_CHARS))
        if cut <= 0:
            cut = CHUNK_CHARS
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return [p for p in pieces if p]


def _make_chunks(segments: list) -> list:
    chunks = []
    for page, raw in segments:
        text = raw.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t\f\v]+", " ", text)
        pieces = []
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            pieces.extend(_split_long(paragraph) if len(paragraph) > CHUNK_CHARS else [paragraph])
        buf = ""
        for piece in pieces:
            if buf and len(buf) + len(piece) + 2 > CHUNK_CHARS:
                chunks.append({"page": page, "text": buf})
                tail = buf[-CHUNK_OVERLAP:]  # the end of one chunk starts the next, so ideas are not cut in half
                cut = tail.find(" ")
                buf = tail[cut + 1:].strip() if cut >= 0 else ""
            buf = (buf + "\n\n" + piece) if buf else piece
        if buf.strip():
            chunks.append({"page": page, "text": buf})
    return chunks


# ---------------------------------------------------------------
# Adding documents
# ---------------------------------------------------------------
def add_document(name: str, data: bytes) -> dict:
    """Read a file, split it into passages and add it. Adding a file with the same name again replaces
    the old one. Raises ValueError with a message meant for you."""
    if not data:
        raise ValueError("That file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"That file is too big (the limit is {MAX_FILE_BYTES // (1024 * 1024)} MB).")
    clean = uploads.safe_filename(name)
    kind = uploads.classify(clean)
    if kind not in ("text", "office", "pdf"):
        raise ValueError("Supported files: PDF, Word, Excel, PowerPoint, and text, notes, data or code files.")
    ext = Path(clean).suffix.lower()
    problem = uploads.validate_content(kind, ext, data)
    if problem:
        raise ValueError(problem)
    chunks = _make_chunks(_extract(kind, ext, data))
    if not chunks:
        raise ValueError("No text was found in that file.")
    truncated = len(chunks) > MAX_CHUNKS_PER_DOC
    chunks = chunks[:MAX_CHUNKS_PER_DOC]

    with LOCK:
        docs = [d for d in _load_docs() if d.get("name", "").lower() != clean.lower()]  # replace by name
        if len(docs) >= MAX_DOCS:
            raise ValueError(f"The knowledge base already has {MAX_DOCS} documents. Remove some first.")
        if sum(len(d.get("chunks", [])) for d in docs) + len(chunks) > MAX_TOTAL_CHUNKS:
            raise ValueError("The knowledge base is full. Remove some documents first.")
        doc = {"id": uuid.uuid4().hex[:8], "name": clean, "kind": kind, "size": len(data),
               "added": _now(), "truncated": truncated, "chunks": chunks}
        docs.append(doc)
        _save_docs(docs)
    return _public(doc)


def add_text(title: str, text: str) -> dict:
    """Add a note you typed (kept as a .md document with that title)."""
    title = " ".join((title or "").split()) or "note"
    text = (text or "").strip()
    if not text:
        raise ValueError("The note is empty.")
    name = title if Path(title).suffix.lower() in (".md", ".txt") else title + ".md"
    return add_document(name, text.encode("utf-8"))


# ---------------------------------------------------------------
# Search (BM25 over the passages)
# ---------------------------------------------------------------
def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> list:
    return [_stem(w) for w in re.findall(r"\w+", (text or "").lower()) if len(w) >= 2 and w not in STOP]


_CACHE = {"stamp": "unset", "chunks": [], "lengths": [], "postings": {}, "avg": 0.0}


def _stamp():
    try:
        stat = INDEX_FILE.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def _build(docs: list) -> dict:
    chunks, lengths, postings = [], [], {}
    for doc in docs:
        for chunk in doc.get("chunks", []):
            tokens = _tokens(chunk.get("text", ""))
            index = len(chunks)
            chunks.append((doc.get("name", "?"), chunk.get("page"), chunk.get("text", "")))
            lengths.append(len(tokens))
            counts: dict = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                postings.setdefault(token, []).append((index, count))
    average = (sum(lengths) / len(lengths)) if lengths else 0.0
    return {"chunks": chunks, "lengths": lengths, "postings": postings, "avg": average}


def _index() -> dict:
    stamp = _stamp()
    with LOCK:
        if _CACHE["stamp"] != stamp:
            _CACHE.update(_build(_load_docs()))
            _CACHE["stamp"] = stamp
        return _CACHE


def search(query: str, count: int = 5) -> list:
    """The best matching passages: [{"source", "document", "page", "score", "text"}]."""
    query = " ".join((query or "").split())[:MAX_QUERY_CHARS]
    terms = set(_tokens(query))
    if not terms:
        return []
    try:
        count = max(1, min(int(count or 5), MAX_RESULTS))
    except (TypeError, ValueError):
        count = 5
    index = _index()
    total = len(index["chunks"])
    if not total:
        return []
    scores: dict = {}
    for term in terms:
        postings = index["postings"].get(term)
        if not postings:
            continue
        idf = math.log(1 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
        for chunk_index, frequency in postings:
            length = index["lengths"][chunk_index] or 1
            norm = frequency * (BM25_K1 + 1) / (
                frequency + BM25_K1 * (1 - BM25_B + BM25_B * length / (index["avg"] or 1)))
            scores[chunk_index] = scores.get(chunk_index, 0.0) + idf * norm
    best = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:count]
    results = []
    for chunk_index, score in best:
        name, page, text = index["chunks"][chunk_index]
        results.append({
            "source": name + (f", page {page}" if page else ""),
            "document": name,
            "page": page,
            "score": round(score, 2),
            "text": text[:SNIPPET_CHARS],
        })
    return results