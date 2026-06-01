"""Ingesta de la base de conocimiento -> chunks con metadatos para RAG.

Recorre recursivamente data/raw buscando documentos (.md, .pdf, .txt) e ignora
los datos tabulares (.xlsx/.csv, que maneja la capa DuckDB). Produce chunks con
metadatos para trazabilidad/citación:

  - El manual operativo (.md) se parte por estructura: SECCIÓN > Tema, conservando
    el rango de timestamps [MM:SS] como ancla de cita (ej. [manual, 08:15-09:30]).
  - Los PDF de recursos se parten por página, conservando el número de página.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DOC_EXTS = (".md", ".pdf", ".txt")
MAX_CHARS = 900          # ~ límite cómodo para all-MiniLM-L6-v2 (256 tokens)
OVERLAP_CHARS = 150

TS_RE = re.compile(r"\[(\d{1,2}:\d{2})\]")
SECCION_RE = re.compile(r"^##\s*SECCIÓN:\s*(.+?)\s*$")
TEMA_RE = re.compile(r"^###\s*Tema:\s*(.+?)\s*$")
ORIGEN_RE = re.compile(r"^\*Archivo origen:\s*(.+?)\s*\*$")

# Mapa palabra-clave (sobre el path normalizado) -> tipo_doc
TIPO_DOC_RULES = [
    ("camara_frio",        ("camara", "frio")),
    ("recepcion_mercaderia", ("recepcion", "mercaderia")),
    ("diseno_tiendas",     ("diseno", "funcional")),
    ("percepcion_riesgo",  ("percepcion", "riesgo")),
    ("catalogo_productos", ("productos",)),
    ("auditoria",          ("auditoria",)),
    ("manual_operativo",   ("manual", "operativo")),
]


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w]+", "_", s.strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "doc"


def _infer_tipo_doc(rel_path: str) -> str:
    norm = _slug(rel_path)
    for tipo, keys in TIPO_DOC_RULES:
        if all(k in norm for k in keys):
            return tipo
    return "otro"


def _strip_ts(text: str) -> str:
    return TS_RE.sub("", text).strip()


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    tipo_doc: str
    text: str
    n_chars: int = 0
    section: str | None = None
    tema: str | None = None
    page: int | None = None
    ts_start: str | None = None
    ts_end: str | None = None

    def __post_init__(self):
        self.n_chars = len(self.text)


def _window_lines(lines: list[str], max_chars: int, overlap: int) -> list[str]:
    """Agrupa líneas en bloques de ~max_chars con solapamiento por caracteres."""
    blocks: list[str] = []
    buf: list[str] = []
    size = 0
    for ln in lines:
        if buf and size + len(ln) > max_chars:
            blocks.append("\n".join(buf).strip())
            # arrancar el siguiente bloque con un solapamiento (últimas líneas)
            carry, csize = [], 0
            for prev in reversed(buf):
                if csize + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                csize += len(prev)
            buf, size = carry[:], csize
        buf.append(ln)
        size += len(ln)
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]


def load_markdown(path: Path, raw_dir: Path) -> list[Chunk]:
    rel = str(path.relative_to(raw_dir))
    doc_id = _slug(path.stem)
    tipo = _infer_tipo_doc(rel)
    chunks: list[Chunk] = []

    section: str | None = None
    tema: str | None = None
    body: list[str] = []
    idx = 0

    def flush():
        nonlocal idx
        if not body:
            return
        text_block = "\n".join(body).strip()
        if not text_block:
            return
        for piece in _window_lines(body, MAX_CHARS, OVERLAP_CHARS):
            ts = TS_RE.findall(piece)
            clean = _strip_ts(piece)
            if not clean:
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::{idx:04d}",
                doc_id=doc_id, source=rel, tipo_doc=tipo,
                text=clean, section=section, tema=tema,
                ts_start=ts[0] if ts else None,
                ts_end=ts[-1] if ts else None,
            ))
            idx += 1

    for line in path.read_text(encoding="utf-8").splitlines():
        m_sec = SECCION_RE.match(line)
        m_tema = TEMA_RE.match(line)
        if m_sec:
            flush(); body = []
            section = m_sec.group(1); tema = None
            continue
        if m_tema:
            flush(); body = []
            tema = m_tema.group(1)
            continue
        if ORIGEN_RE.match(line) or line.strip() in ("", "---") or line.startswith("# "):
            continue
        body.append(line)
    flush()
    return chunks


def load_pdf(path: Path, raw_dir: Path) -> list[Chunk]:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError("Instalá pymupdf para leer PDFs: pip install pymupdf") from e

    rel = str(path.relative_to(raw_dir))
    doc_id = _slug(path.stem)
    tipo = _infer_tipo_doc(rel)
    chunks: list[Chunk] = []
    idx = 0
    with fitz.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            text = (page.get_text() or "").strip()
            if not text:
                continue
            lines = [ln for ln in text.splitlines() if ln.strip()]
            for piece in _window_lines(lines, MAX_CHARS, OVERLAP_CHARS):
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}::p{pno:03d}_{idx:04d}",
                    doc_id=doc_id, source=rel, tipo_doc=tipo,
                    text=piece, page=pno,
                ))
                idx += 1
    return chunks


def build_chunks(raw_dir: str | Path) -> list[Chunk]:
    raw = Path(raw_dir)
    chunks: list[Chunk] = []
    docs = sorted(p for p in raw.rglob("*") if p.suffix.lower() in DOC_EXTS)
    if not docs:
        log.warning("No se encontraron documentos (.md/.pdf/.txt) en %s", raw)
    for p in docs:
        try:
            new = load_pdf(p, raw) if p.suffix.lower() == ".pdf" else load_markdown(p, raw)
            chunks.extend(new)
            log.info("Ingestado %s -> %d chunks", p.relative_to(raw), len(new))
        except Exception as e:  # noqa: BLE001
            log.error("Falló la ingesta de %s: %s", p, e)
    return chunks


def save_jsonl(chunks: list[Chunk], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
