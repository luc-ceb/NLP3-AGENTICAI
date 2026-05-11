from dataclasses import dataclass
from typing import List, Optional
import re
import unicodedata

@dataclass
class Document:
    id: str
    text: str
    source: str = ""
    page: Optional[int] = None

# tokenización básica (casefold + letras con acentos y dígitos) 
_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")

def simple_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())

# utilidades para chunking basado en oraciones 
# divide en oraciones simples: punto/exclamación/interrogación + espacio + mayúscula
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-ZÁÉÍÓÚÑ])")
_SOFT_HYPH = re.compile(r"[\u00AD]")               # soft hyphen
_HARD_HYPH = re.compile(r"(\w)-\n(\w)")            # palabra-\ncontinuación

def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("\u00A0", " ")  # NBSP → espacio
    # de-hyphenation
    t = _SOFT_HYPH.sub("", t)
    t = _HARD_HYPH.sub(r"\1\2", t)
    # compacta espacios
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # no asumimos que todo empiece en mayúscula; si no hay cortes, devolvemos el texto entero
    parts = _SENT_SPLIT.split(text.strip())
    if len(parts) <= 1:
        return [text.strip()]
    # limpiar espacios
    return [" ".join(p.split()) for p in parts if p.strip()]

def _count_tokens(t: str) -> int:
    return len(simple_tokenize(t))

def _slide_merge(sents: List[str], max_tok: int, overlap: int) -> List[str]:
    """
    Arma chunks concatenando oraciones hasta max_tok.
    Aplica solapamiento de ~overlap tokens usando la cola del chunk previo.
    Hace split adicional por ; : si una oración es enorme.
    """
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    def flush():
        nonlocal buf, buf_tokens
        if buf_tokens >= 40:  # mínimo útil
            txt = " ".join(buf).split()
            # hard cap suave (max_tok + 20)
            if len(txt) > max_tok + 20:
                txt = txt[:max_tok + 20]
            chunks.append(" ".join(txt))
        buf, buf_tokens = [], 0

    for s in sents:
        st = _count_tokens(s)
        # si una oración es enorme, intentá partir por ; :
        if st > max_tok:
            pieces = re.split(r"[;:]\s+", s)
            for p in pieces:
                pt = _count_tokens(p)
                if pt == 0:
                    continue
                if pt > max_tok:
                    # si sigue siendo enorme, recortá
                    p = " ".join(p.split()[:max_tok])
                    pt = _count_tokens(p)
                if buf_tokens + pt <= max_tok:
                    buf.append(p); buf_tokens += pt
                else:
                    flush()
                    # overlap: tomar cola del último chunk
                    if chunks:
                        tail = " ".join(chunks[-1].split()[-overlap:])
                        buf, buf_tokens = [tail], _count_tokens(tail)
                    buf.append(p); buf_tokens += pt
            continue

        # caso normal
        if buf_tokens + st <= max_tok:
            buf.append(s); buf_tokens += st
        else:
            flush()
            if chunks:
                tail = " ".join(chunks[-1].split()[-overlap:])
                buf, buf_tokens = [tail], _count_tokens(tail)
            buf.append(s); buf_tokens += st

    if buf:
        flush()
    return chunks

def chunk_text(text: str, max_tokens: int = 200, overlap: int = 60) -> List[str]:
    """
    Chunking:
      - target ~200 tokens por chunk
      - solapamiento ~60 tokens
      - descarta chunks < 20 tokens
      - recorta > 220 tokens
    """
    if not text:
        return []
    norm = _normalize(text)
    sents = _split_sentences(norm)
    if not sents:
        return []
    chunks = _slide_merge(sents, max_tok=max_tokens, overlap=overlap)

    # filtro final de longitud
    out: List[str] = []
    for ch in chunks:
        nt = _count_tokens(ch)
        if 20 <= nt <= 220:  
            out.append(ch)
    return out
