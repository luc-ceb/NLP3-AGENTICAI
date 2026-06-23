"""Genera un manual operativo SINTÉTICO usando un LLM: toma la estructura (secciones
y temas) de un manual real y escribe, por cada tema, un guion genérico y ficticio de
heladería. NO copia ninguna transcripción real ni usa marcas reales.

- Reutiliza tu config de LLM (FAST_PROVIDER/FAST_MODEL; Groq por defecto).
- Cachea cada tema en data/processed/manual_cache.jsonl -> resumible si se corta.
- Retry con backoff para los rate limits.

Uso:
  python scripts/make_sample_manual.py <manual_real.md> <salida.md> [--limit N] [--sleep S]
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
rng = random.Random(42)

_BRAND = re.compile(r"(?i)grido|\bguido\b")


def sane(s: str) -> str:
    return _BRAND.sub("Polar", s)


def topic(slug: str) -> str:
    parts = slug.split("-")
    while parts and parts[0].isdigit():
        parts.pop(0)
    return " ".join(parts).strip() or slug.replace("-", " ")


def ts(sec: int) -> str:
    return f"[{sec // 60:02d}:{sec % 60:02d}]"


PROMPT = """Sos redactor de un curso interno de operaciones para una heladería \
(marca ficticia "Polar"). Escribí el guion hablado de un breve video instructivo \
sobre el tema: "{topic}" (sección: "{section}").

Requisitos:
- 6 a 10 oraciones, concretas y realistas sobre la operación de una heladería.
- Procedimientos GENÉRICOS del rubro (pasos, controles, buenas prácticas).
- NADA de empresas o marcas reales, ni datos personales.
- Tono de instructor explicando el paso a paso.
- Devolvé SOLO el guion: una oración por línea, sin títulos, sin numeración, sin timestamps.
"""


def parse_structure(text: str):
    section, items = None, []
    for line in text.splitlines():
        m = re.match(r"^## SECCIÓN:\s*(.+)$", line)
        if m:
            section = m.group(1).strip()
            items.append(("section", section))
            continue
        m = re.match(r"^### Tema:\s*(.+)$", line)
        if m and section is not None:
            items.append(("tema", section, m.group(1).strip()))
    return items


def _complete_retry(llm, prompt, tries=5):
    delay = 2.0
    for i in range(tries):
        try:
            return llm.complete(prompt)
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"    reintento {i+1} ({e})", flush=True)
            time.sleep(delay)
            delay *= 2


def content_for(llm, section, slug, cache, cache_fp, sleep):
    key = f"{section}|{slug}"
    if key in cache:
        return cache[key]
    txt = sane(_complete_retry(llm, PROMPT.format(topic=topic(slug), section=section)))
    lines = [ln.strip("-•* \t") for ln in txt.splitlines() if ln.strip()]
    cache[key] = lines
    cache_fp.write(json.dumps({"key": key, "lines": lines}, ensure_ascii=False) + "\n")
    cache_fp.flush()
    if sleep:
        time.sleep(sleep)
    return lines


def generate(items, llm, out_path, cache_path, limit=None, sleep=0.0):
    cache = {}
    if Path(cache_path).exists():
        for line in Path(cache_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["key"]] = d["lines"]
        print(f"caché: {len(cache)} temas ya generados")

    temas = [it for it in items if it[0] == "tema"]
    if limit:
        keep = {f"{s}|{sl}" for _, s, sl in temas[:limit]}
    out = ["# MANUAL OPERATIVO Y DE GESTIÓN DE LA FRANQUICIA (DEMO SINTÉTICO)", "",
           "Documento de ejemplo con contenido **ficticio y genérico** de heladería, "
           "generado con un LLM. Conserva la estructura de temas de un manual operativo "
           "pero no contiene información real de ninguna empresa.", "", "---", ""]
    done = 0
    cache_fp = open(cache_path, "a", encoding="utf-8")
    try:
        for it in items:
            if it[0] == "section":
                out += [f"## SECCIÓN: {sane(it[1])}", ""]
                continue
            _, section, slug = it
            section, slug = sane(section), sane(slug)
            if limit and f"{section}|{slug}" not in keep:
                continue
            done += 1
            print(f"[{done}] {section} > {slug}", flush=True)
            lines = content_for(llm, section, slug, cache, cache_fp, sleep)
            sec = 0
            body = []
            for ln in lines:
                body.append(f"{ts(sec)} {ln}")
                sec += rng.randint(4, 8)
            out += [f"### Tema: {slug}",
                    f"*Archivo origen: videos > {section.lower()} > {slug}.mp4*", ""]
            out += body + ["", "---", ""]
    finally:
        cache_fp.close()
    Path(out_path).write_text("\n".join(out), encoding="utf-8")
    print(f"\nManual sintético escrito en {out_path}  ({done} temas)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = "data/raw/manual-operativo-general/manual_operativo_completo.md" # args[0] if args else "data/raw/manual-operativo-general/manual_operativo_completo.md"
    dst = args[1] if len(args) > 1 else "manual_operativo_completo_sintetico.md"
    limit = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), None)
    sleep = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--sleep=")), 0.0)

    from src.llm.clients import make_llm
    llm = make_llm(os.getenv("FAST_MODEL"), provider=os.getenv("FAST_PROVIDER", "groq"))
    items = parse_structure(Path(src).read_text(encoding="utf-8"))
    generate(items, llm, dst, ROOT / "data" / "processed" / "manual_cache.jsonl",
             limit=limit, sleep=sleep)


if __name__ == "__main__":
    main()
