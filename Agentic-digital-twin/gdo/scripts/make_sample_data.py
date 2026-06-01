"""Genera datos sintéticos con estructura plausible para probar el pipeline.
Reemplazá estos archivos por tus .xlsx reales en data/raw/ y todo sigue andando.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

SUCURSALES = ["Centro", "Nueva Cordoba", "Cerro", "Alta Cordoba", "Villa Allende"]
PRODUCTOS = ["1/4 kg", "1/2 kg", "1 kg", "Cucurucho", "Palito", "Postre"]
FRANJAS = ["Mañana", "Tarde", "Noche"]
DIAS = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]


def ventas(n=2000):
    fechas = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 120, n), "D")
    return pd.DataFrame({
        "id_venta": np.arange(1, n + 1),
        "fecha": fechas,
        "sucursal": rng.choice(SUCURSALES, n),
        "producto": rng.choice(PRODUCTOS, n),
        "gramaje_gr": rng.choice([250, 500, 1000, 80, 60, 150], n),
        "importe": np.round(rng.uniform(1500, 9000, n), 2),
        "dia_semana": rng.choice(DIAS, n, p=[.12, .12, .12, .12, .15, .19, .18]),
        "franja_horaria": rng.choice(FRANJAS, n, p=[.25, .35, .40]),
        "ticket_duration_min": np.round(rng.gamma(2.0, 1.6, n) + 1, 1),
    })


def encuestas(n, signo):
    pos = ["Excelente atención", "Helado riquísimo", "Muy rápido", "Local impecable"]
    neg = ["Mucha espera", "Helado derretido", "Atención lenta", "Faltaba stock"]
    txt = pos if signo > 0 else neg
    return pd.DataFrame({
        "id_encuesta": np.arange(1, n + 1),
        "fecha": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 120, n), "D"),
        "sucursal": rng.choice(SUCURSALES, n),
        "nps": rng.integers(8, 11, n) if signo > 0 else rng.integers(0, 7, n),
        "comentario": rng.choice(txt, n),
    })


if __name__ == "__main__":
    ventas().to_excel(RAW / "datos-ventas.xlsx", index=False)
    encuestas(600, +1).to_excel(RAW / "encuestas-buena-experiencia.xlsx", index=False)
    encuestas(400, -1).to_excel(RAW / "encuestas-mala-experiencia.xlsx", index=False)
    print(f"Datos sintéticos escritos en {RAW}")
