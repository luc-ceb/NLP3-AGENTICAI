"""KPIs determinísticos del Caso 2 (volumen y facturación por PDV y mes).

Sin Text-to-SQL: las métricas se calculan con SQL fija y robusta (decisión de
diseño del MVP). La fuente es agregada diaria por (sucursal, producto), sin
clientes/tickets/unidades, así que los KPIs son de **volumen** (kilos) y
**facturación estimada** (Σ kilos × precio, proxy: el precio es por unidad). El
% en promoción acompaña como contexto. Eje temporal: interanual sobre kilos
(volumen real, sin inflación); la facturación se compara solo vs la red.
"""
from .metrics import (
    FamiliaKpi, PDVKpis, Periodo, compute_kpis, latest_month, list_pdvs,
    periodo_por_defecto,
)
from .deviations import Desvio, detectar_desvio

__all__ = [
    "FamiliaKpi", "PDVKpis", "Periodo", "compute_kpis", "latest_month",
    "list_pdvs", "periodo_por_defecto", "Desvio", "detectar_desvio",
]
