"""KPIs determinísticos del Caso 2 (volumen/tráfico por PDV y mes).

Sin Text-to-SQL: las métricas se calculan con SQL fija y robusta (decisión de
diseño del MVP). Solo se usan columnas de volumen (no hay precio/ingreso):
kilos, unidades, visitas (proxy de ticket) y clientes únicos.

Nota de datos: `numero` NO es un id de ticket — tiene 6 valores 1:1 con las 6
sucursales (es un código de sucursal/lista). El proxy de "ticket" es la VISITA:
una compra de un cliente en una sucursal un día -> distinct (customerid, fecha).
"""
from .metrics import PDVKpis, ProductoKpi, compute_kpis, latest_month, list_pdvs
from .deviations import Desvio, detectar_desvio

__all__ = [
    "PDVKpis", "ProductoKpi", "compute_kpis", "latest_month", "list_pdvs",
    "Desvio", "detectar_desvio",
]
