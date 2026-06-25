"""Detección de desvíos basada en reglas (sin LLM).

A partir de un :class:`PDVKpis` arma candidatos a desvío y elige el MÁS relevante
para diagnosticar contra el manual (Caso 2).

Prioridad (decisión de diseño): se prioriza el desempeño **vs la red** (volumen y
facturación contra la media de la red en el mismo mes) sobre la variación
**interanual** (YoY de kilos). Ambas comparaciones están libres de estacionalidad,
pero miden cosas distintas: vs red aísla a la sucursal que desentona de sus pares
hoy, mientras que la interanual detecta el deterioro propio respecto de un año
atrás. La facturación SOLO se evalúa vs red (su YoY estaría dominado por la
inflación). Las caídas por familia (línea comercial) y la intensidad de promoción
quedan como apoyo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import PDVKpis

# Umbral mínimo de variación (en %) para considerar un desvío "relevante".
UMBRAL_PCT = 10.0
# Para que una caída de familia (línea comercial) sea relevante, su volumen previo
# debe ser significativo y la baja, material en kilos (las familias tienen bases
# grandes —cientos de kg/mes—, así que además se exige el umbral porcentual).
FAM_KILOS_PREV_MIN = 50.0
FAM_KILOS_PERDIDOS_MIN = 30.0


# Niveles de prioridad (menor = más relevante operativamente). Permiten ordenar
# desvíos entre sucursales (Caso 5) de forma coherente con la selección por
# sucursal: un problema de tienda vs la red pesa más que una caída de familia,
# aunque esta última tenga un % mayor sobre una base más chica.
PRIO_VS_RED = 0
PRIO_YOY = 1
PRIO_FAMILIA = 2


@dataclass
class Desvio:
    """Un desvío detectado y priorizado."""
    dimension: str          # clave de la métrica (ej. 'kilos_red')
    titulo: str             # etiqueta legible
    valor_pct: float        # variación con signo (%)
    severidad: float        # magnitud (abs del pct), para ordenar dentro de un nivel
    direccion: str          # 'caida' | 'suba'
    prioridad: int          # nivel operativo (PRIO_VS_RED < PRIO_YOY < PRIO_FAMILIA)
    detalle: str            # frase descriptiva con cifras
    hipotesis: str          # hipótesis operativa para auditar contra la norma
    contexto: list[str] = field(default_factory=list)  # señales de apoyo (YoY, promo, productos)

    @property
    def es_relevante(self) -> bool:
        return self.severidad >= UMBRAL_PCT

    @property
    def orden(self) -> tuple[int, float]:
        """Clave de orden para rankear desvíos (Caso 5): menor primero.

        Ordena por nivel de prioridad y, dentro del nivel, por mayor severidad.
        Uso: ``sorted(desvios, key=lambda d: d.orden)`` deja primero al más grave.
        """
        return (self.prioridad, -self.severidad)


def _mk(dimension: str, titulo: str, pct: float | None, prioridad: int,
        detalle: str, hipotesis: str) -> Desvio | None:
    if pct is None:
        return None
    return Desvio(
        dimension=dimension, titulo=titulo, valor_pct=pct, severidad=abs(pct),
        direccion="caida" if pct < 0 else "suba", prioridad=prioridad,
        detalle=detalle, hipotesis=hipotesis)


def _vs_red(k: PDVKpis) -> list[Desvio]:
    """Candidatos primarios: desempeño de la sucursal vs la media de la red."""
    cands = [
        _mk("kilos_red", "Volumen de kilos vs red", k.kilos_var_red, PRIO_VS_RED,
            f"Kilos {k.kilos:.0f} vs media de la red {k.kilos_red_media:.0f} "
            f"({k.kilos_var_red:+.1f}%)." if k.kilos_var_red is not None else "",
            "Volumen de la sucursal por debajo de la media de la red en el mismo "
            "mes (desempeño propio, no estacional); revisar estándares de "
            "conservación, exhibición y cadena de frío del manual."),
        _mk("facturacion_red", "Facturación vs red", k.facturacion_var_red, PRIO_VS_RED,
            f"Facturación est. {k.facturacion:,.0f} vs media de la red "
            f"{k.facturacion_red_media:,.0f} ({k.facturacion_var_red:+.1f}%)."
            if k.facturacion_var_red is not None else "",
            "Facturación de la sucursal por debajo de la media de la red; revisar "
            "mix de productos, venta sugerida (upselling) y prácticas de venta del "
            "manual."),
    ]
    return [c for c in cands if c is not None]


def _yoy(k: PDVKpis) -> list[Desvio]:
    """Candidatos secundarios: variación interanual de volumen (mismo mes, año anterior)."""
    cands = [
        _mk("kilos_yoy", "Volumen de kilos (interanual)", k.kilos_var_yoy, PRIO_YOY,
            f"Kilos {k.kilos:.0f} vs {k.kilos_yoy:.0f} el mismo período del año anterior "
            f"({k.kilos_var_yoy:+.1f}%)." if k.kilos_var_yoy is not None else "",
            "Caída del volumen de ventas respecto del mismo período del año anterior; "
            "revisar conservación, quiebres de stock y cadena de frío."),
    ]
    return [c for c in cands if c is not None]


def _familia(k: PDVKpis) -> Desvio | None:
    """Candidato de caída de familia (línea comercial), solo si es material y relevante."""
    for p in k.caidas_familias:  # ya ordenados por mayor caída
        perdidos = p.kilos_yoy - p.kilos
        if (p.var_yoy is not None and p.var_yoy <= -UMBRAL_PCT
                and p.kilos_yoy >= FAM_KILOS_PREV_MIN
                and perdidos >= FAM_KILOS_PERDIDOS_MIN):
            return _mk(
                "familia_caida", f"Caída de familia: {p.familia}", p.var_yoy,
                PRIO_FAMILIA,
                f"{p.familia}: {p.kilos:.1f} kg vs {p.kilos_yoy:.1f} kg el mismo "
                f"período del año anterior ({p.var_yoy:+.1f}%, -{perdidos:.0f} kg).",
                f"Caída material de la línea '{p.familia}'; revisar conservación, "
                "reposición y exhibición de esa familia según el manual operativo.")
    return None


def _contexto(k: PDVKpis, elegido: Desvio) -> list[str]:
    """Señales de apoyo para enmarcar el desvío elegido."""
    ctx: list[str] = []
    if k.kilos_var_yoy is not None:
        ctx.append(f"Kilos interanual {k.kilos_var_yoy:+.1f}%.")
    # Intensidad de promoción vs la red (puede explicar parte de un desvío).
    if k.pct_promocion or k.pct_promocion_red_media:
        ctx.append(f"En promoción {k.pct_promocion:.1f}% de kilos "
                   f"(red {k.pct_promocion_red_media:.1f}%).")
    if k.caidas_familias and elegido.dimension != "familia_caida":
        p = k.caidas_familias[0]
        ctx.append(f"Mayor caída de familia: {p.familia} ({p.var_yoy:+.1f}%).")
    return ctx


def detectar_desvio(k: PDVKpis) -> Desvio | None:
    """Elige el desvío más relevante, priorizando el desempeño vs la red.

    Solo se devuelven CAÍDAS (problemas operativos accionables): una sucursal que
    rinde por encima de la red no tiene un problema que auditar y debe abstenerse.

    Orden de prioridad:
      1. Caída **vs la red** (volumen o facturación, mismo período) que supere el umbral.
      2. Si la sucursal no cae vs la red, caída **interanual** de volumen (fallback).
      3. Si tampoco, una **caída de familia** (línea comercial) material y relevante.

    Entre caídas del mismo nivel elige la de mayor magnitud. Devuelve ``None`` (→
    abstención) si la sucursal no presenta ninguna caída relevante. El desvío
    elegido se enriquece con ``contexto`` (interanual, promoción, familia).
    """
    def elegir(cands: list[Desvio]) -> Desvio | None:
        caidas = [c for c in cands if c.es_relevante and c.direccion == "caida"]
        return max(caidas, key=lambda c: c.severidad) if caidas else None

    # El fallback interanual solo aplica si la sucursal NO está por encima de la
    # red en volumen: si ya supera a sus pares hoy, el foco operativo está en otra
    # sucursal y esta debe abstenerse antes de mirar su propia evolución interanual.
    vs_red = elegir(_vs_red(k))
    por_encima_de_red = (k.kilos_var_red or 0) > 0
    elegido = vs_red or (None if por_encima_de_red else elegir(_yoy(k))) or _familia(k)
    if elegido is not None:
        elegido.contexto = _contexto(k, elegido)
    return elegido
