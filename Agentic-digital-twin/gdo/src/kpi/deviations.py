"""Detección de desvíos basada en reglas (sin LLM).

A partir de un :class:`PDVKpis` arma candidatos a desvío y elige el MÁS relevante
para diagnosticar contra el manual (Caso 2).

Prioridad (decisión de diseño): se prioriza el desempeño **vs la red** sobre la
variación **interanual** (YoY). Ambas comparaciones están libres de estacionalidad
—la red se compara en el mismo mes y la interanual contra el mismo mes del año
anterior—, pero miden cosas distintas: vs red aísla a la sucursal que desentona
de sus pares hoy, mientras que la interanual detecta el deterioro propio respecto
de un año atrás (incluso si toda la red cayó). Se usa vs red como señal primaria
y la interanual como fallback/contexto; las caídas de producto quedan como apoyo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import PDVKpis

# Umbral mínimo de variación (en %) para considerar un desvío "relevante".
UMBRAL_PCT = 10.0
# Para que una caída de producto sea relevante, su volumen previo debe ser
# significativo (evita que productos de 2 kg con -90% dominen la priorización).
PROD_KILOS_PREV_MIN = 20.0
PROD_KILOS_PERDIDOS_MIN = 15.0


# Niveles de prioridad (menor = más relevante operativamente). Permiten ordenar
# desvíos entre sucursales (Caso 5) de forma coherente con la selección por
# sucursal: un problema de tienda vs la red pesa más que una caída de producto,
# aunque esta última tenga un % mayor sobre una base chica.
PRIO_VS_RED = 0
PRIO_YOY = 1
PRIO_PRODUCTO = 2


@dataclass
class Desvio:
    """Un desvío detectado y priorizado."""
    dimension: str          # clave de la métrica (ej. 'kilos_red')
    titulo: str             # etiqueta legible
    valor_pct: float        # variación con signo (%)
    severidad: float        # magnitud (abs del pct), para ordenar dentro de un nivel
    direccion: str          # 'caida' | 'suba'
    prioridad: int          # nivel operativo (PRIO_VS_RED < PRIO_YOY < PRIO_PRODUCTO)
    detalle: str            # frase descriptiva con cifras
    hipotesis: str          # hipótesis operativa para auditar contra la norma
    contexto: list[str] = field(default_factory=list)  # señales de apoyo (YoY, productos)

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
        _mk("visitas_red", "Tráfico de visitas vs red", k.visitas_var_red, PRIO_VS_RED,
            f"Visitas {k.visitas} vs media de la red {k.visitas_red_media:.0f} "
            f"({k.visitas_var_red:+.1f}%)." if k.visitas_var_red is not None else "",
            "Tráfico de la sucursal por debajo de la media de la red; revisar "
            "atención al cliente, horarios y experiencia en el local según la norma."),
        _mk("clientes_red", "Clientes únicos vs red", k.clientes_var_red, PRIO_VS_RED,
            f"Clientes {k.clientes} vs media de la red {k.clientes_red_media:.0f} "
            f"({k.clientes_var_red:+.1f}%)." if k.clientes_var_red is not None else "",
            "Clientes únicos por debajo de la media de la red; revisar "
            "fidelización y calidad de servicio según la norma."),
    ]
    return [c for c in cands if c is not None]


def _yoy(k: PDVKpis) -> list[Desvio]:
    """Candidatos secundarios: variación interanual (mismo mes del año anterior)."""
    cands = [
        _mk("kilos_yoy", "Volumen de kilos (interanual)", k.kilos_var_yoy, PRIO_YOY,
            f"Kilos {k.kilos:.0f} vs {k.kilos_yoy:.0f} el mismo mes del año anterior "
            f"({k.kilos_var_yoy:+.1f}%)." if k.kilos_var_yoy is not None else "",
            "Caída del volumen de ventas respecto del mismo mes del año anterior; "
            "revisar conservación, quiebres de stock y cadena de frío."),
        _mk("visitas_yoy", "Tráfico de visitas (interanual)", k.visitas_var_yoy, PRIO_YOY,
            f"Visitas {k.visitas} vs {k.visitas_yoy} el mismo mes del año anterior "
            f"({k.visitas_var_yoy:+.1f}%)." if k.visitas_var_yoy is not None else "",
            "Caída de tráfico respecto del mismo mes del año anterior; revisar "
            "atención al cliente y experiencia en el local."),
        _mk("kilos_visita_yoy", "Kilos por visita (interanual)",
            k.kilos_por_visita_var_yoy, PRIO_YOY,
            f"Kilos/visita {k.kilos_por_visita:.2f} vs "
            f"{k.kilos_por_visita_yoy:.2f} el mismo mes del año anterior "
            f"({k.kilos_por_visita_var_yoy:+.1f}%)."
            if k.kilos_por_visita_var_yoy is not None else "",
            "Caída del tamaño de compra (kilos por visita); revisar surtido, "
            "porcionado y prácticas de venta del manual."),
    ]
    return [c for c in cands if c is not None]


def _producto(k: PDVKpis) -> Desvio | None:
    """Candidato de caída de producto, solo si es materialmente significativo."""
    for p in k.caidas_productos:  # ya ordenados por mayor caída
        perdidos = p.kilos_yoy - p.kilos
        if p.kilos_yoy >= PROD_KILOS_PREV_MIN and perdidos >= PROD_KILOS_PERDIDOS_MIN:
            return _mk(
                "producto_caida", f"Caída de producto: {p.descripcion}", p.var_yoy,
                PRIO_PRODUCTO,
                f"{p.descripcion}: {p.kilos:.1f} kg vs {p.kilos_yoy:.1f} kg el mismo "
                f"mes del año anterior ({p.var_yoy:+.1f}%, -{perdidos:.0f} kg).",
                f"Caída material de '{p.descripcion}'; revisar conservación, "
                "reposición y manejo de ese producto según el manual operativo.")
    return None


def _contexto(k: PDVKpis, elegido: Desvio) -> list[str]:
    """Señales de apoyo para enmarcar el desvío elegido."""
    ctx: list[str] = []
    if k.kilos_var_yoy is not None:
        ctx.append(f"Kilos interanual {k.kilos_var_yoy:+.1f}%.")
    if k.caidas_productos and elegido.dimension != "producto_caida":
        p = k.caidas_productos[0]
        ctx.append(f"Mayor caída de producto: {p.descripcion} ({p.var_yoy:+.1f}%).")
    return ctx


def detectar_desvio(k: PDVKpis) -> Desvio | None:
    """Elige el desvío más relevante, priorizando el desempeño vs la red.

    Solo se devuelven CAÍDAS (problemas operativos accionables): una sucursal que
    rinde por encima de la red no tiene un problema que auditar y debe abstenerse.

    Orden de prioridad:
      1. Caída **vs la red** (mismo mes) que supere el umbral.
      2. Si la sucursal no cae vs la red, caída **interanual** relevante (fallback).
      3. Si tampoco, una **caída de producto** materialmente significativa.

    Entre caídas del mismo nivel elige la de mayor magnitud. Devuelve ``None`` (→
    abstención) si la sucursal no presenta ninguna caída relevante. El desvío
    elegido se enriquece con ``contexto`` (incl. la señal interanual).
    """
    def elegir(cands: list[Desvio]) -> Desvio | None:
        caidas = [c for c in cands if c.es_relevante and c.direccion == "caida"]
        return max(caidas, key=lambda c: c.severidad) if caidas else None

    # El fallback interanual solo aplica si la sucursal NO está por encima de la
    # red en volumen: si ya supera a sus pares hoy, el foco operativo está en otra
    # sucursal y esta debe abstenerse antes de mirar su propia evolución interanual.
    vs_red = elegir(_vs_red(k))
    por_encima_de_red = (k.kilos_var_red or 0) > 0
    elegido = vs_red or (None if por_encima_de_red else elegir(_yoy(k))) or _producto(k)
    if elegido is not None:
        elegido.contexto = _contexto(k, elegido)
    return elegido
