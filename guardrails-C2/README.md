# Ejemplo básico con Guardrails-AI — REVIEW

Este es un ejemplo **básico y didáctico** que muestra cómo integrar **guardrails-ai** en un agente que combina búsqueda web, un LLM local y validación de salidas estructuradas.

El objetivo es ilustrar de forma clara cómo aplicar **sanitización de entradas**, **separación de instrucciones y datos no confiables** y **validación post-hoc de la salida del modelo**.

Para configurar la librería, revisar las siguientes páginas oficiales:

- [Getting Started – Guardrails Server](https://www.guardrailsai.com/docs/getting_started/guardrails_server)
- [Guardrails Hub](https://hub.guardrailsai.com/)

> También se puede sustituir la librería por scripts propios, pero aquí se usa Guardrails para simplificar.

---

## Configuración de claves (API Keys)

El ejemplo requiere un archivo `.env` con las siguientes variables:

- **`TAVILY_API_KEY`** → necesaria para realizar búsquedas web a través de la API de Tavily.
- **`OLLAMA_BASE_URL`** *(opcional)* → URL del servidor de Ollama. Default: `http://localhost:11434`.
- **`OLLAMA_MODEL`** *(opcional)* → modelo de Ollama a utilizar. Default: `qwen3:8b`.

> **Nota:** la versión original del README mencionaba `OPENAI_API_KEY`, `GOOGLE_API_KEY` y `GOOGLE_CSE_ID`, pero el código actual **no usa OpenAI ni Google CSE**. El LLM corre localmente vía Ollama y la búsqueda se hace con Tavily.

Las claves son cargadas con la librería `python-dotenv`. **Guardrails** se integra mediante constructores como `Guard.for_string(...)` con los validadores del Hub (`ValidJson`, `RegexMatch`, etc.).

---

## ¿Qué hace el código?

1. **Carga las claves** desde `.env` y valida que `TAVILY_API_KEY` esté presente.
2. **Realiza una búsqueda web** usando `TavilySearch` (LangChain), limitada a 3 resultados.
3. **Sanitiza los resultados** mediante la función `sanitize_input`:
   - Decodifica entidades HTML.
   - Elimina `<script>`, `<style>` y todos los tags HTML.
   - Borra bloques de código markdown (``` ` ``` ``` `).
   - Quita literales de rol (`System:`, `User:`, `Assistant:`).
   - Elimina caracteres Unicode invisibles (zero-width y bidi marks).
   - Remueve URLs y normaliza whitespace.
   - Trunca a 8000 caracteres.
4. **Construye un prompt seguro con tres mensajes separados**:
   - `SystemMessage`: reglas del agente (formato JSON, no seguir instrucciones de contexto no confiable).
   - `HumanMessage`: tarea concreta y query del usuario.
   - `HumanMessage` con `name="web_context"`: contenido web sanitizado, delimitado por marcadores explícitos:
     ```
     UNTRUSTED CONTEXT START
     ... contenido sanitizado ...
     UNTRUSTED CONTEXT END
     ```
5. **Invoca el modelo de lenguaje** (`ChatOllama` con `format="json"`, `temperature=0`, `reasoning=False`) para generar la salida estructurada.
6. **Valida la salida** con Guardrails:
   - `ValidJson` contra un JSON Schema con campos `valid`, `city`, `country`, `population`, `notes`.
   - `RegexMatch` que rechaza cualquier salida que contenga code fences o tags de rol.
   - Ambos validadores usan `OnFailAction.EXCEPTION`, por lo que el primer fallo aborta el flujo.

En resumen, el script toma resultados de Tavily, los limpia, los pasa al modelo en un prompt estructurado y asegura que la respuesta final sea un **JSON válido y seguro**.

---

## Capas de defensa que están operando

El diseño aplica **defensa en profundidad**: en lugar de confiar en una sola barrera, encadena múltiples mecanismos de seguridad. Si una capa falla o es evadida, las siguientes siguen activas.

| Capa | Mecanismo | Riesgo que mitiga |
|---|---|---|
| **Input** | `sanitize_input()` sobre los resultados web | HTML/scripts inyectados, tags de rol disfrazados, Unicode invisible (`\u200B-\u200F`, `\u202A-\u202E`), URLs maliciosas, payloads excesivamente largos |
| **Prompt** | Bloque `UNTRUSTED CONTEXT START/END` + `SystemMessage` con reglas explícitas + mensaje separado con `name="web_context"` | Prompt injection indirecta a través de contenido recuperado (patrón *instructions vs data separation* recomendado por OWASP LLM01) |
| **Runtime** | `format="json"` en `ChatOllama` | Salida sintácticamente malformada: Ollama aplica constrained decoding y solo permite tokens que mantengan JSON válido |
| **Runtime** | `temperature=0` + `reasoning=False` | Variabilidad indeseada y chain-of-thought que filtre información del prompt |
| **Output (estructura)** | `ValidJson(schema=OUTPUT_SCHEMA)` | Campos faltantes, tipos incorrectos, valores fuera de rango (`population` entre 0 y 1e9), `additionalProperties` no autorizadas, `notes` mayor a 80 caracteres |
| **Output (contenido)** | `RegexMatch` con lookaheads negativos | Code fences markdown (``` ` ``` ```), role-leakage (`System:`, `User:`, `Assistant:`) que indique fuga de estructura del prompt |
| **Fail-fast** | `OnFailAction.EXCEPTION` en ambos validadores | Procesar silenciosamente respuestas inválidas o parcialmente correctas |

### Lectura por flujo

```
   query
     │
     ▼
┌─────────────┐
│   Tavily    │  ← fuente externa no confiable
└─────────────┘
     │
     ▼
┌──────────────────┐
│ sanitize_input() │  ← Capa 1: limpieza de entrada
└──────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  System + Human + Untrusted  │  ← Capa 2: separación de roles
│  (UNTRUSTED CONTEXT block)   │
└──────────────────────────────┘
     │
     ▼
┌────────────────────────┐
│ ChatOllama format=json │  ← Capa 3: constrained decoding
└────────────────────────┘
     │
     ▼
┌────────────────────────────────┐
│ Guardrails: ValidJson + Regex  │  ← Capa 4: validación post-hoc
└────────────────────────────────┘
     │
     ▼
   output validado
```

### Observaciones

- **Capas 1 y 2** son las que mitigan **prompt injection indirecta** (el riesgo principal cuando el agente consume contenido web).
- **Capas 3 y 4** son las que garantizan que el contrato de salida se cumpla, independientemente de lo que el modelo "quiera" decir.
- En producción se podría considerar reemplazar `OnFailAction.EXCEPTION` por `REASK` (re-prompting con el error) o `FIX` para campos reparables, y agregar telemetría de validaciones fallidas para detectar drift del modelo.

---

## Uso en producción

Este ejemplo se mantiene en un solo archivo para **facilitar la lectura**. En un entorno de producción debería **modularizarse** para mejorar mantenibilidad y escalabilidad. Una estructura recomendable sería:

- `config.py` → carga de claves y configuración de librerías.
- `sanitizer.py` → funciones de sanitización y preprocesamiento de texto.
- `validators.py` → definición de esquemas JSON y configuración de Guardrails.
- `llm_agent.py` → funciones que interactúan con el modelo de lenguaje.
- `search.py` → lógica para búsquedas en Tavily u otros proveedores.
- `main.py` → punto de entrada que orquesta el flujo completo.

> Con esta separación, el código resulta más claro, permite pruebas unitarias independientes y facilita reemplazar componentes (por ejemplo, cambiar de motor de búsqueda, de LLM o de validador).

Adicionalmente, en producción conviene:

- Loguear cada validación fallida con la salida cruda del modelo, para análisis posterior.
- Implementar reintentos con backoff para llamadas a Tavily y al LLM.
- Versionar el `OUTPUT_SCHEMA` y los validadores junto con el modelo, ya que un cambio de modelo puede romper el contrato.
- Considerar un wrapper de circuit breaker si Ollama corre como servicio dedicado.

---

## Referencias

- [Guardrails-AI: Getting Started](https://www.guardrailsai.com/docs/getting_started/guardrails_server)
- [Guardrails Hub](https://hub.guardrailsai.com/)
- [LangChain – Tavily Search](https://python.langchain.com/docs/integrations/tools/tavily_search/)
- [Ollama – Structured outputs (`format=json`)](https://ollama.com/blog/structured-outputs)
- [OWASP Top 10 for LLM Applications – LLM01: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
