# Práctica 2 — Experimentos con Tokens, Temperatura, Embeddings y Comparativa de Modelos

## Metadatos

| Campo            | Detalle                                      |
|------------------|----------------------------------------------|
| **Duración**     | 27 minutos                                   |
| **Complejidad**  | Media                                        |
| **Nivel Bloom**  | Aplicar (*Apply*)                            |
| **Módulo**       | 2 — Componentes técnicos clave de GenAI      |
| **Costo estimado** | ~$0.10 USD por estudiante (GPT-4o-mini)    |

---

## Descripción General

En este laboratorio explorarás de forma experimental los cuatro pilares técnicos que determinan el comportamiento práctico de los modelos de lenguaje: tokenización, embeddings, parámetros de generación y selección de modelos. Partirás del concepto de token visto en la Lección 2.1 y lo llevarás a la práctica midiendo costos reales, generando representaciones vectoriales y observando cómo la temperatura transforma la naturaleza de las respuestas. Todos los experimentos quedarán registrados en DataFrames de Pandas para facilitar el análisis comparativo.

---

## Objetivos de Aprendizaje

Al completar este laboratorio serás capaz de:

- [ ] Usar `tiktoken` para contar tokens de prompts reales en español e inglés y calcular su costo estimado de forma programática.
- [ ] Generar embeddings con `text-embedding-3-small`, calcular similitud coseno entre frases y visualizar clusters semánticos en un mapa 2D con PCA.
- [ ] Observar y documentar el efecto de la temperatura (0.0, 0.5, 0.9, 1.2) en tareas creativas y analíticas, registrando métricas en Pandas.
- [ ] Comparar GPT-4o-mini y GPT-3.5-turbo en costo por token, latencia y calidad percibida usando criterios objetivos.

---

## Prerrequisitos

### Conocimiento previo

- Haber completado el Lab 01-00-01 **o** conocer qué es un modelo de lenguaje y para qué sirve.
- Python básico: definir funciones, usar bucles `for`, trabajar con diccionarios y listas.
- Comprender qué es un token según la Lección 2.1 (regla de ~4 caracteres por token en inglés).

### Acceso y credenciales

- API key de OpenAI activa con crédito disponible (mínimo $0.10 USD).
- Variable de entorno `OPENAI_API_KEY` configurada **antes** de iniciar el notebook.
- Acceso a internet estable (las cuatro secciones realizan llamadas a la API de OpenAI).

> ⚠️ **Privacidad:** No ingreses datos personales, confidenciales ni información de tu empresa en ningún prompt durante este laboratorio. Consulta la política de datos de OpenAI en [platform.openai.com/privacy](https://platform.openai.com/privacy).

---

## Entorno de Laboratorio

### Hardware mínimo recomendado

| Recurso      | Mínimo          | Recomendado     |
|--------------|-----------------|-----------------|
| RAM          | 8 GB            | 16 GB           |
| CPU          | 4 núcleos       | 8 núcleos       |
| Almacenamiento | 500 MB libres | 1 GB libres     |
| Internet     | 10 Mbps         | 25 Mbps         |

### Software requerido

| Paquete               | Versión mínima | Uso en este lab                        |
|-----------------------|----------------|----------------------------------------|
| Python                | 3.10           | Entorno base                           |
| `openai`              | 1.30           | Llamadas a la API (chat + embeddings)  |
| `tiktoken`            | 0.7            | Conteo de tokens antes de enviar       |
| `numpy`               | 1.26           | Cálculo de similitud coseno            |
| `scikit-learn`        | 1.4            | PCA para reducción dimensional         |
| `pandas`              | 2.1            | Registro de métricas                   |
| `matplotlib`          | 3.8            | Visualización de clusters y gráficos   |
| `seaborn`             | 0.13           | Estilo visual de gráficos              |
| `python-dotenv`       | 1.0            | Carga segura de variables de entorno   |

### Comandos de instalación y configuración inicial

Ejecuta la siguiente celda **una sola vez** al inicio del notebook:

```bash
pip install openai tiktoken numpy scikit-learn pandas matplotlib seaborn python-dotenv --quiet
```

Luego crea el archivo `.env` en el directorio de trabajo (o configura la variable directamente en tu entorno):

```bash
# En terminal (Linux/macOS/Windows PowerShell)
echo "OPENAI_API_KEY=sk-..." > .env
```

> **Alternativa sin costo:** Si no dispones de API key de OpenAI, consulta al instructor sobre la configuración con Ollama + `sentence-transformers`. Las Secciones 1 y 2 pueden adaptarse con modelos locales.

---

## Instrucciones Paso a Paso

### Configuración Inicial del Notebook

**Objetivo:** Importar todas las dependencias, cargar la API key y definir constantes globales que se usarán en las cuatro secciones.

**Instrucciones:**

1. Abre Jupyter Notebook o JupyterLab y crea un nuevo notebook llamado `lab02_experimentos.ipynb`.
2. En la primera celda, copia y ejecuta el siguiente bloque de configuración:

```python
# ─── Celda 1: Importaciones y configuración global ───────────────────────────
import os
import time
import warnings
warnings.filterwarnings("ignore")

import tiktoken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from openai import OpenAI

# Carga la API key desde el archivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No se encontró OPENAI_API_KEY. Verifica tu archivo .env.")

client = OpenAI(api_key=api_key)

# ─── Constantes de precios (USD por millón de tokens, referencia 2024) ────────
PRECIOS = {
    "gpt-4o-mini": {
        "entrada": 0.15,   # USD / 1M tokens de entrada
        "salida":  0.60,   # USD / 1M tokens de salida
    },
    "gpt-3.5-turbo": {
        "entrada": 0.50,
        "salida":  1.50,
    },
    "text-embedding-3-small": {
        "entrada": 0.02,
        "salida":  0.00,
    },
}

# Estilo visual unificado
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams["figure.dpi"] = 110

print("✅ Configuración completada. API key cargada correctamente.")
print(f"   Versión de openai SDK: {__import__('openai').__version__}")
```

**Salida esperada:**
```
✅ Configuración completada. API key cargada correctamente.
   Versión de openai SDK: 1.x.x
```

**Verificación:** Si ves un `ValueError`, revisa que el archivo `.env` exista en el mismo directorio que el notebook y que la clave comience con `sk-`.

---

### Sección 1 — Tokenización: Conteo, Idioma y Costo

**Objetivo:** Usar `tiktoken` para tokenizar textos en español e inglés, comparar eficiencia entre idiomas y calcular costos estimados de prompts reales.

**Instrucciones:**

1. **Define la función de conteo y costo.** Copia la siguiente celda:

```python
# ─── Celda 2: Función de tokenización con costo ───────────────────────────────
def analizar_texto(texto: str, modelo: str = "gpt-4o-mini") -> dict:
    """
    Tokeniza un texto y calcula métricas de costo estimado.

    Args:
        texto: Cadena de texto a analizar.
        modelo: Nombre del modelo de OpenAI.

    Returns:
        Diccionario con métricas de tokenización.
    """
    # tiktoken usa el mismo codificador para gpt-4o-mini y gpt-4o
    nombre_codificador = "cl100k_base"
    codificador = tiktoken.get_encoding(nombre_codificador)

    tokens_ids = codificador.encode(texto)
    n_tokens = len(tokens_ids)
    n_chars = len(texto)
    n_palabras = len(texto.split())

    # Decodifica cada token para mostrar la segmentación
    segmentos = [codificador.decode([tid]) for tid in tokens_ids]

    # Costo estimado como si fuera solo prompt de entrada
    precio_entrada = PRECIOS.get(modelo, {}).get("entrada", 0.15)
    costo_usd = (n_tokens / 1_000_000) * precio_entrada

    return {
        "texto": texto[:60] + "..." if len(texto) > 60 else texto,
        "modelo": modelo,
        "n_tokens": n_tokens,
        "n_chars": n_chars,
        "n_palabras": n_palabras,
        "ratio_tokens_palabra": round(n_tokens / max(n_palabras, 1), 2),
        "ratio_tokens_char": round(n_tokens / max(n_chars, 1), 3),
        "costo_usd": round(costo_usd, 8),
        "segmentos": segmentos,
    }

print("✅ Función analizar_texto() definida.")
```

2. **Ejecuta el experimento comparativo español vs. inglés:**

```python
# ─── Celda 3: Experimento español vs. inglés ─────────────────────────────────
textos_experimento = [
    # Pares equivalentes (misma idea, distinto idioma)
    ("ES", "La inteligencia artificial está transformando la industria."),
    ("EN", "Artificial intelligence is transforming the industry."),
    ("ES", "La tokenización es fundamental para entender los costos de la API."),
    ("EN", "Tokenization is fundamental to understanding API costs."),
    ("ES", "El procesamiento del lenguaje natural permite a las máquinas comprender texto."),
    ("EN", "Natural language processing allows machines to understand text."),
    # Textos técnicos con vocabulario especializado
    ("ES", "Implementación de arquitecturas transformer con mecanismos de atención multi-cabeza."),
    ("EN", "Implementation of transformer architectures with multi-head attention mechanisms."),
    # Prompt real de sistema
    ("PROMPT", "Eres un asistente experto en análisis financiero. Resume los puntos clave del siguiente informe trimestral en no más de cinco oraciones claras y concisas."),
]

resultados_tokens = []
for idioma, texto in textos_experimento:
    resultado = analizar_texto(texto)
    resultado["idioma"] = idioma
    resultados_tokens.append(resultado)
    print(f"[{idioma}] '{texto[:50]}...' → {resultado['n_tokens']} tokens "
          f"(ratio: {resultado['ratio_tokens_palabra']} tok/palabra)")

df_tokens = pd.DataFrame(resultados_tokens)
print("\n📊 DataFrame de tokenización creado con", len(df_tokens), "registros.")
```

3. **Visualiza la comparación de eficiencia:**

```python
# ─── Celda 4: Visualización comparativa ES vs EN ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gráfico 1: tokens por texto
colores = {"ES": "#2196F3", "EN": "#FF9800", "PROMPT": "#9C27B0"}
bar_colors = [colores.get(row["idioma"], "#607D8B") for _, row in df_tokens.iterrows()]
axes[0].barh(
    range(len(df_tokens)),
    df_tokens["n_tokens"],
    color=bar_colors
)
axes[0].set_yticks(range(len(df_tokens)))
axes[0].set_yticklabels([f"[{r['idioma']}] {r['texto'][:35]}..." for _, r in df_tokens.iterrows()],
                         fontsize=7)
axes[0].set_xlabel("Número de tokens")
axes[0].set_title("Tokens por texto")
leyenda = [mpatches.Patch(color=v, label=k) for k, v in colores.items()]
axes[0].legend(handles=leyenda, fontsize=8)

# Gráfico 2: ratio tokens/palabra por idioma
df_idiomas = df_tokens[df_tokens["idioma"].isin(["ES", "EN"])].copy()
axes[1].bar(
    df_idiomas.index.astype(str),
    df_idiomas["ratio_tokens_palabra"],
    color=[colores[i] for i in df_idiomas["idioma"]]
)
axes[1].set_xlabel("Índice de texto")
axes[1].set_ylabel("Tokens por palabra")
axes[1].set_title("Eficiencia de tokenización (menor = más eficiente)")
axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.6, label="1 tok/palabra")
axes[1].legend(fontsize=8)

plt.suptitle("Sección 1: Análisis de Tokenización", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("s1_tokenizacion.png", bbox_inches="tight")
plt.show()
print("✅ Gráfico guardado como s1_tokenizacion.png")
```

4. **Calcula el costo de un prompt real a escala:**

```python
# ─── Celda 5: Proyección de costos a escala ───────────────────────────────────
prompt_sistema = """Eres un asistente experto en servicio al cliente para una empresa de telecomunicaciones.
Tu rol es responder preguntas sobre facturación, planes y soporte técnico de forma clara, empática y concisa.
Siempre saluda al cliente por su nombre si está disponible. No inventes información que no esté en el contexto."""

prompt_usuario = "¿Cuál es el costo de mi plan actual y cuándo vence mi contrato?"

info_sistema = analizar_texto(prompt_sistema)
info_usuario = analizar_texto(prompt_usuario)
tokens_totales_entrada = info_sistema["n_tokens"] + info_usuario["n_tokens"]

# Proyección a diferentes volúmenes
volumenes = [100, 1_000, 10_000, 100_000, 1_000_000]
proyecciones = []
for vol in volumenes:
    costo_entrada = (tokens_totales_entrada * vol / 1_000_000) * PRECIOS["gpt-4o-mini"]["entrada"]
    # Asumimos respuesta promedio de 150 tokens
    tokens_salida_estimados = 150
    costo_salida = (tokens_salida_estimados * vol / 1_000_000) * PRECIOS["gpt-4o-mini"]["salida"]
    proyecciones.append({
        "llamadas": vol,
        "tokens_entrada_total": tokens_totales_entrada * vol,
        "costo_entrada_usd": round(costo_entrada, 4),
        "costo_salida_usd": round(costo_salida, 4),
        "costo_total_usd": round(costo_entrada + costo_salida, 4),
    })

df_proyeccion = pd.DataFrame(proyecciones)
print("📊 Proyección de costos para el prompt de servicio al cliente:")
print(f"   Tokens en prompt: {tokens_totales_entrada} (sistema: {info_sistema['n_tokens']} + usuario: {info_usuario['n_tokens']})")
print()
print(df_proyeccion.to_string(index=False))
```

**Salida esperada (aproximada):**
```
📊 Proyección de costos para el prompt de servicio al cliente:
   Tokens en prompt: ~75 (sistema: ~62 + usuario: ~13)

    llamadas  tokens_entrada_total  costo_entrada_usd  costo_salida_usd  costo_total_usd
         100                 7500             0.0011            0.0090           0.0101
       1,000                75000             0.0113            0.0900           0.1013
      10,000               750000             0.1125            0.9000           1.0125
     100,000              7500000             1.1250            9.0000          10.1250
   1,000,000             75000000            11.2500           90.0000         101.2500
```

**Verificación:** El ratio tokens/palabra para español debe ser ligeramente mayor que para inglés (típicamente 1.1–1.4 vs. 1.0–1.2). Si ves valores iguales, verifica que los textos sean equivalentes semánticos.

---

### Sección 2 — Embeddings: Similitud Semántica y Visualización

**Objetivo:** Generar embeddings con `text-embedding-3-small`, calcular similitud coseno entre frases temáticas e identificar clusters semánticos mediante PCA.

**Instrucciones:**

1. **Define las frases temáticas y genera los embeddings:**

```python
# ─── Celda 6: Corpus de frases temáticas ─────────────────────────────────────
frases_corpus = [
    # Grupo A: Tecnología e IA (índices 0-4)
    "Los modelos de lenguaje grande procesan texto mediante transformers.",
    "El aprendizaje automático requiere grandes cantidades de datos etiquetados.",
    "Las redes neuronales profundas aprenden representaciones jerárquicas.",
    "La inteligencia artificial generativa crea contenido nuevo a partir de patrones.",
    "Los embeddings representan el significado semántico en espacios vectoriales.",
    # Grupo B: Gastronomía (índices 5-9)
    "La paella valenciana lleva arroz, azafrán, pollo y verduras frescas.",
    "El sushi japonés combina arroz vinagrado con pescado crudo de alta calidad.",
    "Cocinar a fuego lento permite que los sabores se integren profundamente.",
    "La fermentación es una técnica culinaria ancestral presente en todo el mundo.",
    "El chocolate negro tiene propiedades antioxidantes beneficiosas para la salud.",
    # Grupo C: Finanzas (índices 10-14)
    "La diversificación de cartera reduce el riesgo de pérdidas significativas.",
    "Los mercados bursátiles reflejan las expectativas futuras de los inversores.",
    "La inflación erosiona el poder adquisitivo del dinero con el tiempo.",
    "Los bonos del tesoro se consideran activos de bajo riesgo en finanzas.",
    "El análisis fundamental evalúa el valor intrínseco de una empresa.",
    # Grupo D: Deportes (índices 15-19)
    "El fútbol es el deporte más popular del mundo con miles de millones de seguidores.",
    "El entrenamiento de alta intensidad mejora la capacidad cardiovascular.",
    "Los Juegos Olímpicos reúnen a atletas de todas las naciones cada cuatro años.",
    "La nutrición deportiva optimiza el rendimiento y la recuperación muscular.",
    "El baloncesto requiere coordinación, velocidad y trabajo en equipo.",
]

ETIQUETAS_GRUPO = {
    range(0, 5):   ("Tecnología/IA", "#2196F3"),
    range(5, 10):  ("Gastronomía",   "#4CAF50"),
    range(10, 15): ("Finanzas",      "#FF9800"),
    range(15, 20): ("Deportes",      "#E91E63"),
}

def obtener_grupo(idx):
    for rango, (nombre, color) in ETIQUETAS_GRUPO.items():
        if idx in rango:
            return nombre, color
    return "Otro", "#607D8B"

print(f"✅ Corpus definido: {len(frases_corpus)} frases en 4 grupos temáticos.")
```

2. **Genera los embeddings mediante la API:**

```python
# ─── Celda 7: Generación de embeddings ───────────────────────────────────────
def generar_embeddings(textos: list, modelo: str = "text-embedding-3-small") -> np.ndarray:
    """
    Genera embeddings para una lista de textos usando la API de OpenAI.

    Args:
        textos: Lista de strings a vectorizar.
        modelo: Modelo de embeddings a usar.

    Returns:
        Array de NumPy con forma (n_textos, dimensión_embedding).
    """
    respuesta = client.embeddings.create(
        input=textos,
        model=modelo
    )
    # Ordena por índice para garantizar correspondencia con la lista de entrada
    vectores = [item.embedding for item in sorted(respuesta.data, key=lambda x: x.index)]
    return np.array(vectores)

print("⏳ Generando embeddings para 20 frases... (esto consume ~0.001 USD)")
t_inicio = time.time()
embeddings = generar_embeddings(frases_corpus)
t_fin = time.time()

print(f"✅ Embeddings generados en {t_fin - t_inicio:.2f}s")
print(f"   Forma del array: {embeddings.shape}  → ({len(frases_corpus)} frases × {embeddings.shape[1]} dimensiones)")
print(f"   Costo estimado: ${(len(frases_corpus) / 1_000_000) * PRECIOS['text-embedding-3-small']['entrada']:.6f} USD")
```

**Salida esperada:**
```
✅ Embeddings generados en ~1.2s
   Forma del array: (20, 1536)  → (20 frases × 1536 dimensiones)
   Costo estimado: $0.000000 USD  (fracción de centavo)
```

3. **Calcula la similitud coseno entre pares seleccionados:**

```python
# ─── Celda 8: Similitud coseno entre pares ────────────────────────────────────
def similitud_coseno(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    norma_a = np.linalg.norm(vec_a)
    norma_b = np.linalg.norm(vec_b)
    if norma_a == 0 or norma_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norma_a * norma_b))

# Pares de comparación: (índice_a, índice_b, descripción_esperada)
pares_comparacion = [
    (0, 1,  "Tecnología vs Tecnología (mismo grupo)"),
    (0, 4,  "IA vs Embeddings (mismo grupo, más específico)"),
    (5, 6,  "Gastronomía vs Gastronomía (mismo grupo)"),
    (10, 11,"Finanzas vs Finanzas (mismo grupo)"),
    (0, 5,  "Tecnología vs Gastronomía (grupos distintos)"),
    (0, 10, "Tecnología vs Finanzas (grupos distintos)"),
    (5, 15, "Gastronomía vs Deportes (grupos distintos)"),
    (10, 15,"Finanzas vs Deportes (grupos distintos)"),
    (1, 14, "ML vs Análisis fundamental (distantes)"),
    (18, 19,"Nutrición deportiva vs Baloncesto (mismo grupo, diferente aspecto)"),
]

resultados_similitud = []
print("📐 Similitudes coseno entre pares de frases:\n")
for idx_a, idx_b, descripcion in pares_comparacion:
    sim = similitud_coseno(embeddings[idx_a], embeddings[idx_b])
    resultados_similitud.append({
        "frase_a": frases_corpus[idx_a][:45] + "...",
        "frase_b": frases_corpus[idx_b][:45] + "...",
        "descripcion": descripcion,
        "similitud": round(sim, 4),
    })
    barra = "█" * int(sim * 20)
    print(f"  {sim:.4f} |{barra:<20}| {descripcion}")

df_similitud = pd.DataFrame(resultados_similitud)
print(f"\n✅ DataFrame df_similitud creado con {len(df_similitud)} pares.")
```

**Salida esperada (valores aproximados):**
```
  0.6821 |█████████████       | Tecnología vs Tecnología (mismo grupo)
  0.5934 |███████████         | IA vs Embeddings (mismo grupo, más específico)
  0.7102 |██████████████      | Gastronomía vs Gastronomía (mismo grupo)
  0.6589 |█████████████       | Finanzas vs Finanzas (mismo grupo)
  0.2341 |████                | Tecnología vs Gastronomía (grupos distintos)
  0.2987 |█████               | Tecnología vs Finanzas (grupos distintos)
  ...
```

4. **Visualiza los clusters semánticos con PCA:**

```python
# ─── Celda 9: Visualización PCA de embeddings ────────────────────────────────
# Reducción de 1536 dimensiones a 2 para visualización
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)

varianza_explicada = pca.explained_variance_ratio_
print(f"📊 Varianza explicada por PCA: PC1={varianza_explicada[0]:.1%}, PC2={varianza_explicada[1]:.1%}")
print(f"   Total: {sum(varianza_explicada):.1%} de la información original")

fig, ax = plt.subplots(figsize=(11, 8))

grupos_vistos = set()
for idx, frase in enumerate(frases_corpus):
    nombre_grupo, color = obtener_grupo(idx)
    label = nombre_grupo if nombre_grupo not in grupos_vistos else None
    grupos_vistos.add(nombre_grupo)

    ax.scatter(
        embeddings_2d[idx, 0],
        embeddings_2d[idx, 1],
        color=color,
        s=120,
        alpha=0.85,
        label=label,
        zorder=3
    )
    # Etiqueta abreviada
    texto_corto = frase[:30] + "..."
    ax.annotate(
        f"{idx}: {texto_corto}",
        (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
        fontsize=6.5,
        xytext=(5, 4),
        textcoords="offset points",
        alpha=0.8
    )

ax.set_xlabel(f"PC1 ({varianza_explicada[0]:.1%} varianza)", fontsize=10)
ax.set_ylabel(f"PC2 ({varianza_explicada[1]:.1%} varianza)", fontsize=10)
ax.set_title("Sección 2: Mapa Semántico de Embeddings (PCA 2D)\n"
             "text-embedding-3-small — 20 frases, 4 grupos temáticos",
             fontsize=11, fontweight="bold")
ax.legend(title="Grupo temático", fontsize=9, title_fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("s2_embeddings_pca.png", bbox_inches="tight")
plt.show()
print("✅ Mapa semántico guardado como s2_embeddings_pca.png")
```

**Verificación:** Los puntos de cada grupo temático deben agruparse visualmente en el mapa 2D. Si los clusters no son claramente distinguibles, es normal: PCA retiene solo ~15–25% de la varianza; los grupos son más separables en el espacio de 1536 dimensiones.

---

### Sección 3 — Temperatura y Top-p: Efecto en la Generación

**Objetivo:** Ejecutar la misma tarea creativa y la misma tarea analítica con temperaturas 0.0, 0.5, 0.9 y 1.2, documentar diferencias cuantitativas y cualitativas en un DataFrame.

**Instrucciones:**

1. **Define la función de llamada con métricas:**

```python
# ─── Celda 10: Función de generación con métricas ────────────────────────────
def llamar_modelo_con_metricas(
    prompt_sistema: str,
    prompt_usuario: str,
    modelo: str = "gpt-4o-mini",
    temperatura: float = 0.7,
    max_tokens: int = 200,
) -> dict:
    """
    Llama al modelo y registra métricas de uso y tiempo.

    Returns:
        Diccionario con respuesta, tokens usados, latencia y costo.
    """
    t_inicio = time.time()
    respuesta = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user",   "content": prompt_usuario},
        ],
        temperature=temperatura,
        max_tokens=max_tokens,
    )
    latencia = round(time.time() - t_inicio, 3)

    tokens_entrada = respuesta.usage.prompt_tokens
    tokens_salida  = respuesta.usage.completion_tokens
    costo = (
        (tokens_entrada / 1_000_000) * PRECIOS[modelo]["entrada"] +
        (tokens_salida  / 1_000_000) * PRECIOS[modelo]["salida"]
    )

    return {
        "modelo": modelo,
        "temperatura": temperatura,
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "latencia_s": latencia,
        "costo_usd": round(costo, 6),
        "respuesta": respuesta.choices[0].message.content.strip(),
        "finish_reason": respuesta.choices[0].finish_reason,
    }

print("✅ Función llamar_modelo_con_metricas() definida.")
```

2. **Ejecuta el experimento de temperatura:**

```python
# ─── Celda 11: Experimento de temperatura ────────────────────────────────────
temperaturas = [0.0, 0.5, 0.9, 1.2]

# ── Tarea A: Creativa ─────────────────────────────────────────────────────────
SISTEMA_CREATIVO = "Eres un escritor creativo con estilo poético y metafórico."
USUARIO_CREATIVO = ("Escribe exactamente 3 oraciones describiendo cómo se siente "
                    "aprender algo nuevo por primera vez.")

# ── Tarea B: Analítica ────────────────────────────────────────────────────────
SISTEMA_ANALITICO = "Eres un analista técnico preciso. Responde siempre con datos concretos."
USUARIO_ANALITICO = ("Lista exactamente 3 ventajas del modelo GPT-4o-mini sobre GPT-3.5-turbo "
                     "en formato: 1. [ventaja]: [explicación breve].")

resultados_temperatura = []

print("⏳ Ejecutando experimento de temperatura (8 llamadas a la API)...\n")
for temp in temperaturas:
    for tipo_tarea, sistema, usuario in [
        ("Creativa",  SISTEMA_CREATIVO,  USUARIO_CREATIVO),
        ("Analítica", SISTEMA_ANALITICO, USUARIO_ANALITICO),
    ]:
        print(f"  🌡️  Temperatura {temp} | Tarea {tipo_tarea}...")
        resultado = llamar_modelo_con_metricas(
            prompt_sistema=sistema,
            prompt_usuario=usuario,
            temperatura=temp,
            max_tokens=180,
        )
        resultado["tipo_tarea"] = tipo_tarea
        resultados_temperatura.append(resultado)
        time.sleep(0.5)  # Pausa cortés para evitar rate limiting

df_temperatura = pd.DataFrame(resultados_temperatura)
print(f"\n✅ Experimento completado. DataFrame df_temperatura: {df_temperatura.shape}")
```

3. **Visualiza y analiza los resultados:**

```python
# ─── Celda 12: Visualización del experimento de temperatura ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ── Gráfico 1: Longitud de respuesta por temperatura ─────────────────────────
for tipo, color in [("Creativa", "#2196F3"), ("Analítica", "#FF9800")]:
    subset = df_temperatura[df_temperatura["tipo_tarea"] == tipo]
    axes[0].plot(subset["temperatura"], subset["tokens_salida"],
                 marker="o", label=tipo, color=color, linewidth=2)
axes[0].set_xlabel("Temperatura")
axes[0].set_ylabel("Tokens de salida")
axes[0].set_title("Longitud de respuesta\nvs. temperatura")
axes[0].legend()
axes[0].set_xticks(temperaturas)

# ── Gráfico 2: Latencia por temperatura ──────────────────────────────────────
for tipo, color in [("Creativa", "#2196F3"), ("Analítica", "#FF9800")]:
    subset = df_temperatura[df_temperatura["tipo_tarea"] == tipo]
    axes[1].plot(subset["temperatura"], subset["latencia_s"],
                 marker="s", label=tipo, color=color, linewidth=2, linestyle="--")
axes[1].set_xlabel("Temperatura")
axes[1].set_ylabel("Latencia (segundos)")
axes[1].set_title("Latencia de respuesta\nvs. temperatura")
axes[1].legend()
axes[1].set_xticks(temperaturas)

# ── Gráfico 3: Costo acumulado por temperatura ────────────────────────────────
df_pivot = df_temperatura.pivot(index="temperatura", columns="tipo_tarea", values="costo_usd")
df_pivot.plot(kind="bar", ax=axes[2], color=["#FF9800", "#2196F3"], alpha=0.8)
axes[2].set_xlabel("Temperatura")
axes[2].set_ylabel("Costo (USD)")
axes[2].set_title("Costo por llamada\nvs. temperatura")
axes[2].tick_params(axis="x", rotation=0)

plt.suptitle("Sección 3: Efecto de la Temperatura en la Generación (GPT-4o-mini)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("s3_temperatura.png", bbox_inches="tight")
plt.show()
```

4. **Imprime las respuestas para análisis cualitativo:**

```python
# ─── Celda 13: Análisis cualitativo de respuestas ────────────────────────────
print("=" * 70)
print("ANÁLISIS CUALITATIVO — TAREA CREATIVA")
print("=" * 70)
for _, fila in df_temperatura[df_temperatura["tipo_tarea"] == "Creativa"].iterrows():
    print(f"\n🌡️  Temperatura {fila['temperatura']} ({fila['tokens_salida']} tokens):")
    print(f"   {fila['respuesta']}")
    print(f"   [Latencia: {fila['latencia_s']}s | Costo: ${fila['costo_usd']:.6f}]")

print("\n" + "=" * 70)
print("ANÁLISIS CUALITATIVO — TAREA ANALÍTICA")
print("=" * 70)
for _, fila in df_temperatura[df_temperatura["tipo_tarea"] == "Analítica"].iterrows():
    print(f"\n🌡️  Temperatura {fila['temperatura']} ({fila['tokens_salida']} tokens):")
    print(f"   {fila['respuesta']}")
    print(f"   [Latencia: {fila['latencia_s']}s | Costo: ${fila['costo_usd']:.6f}]")
```

**Verificación:** Para la tarea analítica, las respuestas con temperatura 0.0 deben ser casi idénticas si se ejecutan múltiples veces. Para la tarea creativa, temperatura 1.2 debe producir respuestas notablemente más variadas y ocasionalmente incoherentes. Si temperatura 1.2 genera texto sin sentido, eso es **exactamente el comportamiento esperado** para documentar.

---

### Sección 4 — Comparativa de Modelos: GPT-4o-mini vs. GPT-3.5-turbo

**Objetivo:** Evaluar ambos modelos en las mismas tareas usando criterios objetivos: costo por token, latencia, longitud de respuesta y calidad percibida.

**Instrucciones:**

1. **Define el benchmark de tareas:**

```python
# ─── Celda 14: Benchmark de tareas para comparativa de modelos ────────────────
tareas_benchmark = [
    {
        "id": "T1_resumen",
        "nombre": "Resumen técnico",
        "sistema": "Eres un asistente técnico experto. Sé preciso y conciso.",
        "usuario": ("Resume en exactamente 4 puntos clave las diferencias entre "
                    "aprendizaje supervisado, no supervisado y por refuerzo en ML."),
        "max_tokens": 250,
        "temperatura": 0.3,
    },
    {
        "id": "T2_codigo",
        "nombre": "Generación de código",
        "sistema": "Eres un experto en Python. Escribe código limpio, comentado y funcional.",
        "usuario": ("Escribe una función Python que calcule la similitud coseno "
                    "entre dos listas de números. Incluye docstring y un ejemplo de uso."),
        "max_tokens": 300,
        "temperatura": 0.2,
    },
    {
        "id": "T3_analisis",
        "nombre": "Análisis de negocio",
        "sistema": "Eres un consultor de negocios senior con experiencia en transformación digital.",
        "usuario": ("¿Cuáles son los 3 principales riesgos de implementar un chatbot de IA "
                    "generativa en el servicio al cliente de un banco? Sé específico."),
        "max_tokens": 280,
        "temperatura": 0.4,
    },
]

MODELOS_COMPARAR = ["gpt-4o-mini", "gpt-3.5-turbo"]
print(f"✅ Benchmark definido: {len(tareas_benchmark)} tareas × {len(MODELOS_COMPARAR)} modelos = "
      f"{len(tareas_benchmark) * len(MODELOS_COMPARAR)} llamadas totales.")
```

2. **Ejecuta el benchmark:**

```python
# ─── Celda 15: Ejecución del benchmark comparativo ───────────────────────────
resultados_comparativa = []

print("⏳ Ejecutando benchmark comparativo (6 llamadas a la API)...\n")
for tarea in tareas_benchmark:
    for modelo in MODELOS_COMPARAR:
        print(f"  🤖 {modelo} | {tarea['nombre']}...")
        resultado = llamar_modelo_con_metricas(
            prompt_sistema=tarea["sistema"],
            prompt_usuario=tarea["usuario"],
            modelo=modelo,
            temperatura=tarea["temperatura"],
            max_tokens=tarea["max_tokens"],
        )
        resultado["tarea_id"]     = tarea["id"]
        resultado["tarea_nombre"] = tarea["nombre"]
        resultados_comparativa.append(resultado)
        time.sleep(0.8)

df_comparativa = pd.DataFrame(resultados_comparativa)
print(f"\n✅ Benchmark completado. DataFrame df_comparativa: {df_comparativa.shape}")
```

3. **Genera el reporte comparativo:**

```python
# ─── Celda 16: Reporte comparativo de modelos ────────────────────────────────
print("=" * 65)
print("REPORTE COMPARATIVO: GPT-4o-mini vs. GPT-3.5-turbo")
print("=" * 65)

# Tabla resumen por modelo
resumen = df_comparativa.groupby("modelo").agg(
    latencia_promedio=("latencia_s",    "mean"),
    tokens_salida_promedio=("tokens_salida", "mean"),
    costo_total_usd=("costo_usd",       "sum"),
    costo_promedio_usd=("costo_usd",    "mean"),
).round(4)

print("\n📊 Resumen por modelo:")
print(resumen.to_string())

# Comparación por tarea
print("\n📋 Detalle por tarea:")
for tarea_id in df_comparativa["tarea_id"].unique():
    subset = df_comparativa[df_comparativa["tarea_id"] == tarea_id]
    nombre = subset["tarea_nombre"].iloc[0]
    print(f"\n  [{tarea_id}] {nombre}:")
    for _, fila in subset.iterrows():
        print(f"    {fila['modelo']:20s} → {fila['tokens_salida']:4d} tok salida | "
              f"{fila['latencia_s']:.2f}s | ${fila['costo_usd']:.6f}")
```

4. **Visualiza la comparativa final:**

```python
# ─── Celda 17: Visualización comparativa de modelos ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colores_modelo = {"gpt-4o-mini": "#1976D2", "gpt-3.5-turbo": "#388E3C"}

tareas_nombres = [t["nombre"] for t in tareas_benchmark]
x = np.arange(len(tareas_nombres))
ancho = 0.35

for i, (metrica, titulo, unidad) in enumerate([
    ("latencia_s",    "Latencia",             "segundos"),
    ("tokens_salida", "Tokens de salida",      "tokens"),
    ("costo_usd",     "Costo por llamada",     "USD"),
]):
    for j, modelo in enumerate(MODELOS_COMPARAR):
        subset = df_comparativa[df_comparativa["modelo"] == modelo]
        valores = [subset[subset["tarea_id"] == t["id"]][metrica].values[0]
                   for t in tareas_benchmark]
        offset = (j - 0.5) * ancho
        bars = axes[i].bar(x + offset, valores, ancho,
                           label=modelo, color=colores_modelo[modelo], alpha=0.85)
        axes[i].bar_label(bars, fmt="%.3f" if metrica == "costo_usd" else "%.1f",
                          fontsize=7, padding=2)

    axes[i].set_xlabel("Tarea")
    axes[i].set_ylabel(unidad)
    axes[i].set_title(f"{titulo} por tarea y modelo")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(tareas_nombres, fontsize=8, rotation=10, ha="right")
    axes[i].legend(fontsize=8)

plt.suptitle("Sección 4: Comparativa GPT-4o-mini vs. GPT-3.5-turbo",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("s4_comparativa_modelos.png", bbox_inches="tight")
plt.show()
print("✅ Comparativa guardada como s4_comparativa_modelos.png")
```

5. **Imprime las respuestas para evaluación cualitativa:**

```python
# ─── Celda 18: Respuestas para evaluación cualitativa ────────────────────────
for tarea in tareas_benchmark:
    print(f"\n{'='*65}")
    print(f"TAREA: {tarea['nombre'].upper()}")
    print(f"{'='*65}")
    for modelo in MODELOS_COMPARAR:
        fila = df_comparativa[
            (df_comparativa["tarea_id"] == tarea["id"]) &
            (df_comparativa["modelo"] == modelo)
        ].iloc[0]
        print(f"\n── {modelo} ──")
        print(fila["respuesta"])
        print(f"[{fila['tokens_salida']} tokens | {fila['latencia_s']}s | ${fila['costo_usd']:.6f}]")
```

**Verificación:** GPT-4o-mini debería ser consistentemente más económico (costo por token ~3–4× menor) que GPT-3.5-turbo, con calidad comparable o superior en la mayoría de las tareas. Si GPT-3.5-turbo resulta más barato, verifica que las constantes `PRECIOS` estén actualizadas según los precios vigentes de la API.

---

## Validación y Pruebas

Ejecuta la siguiente celda al finalizar las cuatro secciones para verificar que todos los artefactos se generaron correctamente:

```python
# ─── Celda 19: Validación final del laboratorio ───────────────────────────────
import os

print("🔍 VALIDACIÓN FINAL DEL LABORATORIO\n")
print("─" * 50)

# 1. Verificar DataFrames
checks = {
    "df_tokens": (df_tokens, 9, ["n_tokens", "costo_usd", "ratio_tokens_palabra"]),
    "df_similitud": (df_similitud, 10, ["similitud", "descripcion"]),
    "df_temperatura": (df_temperatura, 8, ["temperatura", "tipo_tarea", "respuesta"]),
    "df_comparativa": (df_comparativa, 6, ["modelo", "tarea_id", "costo_usd"]),
}

todos_ok = True
for nombre, (df, filas_min, columnas) in checks.items():
    ok_filas = len(df) >= filas_min
    ok_cols  = all(c in df.columns for c in columnas)
    estado   = "✅" if (ok_filas and ok_cols) else "❌"
    if not (ok_filas and ok_cols):
        todos_ok = False
    print(f"  {estado} {nombre}: {len(df)} filas, columnas requeridas {'presentes' if ok_cols else 'FALTANTES'}")

# 2. Verificar archivos de imagen
imagenes = ["s1_tokenizacion.png", "s2_embeddings_pca.png",
            "s3_temperatura.png",  "s4_comparativa_modelos.png"]
print()
for img in imagenes:
    existe = os.path.exists(img)
    estado = "✅" if existe else "❌"
    if not existe:
        todos_ok = False
    print(f"  {estado} {img}")

# 3. Verificar embeddings
print()
emb_ok = embeddings.shape == (20, 1536)
estado = "✅" if emb_ok else "❌"
if not emb_ok:
    todos_ok = False
print(f"  {estado} embeddings.shape == (20, 1536): {embeddings.shape}")

# 4. Resumen de costo total estimado
costo_total = (
    df_tokens["costo_usd"].sum() +
    df_temperatura["costo_usd"].sum() +
    df_comparativa["costo_usd"].sum()
)
print(f"\n💰 Costo total estimado del laboratorio: ${costo_total:.4f} USD")
print()
if todos_ok:
    print("🎉 ¡Laboratorio completado exitosamente! Todos los artefactos verificados.")
else:
    print("⚠️  Algunos artefactos requieren revisión. Verifica las celdas marcadas con ❌.")
```

**Salida esperada:**
```
🔍 VALIDACIÓN FINAL DEL LABORATORIO
──────────────────────────────────────────────────
  ✅ df_tokens: 9 filas, columnas requeridas presentes
  ✅ df_similitud: 10 filas, columnas requeridas presentes
  ✅ df_temperatura: 8 filas, columnas requeridas presentes
  ✅ df_comparativa: 6 filas, columnas requeridas presentes

  ✅ s1_tokenizacion.png
  ✅ s2_embeddings_pca.png
  ✅ s3_temperatura.png
  ✅ s4_comparativa_modelos.png

  ✅ embeddings.shape == (20, 1536): (20, 1536)

💰 Costo total estimado del laboratorio: $0.0XXX USD

🎉 ¡Laboratorio completado exitosamente! Todos los artefactos verificados.
```

---

## Resolución de Problemas

### Problema 1: Error `AuthenticationError` al llamar a la API

**Síntoma:**
```
openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided...'}}
```

**Causa:** La variable de entorno `OPENAI_API_KEY` no está cargada correctamente, el archivo `.env` no está en el directorio de trabajo del notebook, o la API key tiene un error tipográfico.

**Solución:**
```python
# Diagnóstico rápido: ejecuta estas líneas en una celda nueva
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # override=True fuerza la recarga
key = os.getenv("OPENAI_API_KEY", "NO_ENCONTRADA")
print(f"API Key cargada: {key[:8]}...{key[-4:] if len(key) > 12 else '(muy corta)'}")
print(f"Directorio actual: {os.getcwd()}")
print(f"Archivos .env encontrados: {[f for f in os.listdir('.') if f.endswith('.env')]}")
```

Si el archivo `.env` no aparece en el listado, créalo explícitamente:
```python
with open(".env", "w") as f:
    f.write("OPENAI_API_KEY=sk-TU_CLAVE_AQUI\n")
print("Archivo .env creado. Reinicia el kernel y vuelve a ejecutar desde la Celda 1.")
```

---

### Problema 2: `RateLimitError` o respuestas lentas durante el experimento de temperatura

**Síntoma:**
```
openai.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached...'}}
```
O el experimento de temperatura tarda más de 60 segundos en completarse.

**Causa:** Las cuentas nuevas de OpenAI tienen límites de velocidad bajos (RPM = requests per minute). El experimento ejecuta 8 llamadas consecutivas, lo que puede superar el límite en cuentas con tier gratuito o recién creadas.

**Solución:** Aumenta el tiempo de espera entre llamadas modificando la Celda 11:

```python
# Reemplaza time.sleep(0.5) por un backoff exponencial
import random

def llamar_con_reintento(max_intentos=3, **kwargs):
    """Llama al modelo con reintento automático ante RateLimitError."""
    for intento in range(max_intentos):
        try:
            return llamar_modelo_con_metricas(**kwargs)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                espera = (2 ** intento) + random.uniform(0, 1)
                print(f"  ⏳ Rate limit detectado. Esperando {espera:.1f}s (intento {intento+1}/{max_intentos})...")
                time.sleep(espera)
            else:
                raise e
    raise RuntimeError(f"Se superaron {max_intentos} intentos. Verifica tu cuota en platform.openai.com.")

# Luego en el bucle de la Celda 11, reemplaza:
#   resultado = llamar_modelo_con_metricas(...)
# por:
#   resultado = llamar_con_reintento(prompt_sistema=sistema, prompt_usuario=usuario,
#                                    temperatura=temp, max_tokens=180)
# Y aumenta la pausa:
time.sleep(2.0)  # En lugar de time.sleep(0.5)
```

---

## Limpieza

Al finalizar el laboratorio, ejecuta la siguiente celda para liberar memoria y organizar los artefactos generados:

```python
# ─── Celda 20: Limpieza y organización ────────────────────────────────────────
import shutil

# Crea directorio de resultados si no existe
os.makedirs("resultados_lab02", exist_ok=True)

# Mueve imágenes generadas
for img in ["s1_tokenizacion.png", "s2_embeddings_pca.png",
            "s3_temperatura.png",  "s4_comparativa_modelos.png"]:
    if os.path.exists(img):
        shutil.move(img, f"resultados_lab02/{img}")

# Exporta DataFrames a CSV para análisis posterior
df_tokens.to_csv("resultados_lab02/tokens_analisis.csv", index=False)
df_similitud.to_csv("resultados_lab02/similitud_embeddings.csv", index=False)
df_temperatura.drop(columns=["respuesta"]).to_csv(
    "resultados_lab02/temperatura_metricas.csv", index=False)
df_comparativa.drop(columns=["respuesta"]).to_csv(
    "resultados_lab02/comparativa_modelos.csv", index=False)

# Libera la variable de embeddings (1536 × 20 floats)
del embeddings
del embeddings_2d

print("✅ Limpieza completada.")
print("   Artefactos guardados en: ./resultados_lab02/")
print("   Archivos generados:")
for f in sorted(os.listdir("resultados_lab02")):
    size_kb = os.path.getsize(f"resultados_lab02/{f}") / 1024
    print(f"     {f} ({size_kb:.1f} KB)")

print("\n⚠️  Recuerda: NO compartas tu API key ni los archivos .env en repositorios públicos.")
```

---

## Resumen

En este laboratorio completaste cuatro experimentos interconectados que ilustran los componentes técnicos fundamentales de los modelos de lenguaje:

| Sección | Concepto | Herramienta clave | Hallazgo principal |
|---------|----------|-------------------|--------------------|
| 1 — Tokens | Tokenización y costo | `tiktoken` | El español genera ~10–20% más tokens que el inglés para el mismo significado; los tokens de salida dominan el costo a escala |
| 2 — Embeddings | Representación semántica | `text-embedding-3-small` + PCA | Frases del mismo tema se agrupan en el espacio vectorial; la similitud coseno cuantifica la proximidad semántica |
| 3 — Temperatura | Diversidad vs. coherencia | `chat.completions` | Temperatura 0.0 = determinista e ideal para tareas analíticas; temperatura ≥1.0 = creativo pero potencialmente incoherente |
| 4 — Modelos | Costo/calidad/latencia | GPT-4o-mini vs. GPT-3.5-turbo | GPT-4o-mini ofrece mejor relación costo-calidad para la mayoría de tareas empresariales |

### Conceptos clave reforzados

- **Regla práctica:** 1 token ≈ 4 caracteres en inglés / ¾ de palabra; en español el ratio es ligeramente mayor.
- **Asimetría de costos:** Los tokens de salida cuestan entre 2× y 5× más que los de entrada; controlar `max_tokens` es una palanca directa de ahorro.
- **Temperatura como dial:** No existe una temperatura "correcta" universal — la elección depende del tipo de tarea (analítica → baja; creativa → alta).
- **Embeddings como coordenadas:** Vectores cercanos en el espacio semántico representan conceptos relacionados, base fundamental del sistema RAG que se construirá en el Lab 4.

### Recursos adicionales

- [Tokenizador interactivo de OpenAI (platform.openai.com/tokenizer)](https://platform.openai.com/tokenizer)
- [Documentación de `tiktoken`](https://github.com/openai/tiktoken)
- [Guía de embeddings de OpenAI](https://platform.openai.com/docs/guides/embeddings)
- [Comparativa de modelos y precios](https://openai.com/api/pricing)
- [Liu et al. (2023) — "Lost in the Middle"](https://arxiv.org/abs/2307.03172)

---
