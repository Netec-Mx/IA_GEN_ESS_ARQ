# Pipeline de Ingestión y Creación de Índice Vectorial

## 1. Metadatos

| Campo | Valor |
|---|---|
| **Duración estimada** | 72 minutos |
| **Complejidad** | Alta |
| **Nivel Bloom** | Aplicar (*Apply*) |
| **Costo estimado API** | ~$1.50 USD (GPT-4o-mini + text-embedding-3-small) |
| **Laboratorio anterior requerido** | Lab 01-00-01, Lab 02-00-01, Lab 03-00-01 |

---

## 2. Descripción General

En este laboratorio construirás un sistema RAG (*Retrieval-Augmented Generation*) completo de extremo a extremo, aplicando los conceptos de la Lección 4.1. Partirás de un corpus de documentos empresariales en tres formatos distintos (PDF, DOCX, HTML), los ingerirás y limpiarás, aplicarás y compararás tres estrategias de chunking, construirás un índice vectorial FAISS con embeddings de OpenAI y, finalmente, medirás la calidad del sistema con métricas RAGAS e implementarás estrategias de mitigación de riesgos. Al finalizar dispondrás de un endpoint de consulta funcional que responde preguntas con citas de fuente verificables.

> ⚠️ **Aviso de privacidad**: Este laboratorio envía texto a la API de OpenAI. **No utilices documentos que contengan datos personales, información confidencial de tu empresa o cualquier dato sensible.** Usa únicamente el corpus de muestra proporcionado por el curso. Consulta la [política de datos de OpenAI](https://openai.com/policies/api-data-usage-policies) para más información.

---

## 3. Objetivos de Aprendizaje

Al completar este laboratorio serás capaz de:

- [ ] Implementar un pipeline de ingestión multi-formato (PDF, DOCX, HTML) con extracción de texto, limpieza y enriquecimiento de metadatos usando LangChain Document Loaders
- [ ] Aplicar y comparar tres estrategias de chunking evaluando su impacto en número de chunks, tamaño promedio y coherencia semántica
- [ ] Construir un índice vectorial FAISS con embeddings `text-embedding-3-small`, implementar búsqueda por similitud coseno y re-ranking básico
- [ ] Integrar el índice con GPT-4o-mini para respuestas RAG con citas de fuente y medir la calidad con métricas RAGAS (faithfulness, answer relevancy)
- [ ] Implementar detección de duplicados, versionado de documentos y alertas para fuentes contradictorias como estrategias de mitigación de riesgos

---

## 4. Prerrequisitos

### Conocimiento previo
- Haber completado los Labs 01-00-01, 02-00-01 y 03-00-01
- Python intermedio: clases básicas, manejo de archivos, comprensión de listas, manejo de excepciones
- Comprensión del flujo RAG: base de conocimiento → retriever → generador (Lección 4.1)
- Familiaridad con embeddings y búsqueda vectorial (Lab 02-00-01)

### Acceso y recursos
- API key de OpenAI con créditos disponibles (~$1.50 USD para este lab)
- Corpus de documentos de muestra descargado del repositorio del curso (carpeta `corpus/`)
- Mínimo 2 GB de espacio libre en disco
- Conexión a Internet estable (mínimo 10 Mbps)

---

## 5. Entorno de Laboratorio

### Hardware recomendado

| Recurso | Mínimo | Recomendado |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 núcleos (i5/Ryzen 5 8ª gen) | 8 núcleos |
| Almacenamiento libre | 2 GB | 5 GB |
| Conexión | 10 Mbps | 25 Mbps |

### Software requerido

| Paquete | Versión mínima | Uso en el lab |
|---|---|---|
| Python | 3.10 | Entorno base |
| langchain | 0.2 | Loaders, splitters, chains |
| langchain-openai | 0.1 | Embeddings y LLM |
| langchain-community | 0.2 | FAISS VectorStore |
| faiss-cpu | 1.8 | Índice vectorial |
| pymupdf | 1.24 | Ingestión PDF |
| python-docx | 1.1 | Ingestión DOCX |
| beautifulsoup4 | 4.12 | Ingestión HTML |
| ragas | 0.1 | Métricas de calidad |
| pandas | 2.1 | Análisis de métricas |
| numpy | 1.26 | Operaciones vectoriales |
| tiktoken | 0.7 | Conteo de tokens |
| python-dotenv | 1.0 | Gestión de API key |
| matplotlib | 3.8 | Visualizaciones |

### Comandos de configuración del entorno

```bash
# Crear y activar entorno virtual
python -m venv venv_lab4
source venv_lab4/bin/activate          # Linux/macOS
# venv_lab4\Scripts\activate           # Windows

# Instalar dependencias
pip install langchain==0.2.* langchain-openai langchain-community \
            faiss-cpu pymupdf python-docx beautifulsoup4 \
            ragas pandas numpy tiktoken python-dotenv \
            matplotlib seaborn

# Verificar instalación
python -c "import langchain, faiss, fitz, docx, bs4, ragas; print('OK')"
```

### Estructura de directorios del proyecto

```
lab4_rag/
├── corpus/                  # Documentos fuente (descargar del repositorio)
│   ├── pdf/                 # 5-7 documentos PDF
│   ├── docx/                # 5-7 documentos DOCX
│   └── html/                # 5-7 documentos HTML
├── indices/                 # Índice FAISS persistido (se genera en el lab)
├── outputs/                 # Resultados, métricas, reportes
├── .env                     # API key (NO subir a git)
└── lab4_pipeline.ipynb      # Notebook principal
```

```bash
# Crear estructura de directorios
mkdir -p lab4_rag/{corpus/{pdf,docx,html},indices,outputs}
cd lab4_rag

# Crear archivo .env
echo "OPENAI_API_KEY=sk-..." > .env
# Reemplaza sk-... con tu API key real

# Descargar corpus del repositorio del curso
# (el instructor proveerá la URL o los archivos directamente)
# cp -r /ruta/corpus/* corpus/
```

> 💡 **Nota para Google Colab**: Monta tu Google Drive antes de comenzar para garantizar la persistencia del índice FAISS entre sesiones:
> ```python
> from google.colab import drive
> drive.mount('/content/drive')
> ```

---

## 6. Pasos del Laboratorio

### Fase 1 — Ingestión Multi-formato y Enriquecimiento de Metadatos

**Tiempo estimado: 15 minutos**

---

#### Paso 1.1: Configuración inicial y carga de dependencias

**Objetivo**: Inicializar el entorno, cargar credenciales y definir la estructura base del pipeline.

**Instrucciones**:

1. Crea el notebook `lab4_pipeline.ipynb` y ejecuta la siguiente celda de configuración:

```python
# Celda 1: Importaciones y configuración
import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv

# LangChain
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    HTMLHeaderTextSplitter
)
from langchain.prompts import PromptTemplate

# Parsers de documentos
import fitz          # PyMuPDF
from docx import Document as DocxDocument
from bs4 import BeautifulSoup

# Métricas
import tiktoken

# Cargar variables de entorno
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "❌ API key no encontrada. Verifica tu archivo .env"

# Directorios
CORPUS_DIR = Path("corpus")
INDEX_DIR  = Path("indices")
OUTPUT_DIR = Path("outputs")
for d in [INDEX_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

print("✅ Entorno configurado correctamente")
print(f"   Corpus: {CORPUS_DIR}")
print(f"   Índice: {INDEX_DIR}")
```

**Salida esperada**:
```
✅ Entorno configurado correctamente
   Corpus: corpus
   Índice: indices
```

**Verificación**: Confirma que no aparece ningún `AssertionError` ni `ImportError`.

---

#### Paso 1.2: Implementar loaders multi-formato con metadatos enriquecidos

**Objetivo**: Crear funciones de carga para PDF, DOCX y HTML que extraigan texto limpio y metadatos estructurados.

**Instrucciones**:

1. Implementa la clase `DocumentLoader` con los tres métodos de carga:

```python
# Celda 2: Clase DocumentLoader
class DocumentLoader:
    """Pipeline de ingestión multi-formato con enriquecimiento de metadatos."""

    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")

    # ── Utilidades ──────────────────────────────────────────────────────────

    def _limpiar_texto(self, texto: str) -> str:
        """Elimina ruido: múltiples espacios, saltos de línea excesivos,
        caracteres de control y normaliza la puntuación."""
        texto = re.sub(r'\x00', '', texto)                  # nulos
        texto = re.sub(r'[ \t]+', ' ', texto)               # espacios múltiples
        texto = re.sub(r'\n{3,}', '\n\n', texto)            # saltos excesivos
        texto = re.sub(r'[^\S\n]+\n', '\n', texto)          # espacios al final de línea
        return texto.strip()

    def _contar_tokens(self, texto: str) -> int:
        return len(self.enc.encode(texto))

    def _generar_id(self, ruta: str, contenido: str) -> str:
        """Hash SHA-256 para identificar unívocamente cada documento."""
        return hashlib.sha256(f"{ruta}{contenido[:200]}".encode()).hexdigest()[:16]

    # ── Loader PDF ──────────────────────────────────────────────────────────

    def cargar_pdf(self, ruta: Path) -> List[Document]:
        """Extrae texto por página usando PyMuPDF, filtra páginas vacías."""
        documentos = []
        with fitz.open(str(ruta)) as pdf:
            for num_pag, pagina in enumerate(pdf):
                texto = pagina.get_text("text")
                texto = self._limpiar_texto(texto)
                if len(texto) < 50:          # omitir páginas casi vacías
                    continue
                doc = Document(
                    page_content=texto,
                    metadata={
                        "fuente":    ruta.name,
                        "ruta":      str(ruta),
                        "formato":   "pdf",
                        "pagina":    num_pag + 1,
                        "total_pag": len(pdf),
                        "tokens":    self._contar_tokens(texto),
                        "fecha_ing": datetime.now().isoformat(),
                        "doc_id":    self._generar_id(str(ruta), texto),
                    }
                )
                documentos.append(doc)
        return documentos

    # ── Loader DOCX ─────────────────────────────────────────────────────────

    def cargar_docx(self, ruta: Path) -> List[Document]:
        """Extrae párrafos y detecta secciones por estilo de encabezado."""
        docx = DocxDocument(str(ruta))
        seccion_actual = "Introducción"
        parrafos_seccion = []
        documentos = []

        for parrafo in docx.paragraphs:
            texto = parrafo.text.strip()
            if not texto:
                continue
            # Detectar encabezados (Heading 1, 2, 3)
            if parrafo.style.name.startswith("Heading"):
                if parrafos_seccion:
                    contenido = self._limpiar_texto("\n".join(parrafos_seccion))
                    documentos.append(Document(
                        page_content=contenido,
                        metadata={
                            "fuente":    ruta.name,
                            "ruta":      str(ruta),
                            "formato":   "docx",
                            "seccion":   seccion_actual,
                            "tokens":    self._contar_tokens(contenido),
                            "fecha_ing": datetime.now().isoformat(),
                            "doc_id":    self._generar_id(str(ruta), contenido),
                        }
                    ))
                    parrafos_seccion = []
                seccion_actual = texto
            else:
                parrafos_seccion.append(texto)

        # Último bloque
        if parrafos_seccion:
            contenido = self._limpiar_texto("\n".join(parrafos_seccion))
            documentos.append(Document(
                page_content=contenido,
                metadata={
                    "fuente":    ruta.name,
                    "ruta":      str(ruta),
                    "formato":   "docx",
                    "seccion":   seccion_actual,
                    "tokens":    self._contar_tokens(contenido),
                    "fecha_ing": datetime.now().isoformat(),
                    "doc_id":    self._generar_id(str(ruta), contenido),
                }
            ))
        return documentos

    # ── Loader HTML ─────────────────────────────────────────────────────────

    def cargar_html(self, ruta: Path) -> List[Document]:
        """Extrae texto semántico usando BeautifulSoup, preserva estructura."""
        with open(ruta, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Eliminar scripts, estilos y navegación
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        titulo = soup.find("title")
        titulo_txt = titulo.get_text(strip=True) if titulo else ruta.stem

        texto = soup.get_text(separator="\n")
        texto = self._limpiar_texto(texto)

        return [Document(
            page_content=texto,
            metadata={
                "fuente":    ruta.name,
                "ruta":      str(ruta),
                "formato":   "html",
                "titulo":    titulo_txt,
                "tokens":    self._contar_tokens(texto),
                "fecha_ing": datetime.now().isoformat(),
                "doc_id":    self._generar_id(str(ruta), texto),
            }
        )]

    # ── Método principal ─────────────────────────────────────────────────────

    def cargar_corpus(self, directorio: Path) -> List[Document]:
        """Carga todos los documentos del corpus detectando el formato."""
        todos = []
        extensiones = {".pdf": self.cargar_pdf,
                       ".docx": self.cargar_docx,
                       ".html": self.cargar_html,
                       ".htm":  self.cargar_html}
        for ruta in sorted(directorio.rglob("*")):
            if ruta.suffix.lower() in extensiones:
                try:
                    docs = extensiones[ruta.suffix.lower()](ruta)
                    todos.extend(docs)
                    print(f"  ✅ {ruta.name}: {len(docs)} fragmento(s)")
                except Exception as e:
                    print(f"  ⚠️  {ruta.name}: error — {e}")
        return todos
```

2. Ejecuta la carga del corpus:

```python
# Celda 3: Cargar corpus
loader = DocumentLoader()
print("📂 Cargando corpus...\n")
documentos_raw = loader.cargar_corpus(CORPUS_DIR)

print(f"\n📊 Resumen de ingestión:")
print(f"   Total documentos cargados: {len(documentos_raw)}")

# Estadísticas por formato
df_raw = pd.DataFrame([d.metadata for d in documentos_raw])
if not df_raw.empty and "formato" in df_raw.columns:
    print(df_raw.groupby("formato")["tokens"].agg(["count","sum","mean"]).round(1))
```

**Salida esperada**:
```
📂 Cargando corpus...

  ✅ politica_vacaciones.pdf: 4 fragmento(s)
  ✅ manual_onboarding.pdf: 7 fragmento(s)
  ✅ faq_ti.docx: 3 fragmento(s)
  ...

📊 Resumen de ingestión:
   Total documentos cargados: 38
         count      sum    mean
formato
docx        12   8450.0   704.2
html         8   6230.0   778.8
pdf         18  14320.0   795.6
```

**Verificación**: `len(documentos_raw) > 0` y el DataFrame muestra las tres columnas de formato.

---

### Fase 2 — Estrategias de Chunking y Análisis Comparativo

**Tiempo estimado: 15 minutos**

---

#### Paso 2.1: Implementar las tres estrategias de chunking

**Objetivo**: Aplicar `RecursiveCharacterTextSplitter`, `HTMLHeaderTextSplitter` y chunking por sección, comparando sus resultados.

**Instrucciones**:

1. Implementa las tres estrategias:

```python
# Celda 4: Estrategias de chunking

# ── Estrategia A: RecursiveCharacterTextSplitter (tamaño fijo con overlap) ──
splitter_a = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks_a = splitter_a.split_documents(documentos_raw)
# Preservar metadatos de estrategia
for c in chunks_a:
    c.metadata["estrategia"] = "A_recursive"

# ── Estrategia B: HTMLHeaderTextSplitter (semántico por encabezados HTML) ──
# Aplicar solo a documentos HTML; para otros formatos usar splitter_a
headers_to_split = [("h1", "h1"), ("h2", "h2"), ("h3", "h3")]
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split)

chunks_b = []
for doc in documentos_raw:
    if doc.metadata.get("formato") == "html":
        ruta = doc.metadata.get("ruta", "")
        try:
            with open(ruta, "r", encoding="utf-8", errors="replace") as f:
                html_content = f.read()
            splits = html_splitter.split_text(html_content)
            for s in splits:
                s.metadata.update(doc.metadata)
                s.metadata["estrategia"] = "B_html_header"
            chunks_b.extend(splits)
        except Exception:
            # fallback para HTML no accesible directamente
            fallback = splitter_a.split_documents([doc])
            for c in fallback:
                c.metadata["estrategia"] = "B_html_header"
            chunks_b.extend(fallback)
    else:
        fallback = splitter_a.split_documents([doc])
        for c in fallback:
            c.metadata["estrategia"] = "B_html_header"
        chunks_b.extend(fallback)

# ── Estrategia C: Chunking por sección de documento (DOCX nativo) ──
# Los documentos DOCX ya vienen divididos por sección desde el loader
chunks_c = []
splitter_c = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=50,
)
for doc in documentos_raw:
    if doc.metadata.get("formato") == "docx":
        # Cada sección DOCX es un chunk natural
        doc.metadata["estrategia"] = "C_seccion"
        chunks_c.append(doc)
    else:
        splits = splitter_c.split_documents([doc])
        for c in splits:
            c.metadata["estrategia"] = "C_seccion"
        chunks_c.extend(splits)

print(f"Estrategia A (RecursiveChar 500/100):  {len(chunks_a):>4} chunks")
print(f"Estrategia B (HTMLHeader semántico):   {len(chunks_b):>4} chunks")
print(f"Estrategia C (Sección documento):      {len(chunks_c):>4} chunks")
```

2. Genera el análisis comparativo:

```python
# Celda 5: Análisis comparativo de estrategias

def analizar_chunks(chunks: List[Document], nombre: str) -> Dict:
    longitudes = [len(c.page_content) for c in chunks]
    tokens_lst = [loader._contar_tokens(c.page_content) for c in chunks]
    return {
        "estrategia":       nombre,
        "num_chunks":       len(chunks),
        "long_promedio":    np.mean(longitudes).round(1),
        "long_mediana":     np.median(longitudes).round(1),
        "long_std":         np.std(longitudes).round(1),
        "tokens_promedio":  np.mean(tokens_lst).round(1),
        "chunks_vacios":    sum(1 for l in longitudes if l < 20),
    }

resultados = [
    analizar_chunks(chunks_a, "A_recursive"),
    analizar_chunks(chunks_b, "B_html_header"),
    analizar_chunks(chunks_c, "C_seccion"),
]
df_comparativa = pd.DataFrame(resultados)
print(df_comparativa.to_string(index=False))

# Guardar resultados
df_comparativa.to_csv(OUTPUT_DIR / "comparativa_chunking.csv", index=False)

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(df_comparativa["estrategia"], df_comparativa["num_chunks"],
            color=["#4C72B0", "#DD8452", "#55A868"])
axes[0].set_title("Número de chunks por estrategia")
axes[0].set_ylabel("Chunks")

axes[1].bar(df_comparativa["estrategia"], df_comparativa["tokens_promedio"],
            color=["#4C72B0", "#DD8452", "#55A868"])
axes[1].set_title("Tokens promedio por chunk")
axes[1].set_ylabel("Tokens")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparativa_chunking.png", dpi=120)
plt.show()
print("📊 Gráfico guardado en outputs/comparativa_chunking.png")
```

**Salida esperada**:
```
  estrategia  num_chunks  long_promedio  long_mediana  long_std  tokens_promedio  chunks_vacios
A_recursive         142          412.3         450.0     102.4            103.1              0
B_html_header        98          487.6         510.0      89.2            121.9              2
C_seccion           115          534.8         480.0     198.7            133.7              0
```

**Verificación**: Los tres valores de `num_chunks` son mayores que el número de documentos raw. `chunks_vacios` debe ser 0 o muy bajo.

> 📝 **Reflexión**: Anota en una celda Markdown qué estrategia produce chunks más uniformes (menor `long_std`) y por qué eso puede ser ventajoso para la recuperación.

---

### Fase 3 — Construcción del Índice Vectorial FAISS

**Tiempo estimado: 18 minutos**

---

#### Paso 3.1: Generar embeddings y construir el índice FAISS

**Objetivo**: Usar `text-embedding-3-small` para vectorizar los chunks de la estrategia seleccionada y construir el índice FAISS.

**Instrucciones**:

1. Selecciona la estrategia de chunking para el índice (se recomienda la estrategia A como baseline) y genera el índice:

```python
# Celda 6: Generación de embeddings e índice FAISS

# Usar estrategia A como índice principal
CHUNKS_INDEXAR = chunks_a
print(f"📌 Indexando {len(CHUNKS_INDEXAR)} chunks con text-embedding-3-small...")
print("   (Esto puede tardar 2-4 minutos según el tamaño del corpus)\n")

# Inicializar modelo de embeddings
modelo_emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

# Construir índice FAISS
# FAISS.from_documents genera embeddings en lotes y construye el índice
indice_faiss = FAISS.from_documents(CHUNKS_INDEXAR, modelo_emb)

# Persistir en disco
indice_faiss.save_local(str(INDEX_DIR / "indice_principal"))
print(f"✅ Índice FAISS guardado en: {INDEX_DIR / 'indice_principal'}")
print(f"   Vectores indexados: {indice_faiss.index.ntotal}")
print(f"   Dimensión de embeddings: {indice_faiss.index.d}")
```

**Salida esperada**:
```
📌 Indexando 142 chunks con text-embedding-3-small...
   (Esto puede tardar 2-4 minutos según el tamaño del corpus)

✅ Índice FAISS guardado en: indices/indice_principal
   Vectores indexados: 142
   Dimensión de embeddings: 1536
```

---

#### Paso 3.2: Implementar búsqueda por similitud y re-ranking

**Objetivo**: Probar la recuperación con 10 queries de prueba e implementar re-ranking básico por relevancia y metadatos.

**Instrucciones**:

1. Implementa la función de búsqueda con re-ranking:

```python
# Celda 7: Búsqueda con re-ranking

def buscar_con_reranking(
    indice: FAISS,
    query: str,
    k_inicial: int = 10,
    k_final: int = 4,
    boost_formatos: Optional[List[str]] = None,
    penalizar_duplicados: bool = True
) -> List[Tuple[Document, float]]:
    """
    Recupera k_inicial chunks, aplica re-ranking y devuelve k_final.

    Re-ranking considera:
    1. Puntuación de similitud coseno (score FAISS)
    2. Boost por formato preferido
    3. Penalización por contenido casi duplicado
    """
    # Recuperar con scores
    resultados = indice.similarity_search_with_score(query, k=k_inicial)

    # FAISS devuelve distancia L2; convertir a similitud (menor = más similar)
    max_dist = max(score for _, score in resultados) + 1e-9
    resultados_norm = [(doc, 1 - score / max_dist) for doc, score in resultados]

    # Boost por formato
    if boost_formatos:
        resultados_norm = [
            (doc, score * 1.15 if doc.metadata.get("formato") in boost_formatos else score)
            for doc, score in resultados_norm
        ]

    # Penalización por duplicados (similitud de texto > 85%)
    if penalizar_duplicados:
        vistos = []
        filtrados = []
        for doc, score in resultados_norm:
            texto_norm = re.sub(r'\s+', ' ', doc.page_content[:200]).lower()
            es_dup = any(
                sum(a == b for a, b in zip(texto_norm, v)) / max(len(texto_norm), len(v)) > 0.85
                for v in vistos
            )
            if not es_dup:
                filtrados.append((doc, score))
                vistos.append(texto_norm)
        resultados_norm = filtrados

    # Ordenar por score final y tomar k_final
    resultados_norm.sort(key=lambda x: x[1], reverse=True)
    return resultados_norm[:k_final]


# ── Prueba con 10 queries ─────────────────────────────────────────────────
queries_prueba = [
    "¿Cuántos días de vacaciones tiene un empleado?",
    "¿Cómo se solicita un reembolso de gastos?",
    "¿Cuál es el proceso de incorporación para nuevos empleados?",
    "¿Qué herramientas de desarrollo usa el equipo de ingeniería?",
    "¿Cuál es la política de trabajo remoto?",
    "¿Cómo reportar un incidente de seguridad?",
    "¿Qué beneficios de salud están disponibles?",
    "¿Cuál es el proceso de evaluación de desempeño?",
    "¿Cómo se gestiona el acceso a sistemas internos?",
    "¿Qué capacitaciones son obligatorias para el personal?",
]

print("🔍 Prueba de recuperación con 10 queries:\n")
resultados_queries = []

for i, query in enumerate(queries_prueba, 1):
    resultados = buscar_con_reranking(indice_faiss, query, k_inicial=8, k_final=3)
    top_fuente = resultados[0][0].metadata.get("fuente", "N/A") if resultados else "Sin resultados"
    top_score  = resultados[0][1] if resultados else 0.0
    print(f"  Q{i:02d}: {query[:55]:<55} → {top_fuente} (score: {top_score:.3f})")
    resultados_queries.append({
        "query": query,
        "top_fuente": top_fuente,
        "top_score": top_score,
        "num_resultados": len(resultados)
    })

df_queries = pd.DataFrame(resultados_queries)
df_queries.to_csv(OUTPUT_DIR / "resultados_queries.csv", index=False)
print(f"\n✅ Resultados guardados en outputs/resultados_queries.csv")
```

**Salida esperada**:
```
🔍 Prueba de recuperación con 10 queries:

  Q01: ¿Cuántos días de vacaciones tiene un empleado?       → politica_vacaciones.pdf (score: 0.847)
  Q02: ¿Cómo se solicita un reembolso de gastos?           → politica_gastos.docx (score: 0.821)
  ...
✅ Resultados guardados en outputs/resultados_queries.csv
```

**Verificación**: Todos los scores deben estar en el rango `[0, 1]`. Si algún score es 0.0 para todas las queries, verifica que el índice fue construido correctamente.

---

### Fase 4 — Integración RAG con GPT-4o-mini y Evaluación de Calidad

**Tiempo estimado: 18 minutos**

---

#### Paso 4.1: Construir el endpoint RAG con citas de fuente

**Objetivo**: Integrar el índice FAISS con GPT-4o-mini usando una cadena `RetrievalQA` con prompt personalizado que incluya instrucciones de citación.

**Instrucciones**:

1. Define el prompt RAG y construye la cadena:

```python
# Celda 8: Cadena RAG con GPT-4o-mini

# Prompt personalizado que instruye al modelo a citar fuentes
PROMPT_RAG = PromptTemplate(
    input_variables=["context", "question"],
    template="""Eres un asistente corporativo experto. Responde la pregunta
basándote ÚNICAMENTE en el contexto proporcionado. Si la información no está
en el contexto, responde: "No tengo información suficiente sobre este tema
en los documentos disponibles."

Al final de tu respuesta, incluye siempre una sección "Fuentes:" con los
nombres de los documentos que usaste.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
)

# LLM generador
llm_rag = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# Retriever con re-ranking integrado (k=4 chunks)
retriever = indice_faiss.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Cadena RAG
cadena_rag = RetrievalQA.from_chain_type(
    llm=llm_rag,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT_RAG}
)

# ── Función de consulta con formato de salida ────────────────────────────
def consultar_rag(pregunta: str, verbose: bool = True) -> Dict:
    resultado = cadena_rag.invoke({"query": pregunta})
    fuentes = list(set(
        doc.metadata.get("fuente", "desconocida")
        for doc in resultado["source_documents"]
    ))
    if verbose:
        print(f"❓ Pregunta: {pregunta}\n")
        print(f"💬 Respuesta:\n{resultado['result']}\n")
        print(f"📎 Documentos recuperados: {', '.join(fuentes)}\n")
        print("─" * 70)
    return {
        "pregunta": pregunta,
        "respuesta": resultado["result"],
        "fuentes": fuentes,
        "docs_recuperados": resultado["source_documents"]
    }

# Prueba rápida
print("🚀 Probando endpoint RAG...\n")
_ = consultar_rag("¿Cuántos días de vacaciones tiene un empleado nuevo?")
```

**Salida esperada**:
```
🚀 Probando endpoint RAG...

❓ Pregunta: ¿Cuántos días de vacaciones tiene un empleado nuevo?

💬 Respuesta:
Según la política de recursos humanos, los empleados nuevos tienen derecho
a 20 días hábiles de vacaciones al año, los cuales se acumulan
proporcionalmente durante el primer año...

Fuentes: politica_vacaciones.pdf

📎 Documentos recuperados: politica_vacaciones.pdf
──────────────────────────────────────────────────────────────────────────
```

---

#### Paso 4.2: Medir calidad con RAGAS

**Objetivo**: Evaluar el sistema RAG con 20 preguntas usando métricas `faithfulness` y `answer_relevancy` de RAGAS.

**Instrucciones**:

1. Prepara el dataset de evaluación y ejecuta RAGAS:

```python
# Celda 9: Evaluación con RAGAS

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# Dataset de 20 preguntas de evaluación
# En un proyecto real estas vendrían de expertos del dominio
preguntas_eval = [
    "¿Cuántos días de vacaciones tiene un empleado?",
    "¿Cómo se solicita un reembolso de gastos de viaje?",
    "¿Cuál es la política de trabajo remoto de la empresa?",
    "¿Qué herramientas usa el equipo de desarrollo?",
    "¿Cómo se reporta un incidente de seguridad informática?",
    "¿Cuáles son los beneficios médicos disponibles?",
    "¿Cuál es el proceso de evaluación de desempeño anual?",
    "¿Cómo se gestiona el acceso a sistemas internos?",
    "¿Qué capacitaciones son obligatorias para empleados nuevos?",
    "¿Cuál es el proceso de incorporación (onboarding)?",
    "¿Cómo se solicita un permiso de ausencia?",
    "¿Cuál es la política de uso aceptable de equipos?",
    "¿Cómo se escala un problema técnico al equipo de TI?",
    "¿Qué documentos se necesitan para el proceso de contratación?",
    "¿Cuál es la política de confidencialidad de la empresa?",
    "¿Cómo se solicita una capacitación externa?",
    "¿Cuál es el horario laboral estándar?",
    "¿Cómo se gestiona el inventario de equipos de cómputo?",
    "¿Cuál es el procedimiento para cambio de contraseñas?",
    "¿Cómo se realiza la gestión de proveedores externos?",
]

print("⏳ Generando respuestas para el set de evaluación (20 preguntas)...")
print("   Esto puede tardar 3-5 minutos.\n")

# Recopilar respuestas y contextos
data_eval = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for pregunta in preguntas_eval:
    resultado = cadena_rag.invoke({"query": pregunta})
    contextos = [doc.page_content for doc in resultado["source_documents"]]
    data_eval["question"].append(pregunta)
    data_eval["answer"].append(resultado["result"])
    data_eval["contexts"].append(contextos)
    # ground_truth vacío (evaluación sin anotaciones de referencia)
    data_eval["ground_truth"].append("")

dataset_eval = Dataset.from_dict(data_eval)

# Ejecutar evaluación RAGAS
print("📊 Calculando métricas RAGAS...")
metricas_ragas = evaluate(
    dataset_eval,
    metrics=[faithfulness, answer_relevancy],
    llm=llm_rag,
    embeddings=modelo_emb
)

# Mostrar resultados
df_ragas = metricas_ragas.to_pandas()
print("\n✅ Resultados RAGAS:")
print(f"   Faithfulness (fidelidad):     {df_ragas['faithfulness'].mean():.3f}")
print(f"   Answer Relevancy (relevancia): {df_ragas['answer_relevancy'].mean():.3f}")

df_ragas.to_csv(OUTPUT_DIR / "metricas_ragas.csv", index=False)
print(f"\n💾 Métricas guardadas en outputs/metricas_ragas.csv")
```

**Salida esperada**:
```
⏳ Generando respuestas para el set de evaluación (20 preguntas)...

📊 Calculando métricas RAGAS...

✅ Resultados RAGAS:
   Faithfulness (fidelidad):     0.847
   Answer Relevancy (relevancia): 0.812

💾 Métricas guardadas en outputs/metricas_ragas.csv
```

> 📝 **Interpretación**: Un `faithfulness` > 0.80 indica que el modelo responde basándose en el contexto recuperado (baja alucinación). Un `answer_relevancy` > 0.75 indica que las respuestas son pertinentes a las preguntas. Si obtienes valores menores, revisa la calidad del corpus y la estrategia de chunking.

---

#### Paso 4.3: Implementar mitigaciones de riesgo

**Objetivo**: Añadir detección de duplicados por similitud de embeddings, versionado de documentos y alertas para fuentes contradictorias.

**Instrucciones**:

1. Implementa el módulo de mitigación de riesgos:

```python
# Celda 10: Módulo de mitigación de riesgos

class GestorRiesgosRAG:
    """
    Implementa tres estrategias de mitigación de riesgos para el sistema RAG:
    1. Detección de chunks duplicados por similitud de embeddings
    2. Versionado de documentos (control de documentos obsoletos)
    3. Alertas para fuentes potencialmente contradictorias
    """

    def __init__(self, umbral_duplicado: float = 0.95,
                       umbral_contradiccion: float = 0.30):
        self.umbral_dup   = umbral_duplicado
        self.umbral_cont  = umbral_contradiccion
        self.registro_versiones: Dict[str, Dict] = {}
        self.alertas: List[Dict] = []

    # ── 1. Detección de duplicados ────────────────────────────────────────

    def detectar_duplicados(
        self,
        chunks: List[Document],
        embeddings_modelo: OpenAIEmbeddings
    ) -> Tuple[List[Document], List[Dict]]:
        """
        Calcula embeddings de todos los chunks y detecta pares con
        similitud coseno > umbral_duplicado.
        Devuelve lista filtrada y reporte de duplicados.
        """
        print("🔍 Detectando duplicados por similitud de embeddings...")
        textos = [c.page_content for c in chunks]
        vecs = np.array(embeddings_modelo.embed_documents(textos))

        # Normalizar para similitud coseno
        normas = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_norm = vecs / (normas + 1e-9)
        sim_matrix = vecs_norm @ vecs_norm.T

        duplicados_encontrados = []
        indices_eliminar = set()

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                if sim_matrix[i, j] >= self.umbral_dup:
                    duplicados_encontrados.append({
                        "chunk_a": chunks[i].metadata.get("fuente", "?"),
                        "chunk_b": chunks[j].metadata.get("fuente", "?"),
                        "similitud": float(sim_matrix[i, j]),
                        "texto_a": chunks[i].page_content[:100],
                        "texto_b": chunks[j].page_content[:100],
                    })
                    indices_eliminar.add(j)  # eliminar el segundo

        chunks_limpios = [c for idx, c in enumerate(chunks)
                          if idx not in indices_eliminar]
        print(f"   Duplicados detectados: {len(duplicados_encontrados)}")
        print(f"   Chunks antes: {len(chunks)} → después: {len(chunks_limpios)}")
        return chunks_limpios, duplicados_encontrados

    # ── 2. Versionado de documentos ───────────────────────────────────────

    def registrar_version(self, nombre_doc: str, version: str,
                          fecha: str, activo: bool = True) -> None:
        """Registra la versión de un documento en el índice."""
        self.registro_versiones[nombre_doc] = {
            "version": version,
            "fecha":   fecha,
            "activo":  activo,
            "registrado": datetime.now().isoformat()
        }

    def verificar_vigencia(self, fuente: str) -> Dict:
        """Verifica si un documento está vigente según el registro."""
        if fuente not in self.registro_versiones:
            return {"vigente": None, "mensaje": "Documento sin registro de versión"}
        reg = self.registro_versiones[fuente]
        if not reg["activo"]:
            return {"vigente": False,
                    "mensaje": f"⚠️ Documento obsoleto: {fuente} (v{reg['version']}, {reg['fecha']})"}
        return {"vigente": True,
                "mensaje": f"✅ Documento vigente: {fuente} (v{reg['version']})"}

    # ── 3. Alertas para fuentes contradictorias ───────────────────────────

    def detectar_contradicciones(
        self,
        docs_recuperados: List[Document],
        embeddings_modelo: OpenAIEmbeddings
    ) -> List[Dict]:
        """
        Detecta posibles contradicciones entre chunks recuperados.
        Heurística: dos chunks de DISTINTAS fuentes con similitud coseno
        BAJA (< umbral_contradiccion) sobre el mismo tema pueden ser
        contradictorios. Requiere validación humana.
        """
        if len(docs_recuperados) < 2:
            return []

        textos = [d.page_content for d in docs_recuperados]
        vecs = np.array(embeddings_modelo.embed_documents(textos))
        normas = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_norm = vecs / (normas + 1e-9)
        sim_matrix = vecs_norm @ vecs_norm.T

        alertas = []
        for i in range(len(docs_recuperados)):
            for j in range(i + 1, len(docs_recuperados)):
                fuente_i = docs_recuperados[i].metadata.get("fuente", "?")
                fuente_j = docs_recuperados[j].metadata.get("fuente", "?")
                if fuente_i != fuente_j and sim_matrix[i, j] < self.umbral_cont:
                    alertas.append({
                        "tipo":      "posible_contradiccion",
                        "fuente_a":  fuente_i,
                        "fuente_b":  fuente_j,
                        "similitud": float(sim_matrix[i, j]),
                        "mensaje":   f"⚠️ Posible contradicción entre '{fuente_i}' y '{fuente_j}'. Revisar manualmente.",
                    })
        return alertas


# ── Demostración del módulo ───────────────────────────────────────────────
gestor = GestorRiesgosRAG(umbral_duplicado=0.95, umbral_contradiccion=0.30)

# Registrar versiones de ejemplo
gestor.registrar_version("politica_vacaciones.pdf",  "2.1", "2024-01-15", activo=True)
gestor.registrar_version("politica_vacaciones_v1.pdf","1.0","2022-06-01", activo=False)
gestor.registrar_version("manual_onboarding.pdf",    "3.0", "2024-03-20", activo=True)

# Verificar vigencia
for doc_nombre in ["politica_vacaciones.pdf", "politica_vacaciones_v1.pdf",
                   "manual_onboarding.pdf"]:
    info = gestor.verificar_vigencia(doc_nombre)
    print(info["mensaje"])

print()

# Detectar contradicciones en una consulta de ejemplo
consulta_test = "¿Cuántos días de vacaciones corresponden?"
res_test = cadena_rag.invoke({"query": consulta_test})
alertas_cont = gestor.detectar_contradicciones(
    res_test["source_documents"], modelo_emb
)
if alertas_cont:
    print("🚨 Alertas de contradicción:")
    for alerta in alertas_cont:
        print(f"   {alerta['mensaje']}")
else:
    print("✅ No se detectaron contradicciones en los documentos recuperados.")
```

**Salida esperada**:
```
✅ Documento vigente: politica_vacaciones.pdf (v2.1)
⚠️ Documento obsoleto: politica_vacaciones_v1.pdf (v1.0, 2022-06-01)
✅ Documento vigente: manual_onboarding.pdf (v3.0)

✅ No se detectaron contradicciones en los documentos recuperados.
```

---

### Fase Final — Detección de Duplicados y Reporte Consolidado

**Tiempo estimado: 6 minutos**

---

#### Paso 4.4: Detección de duplicados y reporte final

**Objetivo**: Ejecutar la detección de duplicados sobre el corpus completo y generar el reporte consolidado del sistema.

**Instrucciones**:

1. Ejecuta la detección de duplicados (nota: requiere llamadas a la API para generar embeddings de todos los chunks):

```python
# Celda 11: Detección de duplicados y reporte final

print("🔎 Ejecutando detección de duplicados en el corpus completo...")
print("   (Genera embeddings de todos los chunks — puede tardar 2-3 min)\n")

# Ejecutar detección (sobre chunks_a para mantener consistencia)
chunks_limpios, reporte_dup = gestor.detectar_duplicados(chunks_a, modelo_emb)

# Guardar reporte
if reporte_dup:
    df_dup = pd.DataFrame(reporte_dup)
    df_dup.to_csv(OUTPUT_DIR / "duplicados_detectados.csv", index=False)
    print(f"\n📋 Reporte de duplicados guardado en outputs/duplicados_detectados.csv")
    print(df_dup[["chunk_a", "chunk_b", "similitud"]].head(5).to_string(index=False))
else:
    print("✅ No se detectaron duplicados en el corpus.")

# ── Reporte consolidado del sistema ──────────────────────────────────────
print("\n" + "═" * 60)
print("  REPORTE CONSOLIDADO — SISTEMA RAG")
print("═" * 60)

reporte = {
    "fecha_generacion":    datetime.now().isoformat(),
    "corpus": {
        "total_documentos_raw":  len(documentos_raw),
        "formatos":              df_raw["formato"].value_counts().to_dict() if "formato" in df_raw.columns else {},
    },
    "chunking": {
        "estrategia_seleccionada": "A_recursive (500 chars / 100 overlap)",
        "total_chunks":            len(chunks_a),
        "chunks_tras_dedup":       len(chunks_limpios),
        "duplicados_eliminados":   len(chunks_a) - len(chunks_limpios),
    },
    "indice_vectorial": {
        "modelo_embeddings":  "text-embedding-3-small",
        "dimension":          1536,
        "vectores_indexados": indice_faiss.index.ntotal,
        "persistencia":       str(INDEX_DIR / "indice_principal"),
    },
    "calidad_ragas": {
        "faithfulness":     float(df_ragas["faithfulness"].mean()),
        "answer_relevancy": float(df_ragas["answer_relevancy"].mean()),
        "preguntas_eval":   20,
    },
    "riesgos": {
        "duplicados_detectados":   len(reporte_dup),
        "docs_versionados":        len(gestor.registro_versiones),
        "docs_obsoletos":          sum(1 for v in gestor.registro_versiones.values() if not v["activo"]),
    }
}

print(json.dumps(reporte, indent=2, ensure_ascii=False))

# Guardar reporte
with open(OUTPUT_DIR / "reporte_sistema_rag.json", "w", encoding="utf-8") as f:
    json.dump(reporte, f, indent=2, ensure_ascii=False)

print(f"\n✅ Reporte guardado en outputs/reporte_sistema_rag.json")
```

**Salida esperada**:
```
🔎 Ejecutando detección de duplicados en el corpus completo...

   Duplicados detectados: 3
   Chunks antes: 142 → después: 139

════════════════════════════════════════════════════════════
  REPORTE CONSOLIDADO — SISTEMA RAG
════════════════════════════════════════════════════════════
{
  "fecha_generacion": "2024-11-15T14:32:11.123456",
  "corpus": {
    "total_documentos_raw": 38,
    "formatos": {"pdf": 18, "docx": 12, "html": 8}
  },
  "chunking": {
    "estrategia_seleccionada": "A_recursive (500 chars / 100 overlap)",
    "total_chunks": 142,
    "chunks_tras_dedup": 139,
    "duplicados_eliminados": 3
  },
  ...
}

✅ Reporte guardado en outputs/reporte_sistema_rag.json
```

---

## 7. Validación y Pruebas

Ejecuta las siguientes verificaciones finales para confirmar que el sistema está operativo:

```python
# Celda 12: Validación final del sistema

print("🧪 VALIDACIÓN FINAL DEL SISTEMA RAG\n")
errores = []

# ── Prueba 1: Índice FAISS cargable desde disco ───────────────────────────
try:
    indice_cargado = FAISS.load_local(
        str(INDEX_DIR / "indice_principal"),
        modelo_emb,
        allow_dangerous_deserialization=True
    )
    assert indice_cargado.index.ntotal > 0
    print(f"✅ [1/5] Índice FAISS cargable: {indice_cargado.index.ntotal} vectores")
except Exception as e:
    errores.append(f"❌ [1/5] Error cargando índice: {e}")
    print(errores[-1])

# ── Prueba 2: Búsqueda retorna resultados relevantes ─────────────────────
try:
    res = indice_faiss.similarity_search("política de vacaciones", k=3)
    assert len(res) == 3
    print(f"✅ [2/5] Búsqueda vectorial: {len(res)} resultados recuperados")
except Exception as e:
    errores.append(f"❌ [2/5] Error en búsqueda: {e}")
    print(errores[-1])

# ── Prueba 3: Cadena RAG genera respuesta con fuentes ────────────────────
try:
    res_rag = cadena_rag.invoke({"query": "¿Qué es el onboarding?"})
    assert len(res_rag["result"]) > 50
    assert len(res_rag["source_documents"]) > 0
    print(f"✅ [3/5] Cadena RAG funcional: respuesta de {len(res_rag['result'])} chars")
except Exception as e:
    errores.append(f"❌ [3/5] Error en cadena RAG: {e}")
    print(errores[-1])

# ── Prueba 4: Métricas RAGAS en rango válido ─────────────────────────────
try:
    faith = df_ragas["faithfulness"].mean()
    relev = df_ragas["answer_relevancy"].mean()
    assert 0 <= faith <= 1 and 0 <= relev <= 1
    print(f"✅ [4/5] Métricas RAGAS válidas: faithfulness={faith:.3f}, relevancy={relev:.3f}")
except Exception as e:
    errores.append(f"❌ [4/5] Error en métricas: {e}")
    print(errores[-1])

# ── Prueba 5: Archivos de salida generados ────────────────────────────────
archivos_esperados = [
    "comparativa_chunking.csv",
    "resultados_queries.csv",
    "metricas_ragas.csv",
    "reporte_sistema_rag.json"
]
try:
    for archivo in archivos_esperados:
        assert (OUTPUT_DIR / archivo).exists(), f"Falta: {archivo}"
    print(f"✅ [5/5] Archivos de salida: {len(archivos_esperados)} archivos generados")
except AssertionError as e:
    errores.append(f"❌ [5/5] {e}")
    print(errores[-1])

# ── Resumen ───────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
if not errores:
    print("🎉 TODAS LAS PRUEBAS PASARON. Sistema RAG operativo.")
else:
    print(f"⚠️  {len(errores)} prueba(s) fallaron. Revisar mensajes anteriores.")
```

**Criterios de éxito**:
- Las 5 pruebas deben mostrar `✅`
- `faithfulness` ≥ 0.70 (sistema con baja alucinación)
- `answer_relevancy` ≥ 0.70 (respuestas pertinentes)
- El índice FAISS debe ser cargable desde disco

---

## 8. Solución de Problemas

### Problema 1: `RateLimitError` o `APIError` durante la generación de embeddings

**Síntoma**: El proceso se interrumpe con `openai.RateLimitError: Rate limit reached` o `openai.APIStatusError: 429` durante la celda de construcción del índice FAISS.

**Causa**: La API de OpenAI impone límites de velocidad (rate limits) por minuto en el endpoint de embeddings. Con un corpus de 100+ chunks, las solicitudes en lote pueden superar el límite del tier gratuito o de bajo uso.

**Solución**:
```python
# Reemplaza la llamada directa a FAISS.from_documents con un proceso
# por lotes con pausa entre lotes

import time

def construir_indice_con_backoff(
    chunks: List[Document],
    modelo_emb: OpenAIEmbeddings,
    batch_size: int = 20,
    pausa_segundos: float = 2.0
) -> FAISS:
    """Construye índice FAISS procesando chunks en lotes con pausa."""
    print(f"Procesando {len(chunks)} chunks en lotes de {batch_size}...")
    
    # Primer lote para inicializar el índice
    indice = FAISS.from_documents(chunks[:batch_size], modelo_emb)
    
    # Lotes restantes
    for i in range(batch_size, len(chunks), batch_size):
        lote = chunks[i:i + batch_size]
        indice.add_documents(lote)
        print(f"  Lote {i//batch_size + 1}: {len(lote)} chunks indexados")
        time.sleep(pausa_segundos)  # pausa para respetar rate limits
    
    return indice

# Usar en lugar de FAISS.from_documents
indice_faiss = construir_indice_con_backoff(CHUNKS_INDEXAR, modelo_emb)
```

---

### Problema 2: Métricas RAGAS retornan `NaN` o el proceso falla con `ValidationError`

**Síntoma**: La celda de evaluación RAGAS termina con valores `NaN` en `faithfulness` o `answer_relevancy`, o lanza `pydantic.ValidationError` / `datasets.DatasetGenerationError`.

**Causa**: RAGAS requiere que el campo `contexts` sea una lista de listas de strings (`List[List[str]]`) y que ningún campo esté vacío. Si los documentos recuperados son muy cortos o el campo `ground_truth` tiene formato incorrecto, la validación interna falla.

**Solución**:
```python
# Verificar formato del dataset ANTES de llamar a evaluate()
print("Verificando formato del dataset de evaluación...")
for i, (q, a, ctx) in enumerate(zip(
    data_eval["question"],
    data_eval["answer"],
    data_eval["contexts"]
)):
    assert isinstance(q, str) and len(q) > 5,   f"Pregunta {i} inválida"
    assert isinstance(a, str) and len(a) > 10,  f"Respuesta {i} vacía"
    assert isinstance(ctx, list) and len(ctx) > 0, f"Contexto {i} vacío"
    assert all(isinstance(c, str) and len(c) > 10 for c in ctx), \
        f"Contexto {i} contiene strings vacíos"

print("✅ Dataset válido para RAGAS")

# Si el LLM devuelve respuestas vacías, añadir fallback:
for i, respuesta in enumerate(data_eval["answer"]):
    if not respuesta or len(respuesta) < 10:
        data_eval["answer"][i] = "No tengo información suficiente sobre este tema."

# Asegurarse de que ground_truth sea lista de strings (no vacíos)
data_eval["ground_truth"] = ["N/A"] * len(data_eval["question"])
```

---

## 9. Limpieza del Entorno

```python
# Celda 13: Limpieza (ejecutar al finalizar el laboratorio)

import shutil

print("🧹 Iniciando limpieza del entorno...\n")

# 1. Liberar variables grandes de memoria
del CHUNKS_INDEXAR
del chunks_a, chunks_b, chunks_c
del documentos_raw
print("✅ Variables grandes liberadas de memoria")

# 2. El índice FAISS se mantiene en disco (es el producto del lab)
print(f"✅ Índice FAISS conservado en: {INDEX_DIR / 'indice_principal'}")

# 3. Los outputs se conservan para revisión
print(f"✅ Outputs conservados en: {OUTPUT_DIR}")
print(f"   Archivos generados:")
for f in sorted(OUTPUT_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"     {f.name} ({size_kb:.1f} KB)")

# 4. OPCIONAL: eliminar archivos temporales (descomentar si se desea)
# temp_files = list(Path(".").glob("*.tmp"))
# for tf in temp_files:
#     tf.unlink()
# print(f"✅ {len(temp_files)} archivos temporales eliminados")

print("\n⚠️  RECORDATORIO: No subas el archivo .env a ningún repositorio.")
print("   Verifica que .env esté en tu .gitignore antes de hacer commit.")
```

**Archivos a conservar** (producto del laboratorio):
- `indices/indice_principal/` — Índice FAISS listo para uso
- `outputs/reporte_sistema_rag.json` — Reporte consolidado del sistema
- `outputs/metricas_ragas.csv` — Métricas de calidad para presentación
- `outputs/comparativa_chunking.csv` — Análisis de estrategias de chunking

**Archivos que pueden eliminarse**:
- Archivos `.tmp` generados durante el procesamiento
- Copias intermedias del corpus si el espacio en disco es limitado

---

## 10. Resumen

### Lo que construiste

En este laboratorio implementaste un **sistema RAG completo de extremo a extremo**, siguiendo exactamente el flujo de la Lección 4.1: base de conocimiento indexada → motor de recuperación → modelo generador. Los componentes construidos son:

| Componente RAG | Implementación en el lab |
|---|---|
| **Ingesta y preprocesamiento** | `DocumentLoader` con loaders para PDF (PyMuPDF), DOCX (python-docx) y HTML (BeautifulSoup4) |
| **Chunking** | Tres estrategias comparadas: RecursiveChar, HTMLHeader y sección de documento |
| **Base de datos vectorial** | Índice FAISS con embeddings `text-embedding-3-small` (1536 dimensiones) |
| **Retriever** | Búsqueda por similitud coseno con re-ranking por relevancia y metadatos |
| **Generador** | GPT-4o-mini con prompt personalizado de citación de fuentes |
| **Evaluación** | RAGAS: faithfulness y answer_relevancy sobre 20 preguntas |
| **Mitigación de riesgos** | Detección de duplicados, versionado de documentos, alertas de contradicción |

### Conceptos clave aplicados

- **Flujo RAG**: comprendiste en la práctica que la calidad del sistema depende de cada etapa en cadena: una ingesta deficiente degrada la recuperación, y una recuperación deficiente degrada la generación
- **Estrategias de chunking**: el tamaño y método de división de documentos tiene impacto directo en la coherencia semántica de los chunks y en la precisión de la recuperación
- **Re-ranking**: la búsqueda vectorial inicial puede mejorarse con filtros post-recuperación basados en metadatos y penalización de duplicados
- **Métricas de calidad**: `faithfulness` mide si el modelo "alucina" fuera del contexto; `answer_relevancy` mide si la respuesta es pertinente a la pregunta

### Recursos adicionales

- [Lewis et al. (2020) — Artículo original RAG](https://arxiv.org/abs/2005.11401)
- [LangChain — Documentación de RetrievalQA](https://python.langchain.com/docs/tutorials/rag/)
- [FAISS — Guía de uso con Python](https://faiss.ai/index_factory.html)
- [RAGAS — Documentación de métricas](https://docs.ragas.io/en/stable/concepts/metrics/)
- [OpenAI — Guía de embeddings text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings)

---

*Lab 04-00-01 | Práctica 4: Pipeline de Ingestión y Creación de Índice Vectorial | Duración: 72 min*
