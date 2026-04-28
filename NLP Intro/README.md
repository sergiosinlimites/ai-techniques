# Introducción al Procesamiento de Lenguaje Natural (NLP)

## Resumen del PDF

El PDF presenta una introducción al **Procesamiento de Lenguaje Natural (NLP)**, una rama de la Inteligencia Artificial que permite a las computadoras entender, interpretar y manipular el lenguaje humano. El documento covers los siguientes temas fundamentales:

### 1. Conceptos Básicos de NLP
- **Tokenización**: Proceso de dividir texto en unidades más pequeñas (palabras u oraciones)
- **Stop Words**: Palabras comunes que no aportan significado relevante (como "the", "is", "at")
- **Stemming**: Reducción de palabras a su raíz cortando sufijos (ej: "running" → "run")
- **Lemmatización**: Reducción de palabras a su forma base o lemma (ej: "scarves" → "scarf")
- **POS Tagging**: Etiquetado de partes de la oración (sustantivos, verbos, adjetivos, etc.)
- **Chunking**: Agrupación de palabras en frases significativas
- **Chinking**: Exclusión de ciertos elementos del chunking

### 2. Aplicaciones del NLP
- Análisis de sentimiento
- Traducción automática
- chatbots y asistentes virtuales
- Clasificación de texto
- Extracción de información

---

## Implementación en Código

El notebook `TIA_20260210_Introduccion_NLP.ipynb` demuestra cada uno de estos conceptos con código funcional usando la librería **NLTK** (Natural Language Toolkit) de Python.

### Tokenización

```python
from nltk.tokenize import word_tokenize, sent_tokenize

example_string = "Muad'Dib learned rapidly because his first training was in how to learn."

# Por oraciones
sent_tokenize(example_string)
# Resultado: ['Muad\'Dib learned rapidly because his first training was in how to learn.', ...]

# Por palabras
word_tokenize(example_string)
# Resultado: ["Muad'Dib", 'learned', 'rapidly', 'because', ...]
```

### Filtrado de Stop Words

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_text = "This is a simple example demonstrating stop word filtering."
stop_words = set(stopwords.words('english'))

# Filtrar palabras
words = word_tokenize(example_text)
filtered_words = [word for word in words if word.lower() not in stop_words]
# Resultado: ['simple', 'example', 'demonstrating', 'stop', 'word', 'filtering', '.']
```

### Stemming

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "runner", "ran", "easily", "happiness"]
stemmed_words = [stemmer.stem(word) for word in words]
# Resultado: ['run', 'runner', 'ran', 'easili', 'happi']
```

### POS Tagging

```python
import nltk
from nltk.tokenize import word_tokenize

sagan_quote = "If you wish to make an apple pie from scratch, you must first invent the universe."
words = word_tokenize(sagan_quote)
tagged_words = nltk.pos_tag(words)
# Resultado: [('If', 'IN'), ('you', 'PRP'), ('wish', 'VBP'), ...]
```

Etiquetas comunes del Penn Treebank:
- `NN` → Sustantivo (Noun)
- `VB` → Verbo base
- `DT` → Determinante/Artículo
- `JJ` → Adjetivo
- `PRP` → Pronombre personal

### Lematización

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ["scarves", "running", "friends"]
lemmatized = [lemmatizer.lemmatize(word) for word in words]
# Resultado: ['scarf', 'running', 'friend']

# Especificar categoría gramatical
lemmatizer.lemmatize("worst", pos="a")  # 'bad'
```

### Chunking

```python
import nltk
from nltk.tokenize import word_tokenize

# Definir gramática de chunking
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)

# Aplicar POS tagging y luego chunking
words = word_tokenize("It's a dangerous business, Frodo, going out your door.")
tags = nltk.pos_tag(words)
tree = chunk_parser.parse(tags)
tree.pretty_print()
```

### Chinking

```python
grammar = """
    Chunk: {<DT>?<NN.*>}
    Chink: {<JJ>}
"""
chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(tags)
```

---

## Resumen de Conceptos vs Código

| Concepto | Descripción | Código NLTK |
|---------|-------------|-------------|
| Tokenización | Dividir texto en oraciones/palabras | `sent_tokenize()`, `word_tokenize()` |
| Stop Words | Filtrar palabras comunes | `stopwords.words('english')` |
| Stemming | Reducir a raíz (cortando) | `PorterStemmer().stem()` |
| Lematización | Reducir a forma base | `WordNetLemmatizer().lemmatize()` |
| POS Tagging | Etiquetar categorías gramaticales | `nltk.pos_tag()` |
| Chunking | Agrupar en frases | `nltk.RegexpParser()` |
| Chinking | Excluir del chunking | Gramática con `Chunk` y `Chink` |

---

## Requisitos

```bash
pip install nltk==3.5
```

Recursos necesarios de NLTK:
```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab')
```