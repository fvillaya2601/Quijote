import streamlit as st
import pandas as pd
from pathlib import Path
from transformers import pipeline

st.set_page_config(page_title="Don Quijote con Flan-T5-Large", layout="wide")

ROOT = Path(__file__).parent
TXT_PATH = ROOT / "don_quijote.txt"
SENTCSV = ROOT / "labeled_sentences.csv"

st.title("Don Quijote — NLP")
st.markdown("""
Tareas disponibles:
- Clasificación de frases.
- Completado de oraciones.
- QA (pregunta-respuesta).
- Resumen.
""")

# Load dataset
if SENTCSV.exists():
    df = pd.read_csv(SENTCSV)
    st.sidebar.header("Ejemplos etiquetados")
    st.sidebar.dataframe(df.head(10))

# Load Flan pipeline
try:
    generator = pipeline("text2text-generation", model="google/flan-t5-large")
    st.success("Modelo Flan-T5-Large cargado correctamente.")
except Exception as e:
    st.error(f"No se pudo cargar Flan-T5-Large: {e}")
    generator = None

st.header("1) Clasificación de frases (ejemplo)")
if SENTCSV.exists():
    st.table(df.sample(min(5, len(df))))

st.header("2) Completado de oraciones")
seed = st.text_area("Escribe un comienzo", value="En un lugar de la Mancha,")
if st.button("Generar continuación con Flan-T5-Large"):
    if generator:
        prompt = f"Continúa el siguiente texto: {seed}"
        out = generator(prompt, max_new_tokens=80)
        st.write(out[0]["generated_text"])
    else:
        st.warning("No se cargó Flan-T5-Large.")

st.header("3) QA (pregunta-respuesta)")
q = st.text_input("Pregunta", value="¿Qué le pide Don Quijote a Sancho antes de la ínsula?")
if st.button("Responder con Flan-T5-Large"):
    if generator:
        context = TXT_PATH.read_text(encoding="utf-8")[:2000]
        prompt = f"Pregunta: {q}\nContexto: {context}\nRespuesta:"
        res = generator(prompt, max_new_tokens=100)
        st.write(res[0]["generated_text"])
    else:
        st.warning("No se cargó Flan-T5-Large.")

st.header("4) Resumen")
if st.button("Generar resumen con Flan-T5-Large"):
    if generator:
        text = TXT_PATH.read_text(encoding="utf-8")[:2000]
        prompt = f"Resume el siguiente texto en 2 líneas: {text}"
        res = generator(prompt, max_new_tokens=80)
        st.write(res[0]["generated_text"])
    else:
        st.warning("No se cargó Flan-T5-Large.")
