import os
import streamlit as st
from groq import Groq
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Don Quijote con LLaMA 3 (Groq)", layout="wide")

ROOT = Path(__file__).parent
TXT_PATH = ROOT / "don_quijote.txt"
SENTCSV = ROOT / "labeled_sentences.csv"

st.title("Don Quijote — NLP con LLaMA 3 (Groq)")

# Configurar cliente Groq
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("No se encontró GROQ_API_KEY")
else:
    client = Groq(api_key=api_key)
    st.success("Conectado a Groq con LLaMA 3")

# Dataset
if SENTCSV.exists():
    df = pd.read_csv(SENTCSV)
    st.sidebar.header("Ejemplos etiquetados")
    st.sidebar.dataframe(df.head(10))

st.header("1) Clasificación de frases (ejemplo)")
if SENTCSV.exists():
    st.table(df.sample(min(5, len(df))))

def ask_llama3(prompt, max_tokens=200):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

st.header("2) Completado de oraciones")
seed = st.text_area("Escribe un comienzo", value="En un lugar de la Mancha,")
if st.button("Generar continuación"):
    out = ask_llama3(f"Continúa el siguiente texto del Quijote: {seed}", 150)
    st.write(out)

st.header("3) QA (pregunta-respuesta)")
q = st.text_input("Pregunta", value="¿Quién es Don Quijote?")
if st.button("Responder"):
    context = TXT_PATH.read_text(encoding="utf-8")[:6000]
    prompt = f"Usa este contexto del Quijote para responder:\n{context}\n\nPregunta: {q}\nRespuesta:"
    out = ask_llama3(prompt, 250)
    st.write(out)

st.header("4) Resumen")
if st.button("Generar resumen"):
    text = TXT_PATH.read_text(encoding="utf-8")[:6000]
    prompt = f"Resume en 2 líneas el siguiente pasaje del Quijote:\n\n{text}"
    out = ask_llama3(prompt, 220)
    st.write(out)


