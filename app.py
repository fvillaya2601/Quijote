import os
import streamlit as st
from openai import OpenAI
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Don Quijote con GPT", layout="wide")

ROOT = Path(__file__).parent
TXT_PATH = ROOT / "don_quijote.txt"
SENTCSV = ROOT / "labeled_sentences.csv"

st.title("Don Quijote — NLP con GPT-3.5")
st.markdown("""
Esta demo usa **GPT-3.5-Turbo** a través de la API de OpenAI.  
Necesitas configurar tu variable de entorno `OPENAI_API_KEY`.  
""")

client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    st.success("API Key encontrada, listo para usar GPT-3.5-Turbo")
else:
    st.error("No se encontró la variable de entorno OPENAI_API_KEY")

if SENTCSV.exists():
    df = pd.read_csv(SENTCSV)
    st.sidebar.header("Ejemplos etiquetados")
    st.sidebar.dataframe(df.head(10))

st.header("1) Clasificación de frases (ejemplo)")
if SENTCSV.exists():
    st.table(df.sample(min(5, len(df))))

st.header("2) Completado de oraciones")
seed = st.text_area("Escribe un comienzo", value="En un lugar de la Mancha,")
if st.button("Generar continuación con GPT"):
    if client:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Continúa el siguiente texto del Quijote: {seed}"}],
            max_tokens=200
        )
        st.write(response.choices[0].message.content)

st.header("3) QA (pregunta-respuesta)")
q = st.text_input("Pregunta", value="¿Qué le pide Don Quijote a Sancho antes de la ínsula?")
if st.button("Responder con GPT"):
    if client:
        context = TXT_PATH.read_text(encoding="utf-8")[:2000]
        prompt = f"Usa este contexto del Quijote para responder:\n{context}\n\nPregunta: {q}\nRespuesta:"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250
        )
        st.write(response.choices[0].message.content)

st.header("4) Resumen")
if st.button("Generar resumen con GPT"):
    if client:
        text = TXT_PATH.read_text(encoding="utf-8")[:2000]
        prompt = f"Resume en 2 líneas el siguiente pasaje del Quijote:\n\n{text}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        st.write(response.choices[0].message.content)

