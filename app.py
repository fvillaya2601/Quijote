import os
import streamlit as st
import pandas as pd
from pathlib import Path

from transformers import pipeline

st.set_page_config(page_title="Don Quijote con Qwen-7B", layout="wide")

ROOT = Path(__file__).parent
TXT_PATH = ROOT / "don_quijote.txt"
SENTCSV = ROOT / "labeled_sentences.csv"

st.title("Don Quijote — NLP con Qwen-7B (trust_remote_code)")
st.markdown("""
Esta demo usa el modelo **Qwen-7B** desde Hugging Face (requiere `HF_TOKEN`).  
Se agregó `trust_remote_code=True` para permitir cargar el código personalizado del modelo.  
""")

# Load dataset
if SENTCSV.exists():
    df = pd.read_csv(SENTCSV)
    st.sidebar.header("Ejemplos etiquetados")
    st.sidebar.dataframe(df.head(10))

# Load Qwen pipeline if token provided
hf_token = os.environ.get("HF_TOKEN")
generator = None
qa_model = None
summarizer = None
if hf_token:
    try:
        generator = pipeline(
            "text-generation",
            model="Qwen/Qwen-7B",
            use_auth_token=hf_token,
            device_map="auto",
            trust_remote_code=True
        )
        qa_model = pipeline(
            "question-answering",
            model="Qwen/Qwen-7B",
            use_auth_token=hf_token,
            device_map="auto",
            trust_remote_code=True
        )
        summarizer = pipeline(
            "summarization",
            model="Qwen/Qwen-7B",
            use_auth_token=hf_token,
            device_map="auto",
            trust_remote_code=True
        )
        st.success("Modelos Qwen-7B cargados correctamente.")
    except Exception as e:
        st.error(f"No se pudo cargar Qwen-7B: {e}")

st.header("1) Clasificación de frases (ejemplo)")
if SENTCSV.exists():
    st.table(df.sample(min(5, len(df))))

st.header("2) Completado de oraciones")
seed = st.text_area("Escribe un comienzo", value="En un lugar de la Mancha,")
if st.button("Generar continuación con Qwen-7B"):
    if generator:
        out = generator(seed, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7)
        st.write(out[0]["generated_text"])
    else:
        st.warning("No se cargó Qwen-7B. Configura HF_TOKEN.")

st.header("3) QA extractivo / generativo")
q = st.text_input("Pregunta", value="¿Qué le pide Don Quijote a Sancho antes de la ínsula?")
if st.button("Responder con Qwen-7B"):
    if qa_model:
        text = TXT_PATH.read_text(encoding="utf-8")[:4000]
        res = qa_model(question=q, context=text)
        st.write(res)
    else:
        st.warning("No se cargó Qwen-7B QA.")

st.header("4) Resumen")
if st.button("Resumen corto con Qwen-7B"):
    if summarizer:
        text = TXT_PATH.read_text(encoding="utf-8")[:3000]
        res = summarizer(text, max_length=120, min_length=40, do_sample=False)
        st.write(res[0]["summary_text"])
    else:
        st.warning("No se cargó Qwen-7B Summarizer.")
