import os
import streamlit as st
import pandas as pd
from pathlib import Path
from transformers import pipeline

st.set_page_config(page_title="Don Quijote con Mistral-7B-Instruct", layout="wide")

ROOT = Path(__file__).parent
TXT_PATH = ROOT / "don_quijote.txt"
SENTCSV = ROOT / "labeled_sentences.csv"

st.title("Don Quijote — NLP con Mistral-7B-Instruct")
st.markdown("""
Esta demo usa el modelo **Mistral-7B-Instruct-v0.2** desde Hugging Face (requiere `HF_TOKEN`).  
Se agregó `trust_remote_code=True` para permitir cargar el código personalizado del modelo.  
""")

# Load dataset
if SENTCSV.exists():
    df = pd.read_csv(SENTCSV)
    st.sidebar.header("Ejemplos etiquetados")
    st.sidebar.dataframe(df.head(10))

# Load Mistral pipeline if token provided
hf_token = os.environ.get("HF_TOKEN")
generator = None
if hf_token:
    try:
        generator = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            use_auth_token=hf_token,
            device_map="auto",
            trust_remote_code=True
        )
        st.success("Modelo Mistral-7B-Instruct cargado correctamente.")
    except Exception as e:
        st.error(f"No se pudo cargar Mistral-7B-Instruct: {e}")

st.header("1) Clasificación de frases (ejemplo)")
if SENTCSV.exists():
    st.table(df.sample(min(5, len(df))))

st.header("2) Completado de oraciones")
seed = st.text_area("Escribe un comienzo", value="En un lugar de la Mancha,")
if st.button("Generar continuación con Mistral-7B-Instruct"):
    if generator:
        prompt = f"[INST] {seed} [/INST]"
        out = generator(prompt, max_new_tokens=120, do_sample=True, top_p=0.9, temperature=0.7)
        st.write(out[0]["generated_text"])
    else:
        st.warning("No se cargó Mistral-7B-Instruct. Configura HF_TOKEN.")

st.header("3) QA simple (búsqueda por palabras clave)")
q = st.text_input("Pregunta", value="¿Qué le pide Don Quijote a Sancho antes de la ínsula?")
if st.button("Buscar en texto (naive)"):
    text = TXT_PATH.read_text(encoding="utf-8")
    keywords = [w.strip("?,¡!.").lower() for w in q.split() if len(w)>3]
    hits = []
    for line in text.splitlines():
        if any(k in line.lower() for k in keywords):
            hits.append(line.strip())
            if len(hits) >= 5:
                break
    if hits:
        st.write("Fragmentos relevantes:")
        for h in hits:
            st.write(f"> {h}")
    else:
        st.write("No se encontraron fragmentos.")
        
st.header("4) Resumen (fallback naive)")
if st.button("Generar resumen corto"):
    text = TXT_PATH.read_text(encoding="utf-8")[:1500]
    parts = text.split("\n\n")
    summary = " ".join(p.strip() for p in parts[:2] if p.strip())
    st.write(summary)
