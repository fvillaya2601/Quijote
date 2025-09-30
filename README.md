Don_Quijote_Streamlit_App_Flan
=================================

Contenido:
- app.py : Streamlit app conectada a **google/flan-t5-large** (no requiere HF_TOKEN).
- requirements.txt : dependencias necesarias.
- labeled_sentences.csv : dataset extendido de frases etiquetadas.
- don_quijote.txt : texto completo (inclúyelo en la carpeta).

Instrucciones:
1. Coloca `don_quijote.txt` en la misma carpeta.
2. Instala dependencias:
   pip install -r requirements.txt
3. Ejecuta la app:
   streamlit run app.py

Notas:
- Se usa **Flan-T5-Large**, un modelo más ligero que funciona bien para QA, resumen y completado en español.
- No necesitas configurar HF_TOKEN ni instalar dependencias adicionales.
