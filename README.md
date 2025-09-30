Don_Quijote_Streamlit_App_Mistral
=================================

Contenido:
- app.py : Streamlit app conectada directamente a **Mistral-7B-Instruct-v0.2** vía Hugging Face.
- requirements.txt : dependencias necesarias.
- labeled_sentences.csv : dataset extendido de frases etiquetadas.
- don_quijote.txt : texto completo (inclúyelo en la carpeta).

Instrucciones:
1. Coloca `don_quijote.txt` en la misma carpeta.
2. Exporta tu token de Hugging Face (HF_TOKEN):
   ```bash
   export HF_TOKEN=tu_token
   ```
   En Windows PowerShell:
   ```powershell
   $env:HF_TOKEN="tu_token"
   ```
3. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecuta la app:
   ```bash
   streamlit run app.py
   ```

Notas:
- Se usa `trust_remote_code=True` porque Mistral implementa su propio código.
- Este modelo usa el formato de instrucciones `[INST] ... [/INST]` para completar de manera más coherente.
