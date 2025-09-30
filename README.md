Don_Quijote_Streamlit_App_GPT
=================================

Contenido:
- app.py : Streamlit app conectada a GPT-3.5-Turbo.
- requirements.txt : dependencias necesarias.
- labeled_sentences.csv : dataset extendido de frases etiquetadas.
- don_quijote.txt : texto completo (inclúyelo en la carpeta).

Instrucciones:
1. Crea una API Key en https://platform.openai.com y cópiala.
2. Configura la variable de entorno:

   Linux/macOS:
   export OPENAI_API_KEY="sk-..."

   Windows PowerShell:
   $env:OPENAI_API_KEY="sk-..."

   En Streamlit Cloud o Azure:
   - Ve a Settings > Secrets > agrega OPENAI_API_KEY con tu valor.

3. Instala dependencias:
   pip install -r requirements.txt

4. Ejecuta la app:
   streamlit run app.py

Notas:
- Usa GPT-3.5-Turbo por defecto (más estable y accesible).
- Puedes cambiar a gpt-4-turbo o gpt-4o-mini modificando el parámetro `model` en app.py.

