Aplicación en Streamlit que utiliza LLaMA 3 a través de la API de Groq para realizar tareas de NLP sobre el texto de Don Quijote.

Incluye:

Clasificación de frases.

Completado de oraciones.

Pregunta-respuesta (QA).

Resumen automático.

3. Instala dependencias:
   pip install -r requirements.txt

4. Ejecuta la app:
   streamlit run app.py

Notas:
- Usa GPT-3.5-Turbo por defecto (más estable y accesible).
- Puedes cambiar a gpt-4-turbo o gpt-4o-mini modificando el parámetro `model` en app.py.

