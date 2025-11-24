import streamlit as st
import google.generativeai as genai
from PIL import Image

st.set_page_config(page_title="Test de Visión IA", page_icon="👁️")

st.title(" Prueba de Visión Artificial")
st.markdown("Este programa sirve para verificar qué está viendo la IA exactamente.")

# 1. Configuración de API Key
# Intentamos leer de Secrets, si no, pedimos manual
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.success("✅ API Key detectada en el sistema.")
else:
    api_key = st.text_input("Pega tu API Key de Google:", type="password")

# 2. Subir Imagen
uploaded_file = st.file_uploader("Sube la foto del examen", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None and api_key:
    try:
        # Mostrar imagen subida
        image = Image.open(uploaded_file)
        st.image(image, caption="Tu foto original", width=300)
        
        if st.button("🔍 Extraer Texto"):
            with st.spinner("Consultando a Google Gemini..."):
                # Configuración simple
                genai.configure(api_key=api_key)
                
                # Probamos con el modelo estándar estable
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Prompt simple: Solo transcribe
                prompt = """
                Tu única tarea es TRANSCRIBIR el contenido de esta imagen a texto.
                - Si hay fórmulas matemáticas, escríbelas en formato LaTeX o Python.
                - Si hay una matriz, escribe los números ordenados.
                - No resuelvas nada, solo dime qué dice el texto.
                """
                
                response = model.generate_content([prompt, image])
                
                st.subheader("Lo que la IA pudo leer:")
                st.info(response.text)
                
                st.success("Si puedes leer el texto arriba, ¡la conexión funciona!")

    except Exception as e:
        st.error(f"⚠️ Ocurrió un error: {e}")
        st.warning("Si el error dice '404 Not Found', el modelo 'flash' no está disponible para tu clave. Intenta cambiar en el código 'gemini-1g.5-flash' por 'gemini-pro-vision'.")