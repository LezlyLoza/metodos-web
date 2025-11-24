import streamlit as st
import google.generativeai as genai
from PIL import Image

st.set_page_config(page_title="Diagnóstico Gemini", page_icon="🩺")
st.title("🩺 Diagnóstico de Conexión Google")

# 1. API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.success("✅ API Key encontrada.")
else:
    api_key = st.text_input("API Key:", type="password")

if api_key:
    try:
        genai.configure(api_key=api_key)
        
        # BOTÓN DE DIAGNÓSTICO
        if st.button("📋 Listar Modelos Disponibles"):
            st.info("Consultando a Google...")
            modelos = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    modelos.append(m.name)
            
            st.write("### Tus modelos disponibles son:")
            st.json(modelos)
            
            # Prueba automática con el primer modelo compatible
            if 'models/gemini-1.5-flash' in modelos:
                st.success("✅ ¡SÍ TIENES gemini-1.5-flash! Úsalo.")
            elif 'models/gemini-1.5-flash-001' in modelos:
                st.success("✅ Tienes la versión 001. Cambia el código a 'gemini-1.5-flash-001'.")
            else:
                st.error("❌ No veo el modelo Flash. Usa uno de la lista de arriba.")

    except Exception as e:
        st.error(f"Error grave de conexión: {e}")
        st.warning("Si este error dice 'module not found', es el requirements.txt")

# Subida de imagen simple para probar si la lista funciona
uploaded = st.file_uploader("Sube foto para test final")
if uploaded and st.button("Probar Visión") and api_key:
    model = genai.GenerativeModel('gemini-1.5-flash')
    st.write(model.generate_content(["Describe esto", Image.open(uploaded)]).text)