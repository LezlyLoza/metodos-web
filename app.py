import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import google.generativeai as genai
from PIL import Image
import json

# --- 1. CONFIGURACIÓN INICIAL Y STATE ---
st.set_page_config(page_title="Metodos Numéricos Pro", layout="wide", page_icon="⚡")

def init_session():
    defaults = {
        "nav_selection": "1. Raíces (Bisección/Newton)", 
        "bis_func": "0.95*x**3 - 5.9*x**2 + 10.9*x - 6", "bis_a": 3.0, "bis_b": 4.0,
        "newton_func": "0.95*x**3 - 5.9*x**2 + 10.9*x - 6", "newton_x0": 3.5,
        "m_a11": 5.0, "m_a12": 1.0, "m_a13": -15.0, "m_b1": 5.0,
        "m_a21": 13.0, "m_a22": 3.0, "m_a23": 9.0, "m_b2": 30.0,
        "m_a31": 1.0, "m_a32": 4.0, "m_a33": 2.0, "m_b3": 15.0,
        "int_func": "(5*x**3 + x) / sqrt(3*x**2 + 5)", "int_a": 1.0, "int_b": 5.0,
        "der_func": "sqrt(5*x**3 + 1)", "der_x": 1.3, "der_h": 0.1,
        "opciones_detectadas": {} 
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# --- FUNCIÓN DE COMPARACIÓN DE INCISOS ---
def mostrar_inciso_correcto(resultado_calculado):
    opciones = st.session_state.opciones_detectadas
    if not opciones: return
    
    st.divider()
    st.markdown("### 🎯 Veredicto Automático")
    
    mejor_opcion = None
    menor_diferencia = float('inf')
    
    for letra, valor in opciones.items():
        try:
            diff = abs(resultado_calculado - float(valor))
            if diff < menor_diferencia:
                menor_diferencia = diff
                mejor_opcion = letra
        except: pass

    if mejor_opcion and menor_diferencia < 1.0: 
        st.success(f"✅ La respuesta correcta es la opción **{mejor_opcion.upper()}** ({opciones[mejor_opcion]})")
        st.caption(f"Cálculo exacto: {resultado_calculado:.6f}")
    else:
        st.warning(f"El resultado ({resultado_calculado:.4f}) no parece coincidir con ninguna opción detectada.")

# --- 2. LÓGICA DE IA (SELECTOR DE TU LISTA) ---
def analizar_imagen_con_ia(api_key, image):
    genai.configure(api_key=api_key)
    
    # HE ACTUALIZADO ESTA LISTA CON LOS MODELOS QUE ME PASASTE
    modelos_a_probar = [
        "models/gemini-2.0-flash-exp",      # Tu opción #5 (Muy rápido y actual)
        "models/gemini-flash-latest",       # Tu opción #28 (Siempre funciona)
        "models/gemini-2.0-flash",          # Tu opción #6
        "models/gemini-2.5-flash-preview-09-2025" # Tu opción #34 (Futurista)
    ]
    
    prompt = """
    Analiza la imagen. Extrae datos y opciones (a, b, c).
    Devuelve SOLO JSON estricto.
    Formato:
    {
        "metodo": "biseccion" | "newton" | "gauss_seidel" | "integracion" | "derivada",
        "func": "python_syntax",
        "opciones": {"a": 1.2, "b": 3.4},
        ... (otros datos a, b, x0, matrix, x, h)
    }
    """

    for nombre_modelo in modelos_a_probar:
        try:
            # Intentar generar con el modelo de la lista
            model = genai.GenerativeModel(nombre_modelo)
            response = model.generate_content([prompt, image])
            
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            if "{" in clean_text:
                clean_text = clean_text[clean_text.find("{"):clean_text.rfind("}")+1]
                
            return json.loads(clean_text)
            
        except Exception as e:
            print(f"Fallo con {nombre_modelo}: {e}")
            continue
            
    st.sidebar.error("❌ Fallaron todos los modelos. Verifica que tu API Key sea válida para modelos experimentales.")
    return None

# --- 3. BARRA LATERAL ---
with st.sidebar:
    st.title("🤖 Auto-Scanner 2.0")
    
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("Llave Conectada")
    else:
        api_key = st.text_input("API Key Google", type="password")

    uploaded_file = st.file_uploader("📷 Subir Examen", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and api_key:
        if st.button("🚀 Escanear"):
            with st.spinner("Probando modelos de tu lista..."):
                img = Image.open(uploaded_file)
                datos = analizar_imagen_con_ia(api_key, img)
                
                if datos:
                    try:
                        metodo = datos.get("metodo")
                        st.session_state.opciones_detectadas = datos.get("opciones", {})
                        
                        if metodo == "biseccion":
                            st.session_state.nav_selection = "1. Raíces (Bisección/Newton)"
                            st.session_state.bis_func = datos.get('func', '')
                            st.session_state.bis_a = float(datos.get('a', 0))
                            st.session_state.bis_b = float(datos.get('b', 0))
                            
                        elif metodo == "newton":
                            st.session_state.nav_selection = "1. Raíces (Bisección/Newton)"
                            st.session_state.newton_func = datos.get('func', '')
                            st.session_state.newton_x0 = float(datos.get('x0', 0))

                        elif metodo == "gauss_seidel":
                            st.session_state.nav_selection = "2. Sistemas Lineales (Gauss-Seidel)"
                            m = datos.get('matrix', [])
                            if m:
                                st.session_state.m_a11 = float(m[0][0]); st.session_state.m_a12 = float(m[0][1]); st.session_state.m_a13 = float(m[0][2]); st.session_state.m_b1 = float(m[0][3])
                                st.session_state.m_a21 = float(m[1][0]); st.session_state.m_a22 = float(m[1][1]); st.session_state.m_a23 = float(m[1][2]); st.session_state.m_b2 = float(m[1][3])
                                st.session_state.m_a31 = float(m[2][0]); st.session_state.m_a32 = float(m[2][1]); st.session_state.m_a33 = float(m[2][2]); st.session_state.m_b3 = float(m[2][3])

                        elif metodo == "integracion":
                            st.session_state.nav_selection = "3. Integración (Simpson/Gauss)"
                            st.session_state.int_func = datos.get('func', '')
                            st.session_state.int_a = float(datos.get('a', 0))
                            st.session_state.int_b = float(datos.get('b', 0))

                        elif metodo == "derivada":
                            st.session_state.nav_selection = "4. Derivación Numérica"
                            st.session_state.der_func = datos.get('func', '')
                            st.session_state.der_x = float(datos.get('x', 0))
                            st.session_state.der_h = float(datos.get('h', 0))
                            
                        st.toast("✅ ¡Datos extraídos con éxito!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error procesando JSON: {e}")

    st.divider()
    seccion = st.radio("Ir a:", ["1. Raíces (Bisección/Newton)", "2. Sistemas Lineales (Gauss-Seidel)", "3. Integración (Simpson/Gauss)", "4. Derivación Numérica"], key="nav_selection")
    
    if st.session_state.opciones_detectadas:
        st.info(f"Opciones: {st.session_state.opciones_detectadas}")

# --- 4. CALCULADORAS ---

if st.session_state.nav_selection == "1. Raíces (Bisección/Newton)":
    st.header("🔍 Raíces")
    tabs = st.tabs(["Bisección", "Newton"])
    with tabs[0]:
        f_txt = st.text_input("f(x):", key="bis_func")
        c1, c2 = st.columns(2)
        a_val = c1.number_input("a:", key="bis_a")
        b_val = c2.number_input("b:", key="bis_b")
        if st.button("Calcular Bisección"):
            try:
                x = sp.symbols('x'); f = sp.lambdify(x, sp.sympify(f_txt), "numpy")
                if f(a_val)*f(b_val) >= 0: st.error("Error signos")
                else:
                    xr_ant=0; a=a_val; b=b_val; xr=0; datos=[]
                    for i in range(20):
                        xr = (a+b)/2; fxr = f(xr)
                        datos.append({"Iter": i, "xr": xr, "f(xr)": fxr})
                        if abs(fxr)<0.0001: break
                        if f(a)*fxr<0: b=xr
                        else: a=xr
                    mostrar_inciso_correcto(xr)
                    st.dataframe(pd.DataFrame(datos))
            except Exception as e: st.error(e)

    with tabs[1]:
        f_n = st.text_input("f(x):", key="newton_func")
        x0 = st.number_input("x0:", key="newton_x0")
        if st.button("Calcular Newton"):
            try:
                x = sp.symbols('x'); f_s = sp.sympify(f_n); df_s = sp.diff(f_s, x)
                f = sp.lambdify(x, f_s); df = sp.lambdify(x, df_s)
                xi = x0
                for i in range(10): xi = xi - (f(xi)/df(xi))
                mostrar_inciso_correcto(xi)
                st.write(f"Raíz: {xi}")
            except Exception as e: st.error(e)

elif st.session_state.nav_selection == "2. Sistemas Lineales (Gauss-Seidel)":
    st.header("⛓️ Gauss-Seidel")
    c1, c2, c3, c4 = st.columns(4)
    a11=c1.number_input("a11", key="m_a11"); a12=c2.number_input("a12", key="m_a12"); a13=c3.number_input("a13", key="m_a13"); b1=c4.number_input("b1", key="m_b1")
    a21=c1.number_input("a21", key="m_a21"); a22=c2.number_input("a22", key="m_a22"); a23=c3.number_input("a23", key="m_a23"); b2=c4.number_input("b2", key="m_b2")
    a31=c1.number_input("a31", key="m_a31"); a32=c2.number_input("a32", key="m_a32"); a33=c3.number_input("a33", key="m_a33"); b3=c4.number_input("b3", key="m_b3")
    iter_n = st.number_input("Iteración a verificar:", value=3)
    
    if st.button("Calcular"):
        x1=0; x2=0; x3=0
        for k in range(int(iter_n)):
            x1=(b1-a12*x2-a13*x3)/a11
            x2=(b2-a21*x1-a23*x3)/a22
            x3=(b3-a31*x1-a32*x2)/a33
        st.write(f"Iteración {iter_n}: {x1:.4f}, {x2:.4f}, {x3:.4f}")
        
        opciones = st.session_state.opciones_detectadas
        if opciones:
            st.divider()
            found = False
            for l, v in opciones.items():
                if any(abs(val - float(v)) < 0.1 for val in [x1, x2, x3]):
                    st.success(f"✅ Opción **{l.upper()}** ({v}) coincide.")
                    found = True
            if not found: st.warning("Sin coincidencias exactas.")

elif st.session_state.nav_selection == "3. Integración (Simpson/Gauss)":
    st.header("∫ Integración")
    f_txt = st.text_input("f(x):", key="int_func")
    c1, c2 = st.columns(2)
    a_v = c1.number_input("a:", key="int_a"); b_v = c2.number_input("b:", key="int_b")
    met = st.selectbox("Método", ["Gauss-Legendre (2pt)", "Simpson 1/3"])
    
    if st.button("Calcular"):
        try:
            x=sp.symbols('x'); f=sp.lambdify(x, sp.sympify(f_txt), "numpy")
            if "Gauss" in met:
                t=0.577350269; dx=(b_v-a_v)/2; avg=(b_v+a_v)/2
                res = dx*(f(avg-dx*t) + f(avg+dx*t))
            else:
                xv=np.linspace(a_v, b_v, 11); yv=f(xv); h=(b_v-a_v)/10
                res = (h/3)*(yv[0]+yv[-1]+4*sum(yv[1:-1:2])+2*sum(yv[2:-2:2]))
            mostrar_inciso_correcto(res)
        except Exception as e: st.error(e)

elif st.session_state.nav_selection == "4. Derivación Numérica":
    st.header("∂ Derivación")
    f_d = st.text_input("f(x):", key="der_func")
    c1, c2 = st.columns(2)
    xi = c1.number_input("x:", key="der_x"); h = c2.number_input("h:", key="der_h")
    
    if st.button("Calcular"):
        try:
            x=sp.symbols('x'); f=sp.lambdify(x, sp.sympify(f_d), "numpy")
            r1 = (f(xi+h)-f(xi-h))/(2*h)
            r2 = (f(xi+h)-2*f(xi)+f(xi-h))/(h**2)
            st.write(f"1ra: {r1:.5f} | 2da: {r2:.5f}")
            
            opciones = st.session_state.opciones_detectadas
            if opciones:
                st.divider()
                match = False
                for l, v in opciones.items():
                    if abs(r1 - float(v)) < 0.05:
                        st.success(f"✅ Opción **{l.upper()}** es correcta (1ra Derivada)."); match=True
                    elif abs(r2 - float(v)) < 0.05:
                        st.success(f"✅ Opción **{l.upper()}** es correcta (2da Derivada)."); match=True
                if not match: st.warning("Sin coincidencias.")
        except Exception as e: st.error(e)