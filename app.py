import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import google.generativeai as genai
from PIL import Image
import json
import re

# --- 1. CONFIGURACIÓN INICIAL Y STATE ---
st.set_page_config(page_title="Metodos Numéricos Auto-Fill", layout="wide", page_icon="⚡")

# Función para inicializar variables de sesión
def init_session():
    defaults = {
        "nav_selection": "1. Raíces (Bisección/Newton)",
        # Datos Bisección
        "bis_func": "0.95*x**3 - 5.9*x**2 + 10.9*x - 6",
        "bis_a": 3.0, "bis_b": 4.0,
        # Datos Newton
        "newton_func": "0.95*x**3 - 5.9*x**2 + 10.9*x - 6", "newton_x0": 3.5,
        # Datos Sistemas
        "m_a11": 5.0, "m_a12": 1.0, "m_a13": -15.0, "m_b1": 5.0,
        "m_a21": 13.0, "m_a22": 3.0, "m_a23": 9.0, "m_b2": 30.0,
        "m_a31": 1.0, "m_a32": 4.0, "m_a33": 2.0, "m_b3": 15.0,
        # Datos Integración
        "int_func": "(5*x**3 + x) / sqrt(3*x**2 + 5)", "int_a": 1.0, "int_b": 5.0,
        # Datos Derivada
        "der_func": "sqrt(5*x**3 + 1)", "der_x": 1.3, "der_h": 0.1
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# --- 2. LÓGICA DE IA (Usando Gemini 2.5 Flash) ---
def analizar_imagen_con_ia(api_key, image):
    try:
        genai.configure(api_key=api_key)
        
        # --- AQUÍ ESTÁ EL CAMBIO CLAVE: USAMOS TU MODELO DISPONIBLE ---
        model = genai.GenerativeModel('gemini-2.5-flash') 
        # --------------------------------------------------------------
        
        prompt = """
        Eres un asistente matemático para un examen de ingeniería.
        Tu tarea es EXTRAER los datos de la imagen para llenar un formulario.
        Responde ÚNICAMENTE con un JSON válido. Sin texto antes ni después.
        
        Reglas de conversión:
        - Detecta el método: "biseccion", "newton", "gauss_seidel", "integracion", "derivada".
        - Matemáticas a Python: x^2 -> x**2, sen(x) -> np.sin(x), raíz(x) -> sqrt(x).
        
        Formatos JSON esperados:
        - Bisección: {"metodo": "biseccion", "func": "...", "a": 3.0, "b": 4.0}
        - Newton: {"metodo": "newton", "func": "...", "x0": 3.5}
        - Gauss-Seidel (Matriz 3x3): {"metodo": "gauss_seidel", "matrix": [[a11, a12, a13, b1], [a21, a22, a23, b2], [a31, a32, a33, b3]]}
        - Integración: {"metodo": "integracion", "func": "...", "a": 1.0, "b": 5.0}
        - Derivada: {"metodo": "derivada", "func": "...", "x": 1.3, "h": 0.1}
        """
        
        response = model.generate_content([prompt, image])
        
        # Limpieza agresiva del JSON por si la IA habla de más
        texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpio)
        
    except Exception as e:
        st.sidebar.error(f"Error de lectura IA: {e}")
        return None

# --- 3. BARRA LATERAL (SCANNER) ---
with st.sidebar:
    st.title("🤖 Auto-Scanner 2.5")
    
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ Conectado a Gemini 2.5")
    else:
        api_key = st.text_input("API Key", type="password")

    uploaded_file = st.file_uploader("📷 Subir Foto Examen", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and api_key:
        if st.button("🚀 Escanear y Llenar"):
            with st.spinner("Analizando con Gemini 2.5 Flash..."):
                img = Image.open(uploaded_file)
                datos = analizar_imagen_con_ia(api_key, img)
                
                if datos:
                    try:
                        metodo = datos.get("metodo")
                        
                        if metodo == "biseccion":
                            st.session_state.nav_selection = "1. Raíces (Bisección/Newton)"
                            st.session_state.bis_func = datos['func']
                            st.session_state.bis_a = float(datos['a']); st.session_state.bis_b = float(datos['b'])
                            
                        elif metodo == "newton":
                            st.session_state.nav_selection = "1. Raíces (Bisección/Newton)"
                            st.session_state.newton_func = datos['func']
                            st.session_state.newton_x0 = float(datos['x0'])

                        elif metodo == "gauss_seidel":
                            st.session_state.nav_selection = "2. Sistemas Lineales (Gauss-Seidel)"
                            m = datos['matrix']
                            st.session_state.m_a11 = float(m[0][0]); st.session_state.m_a12 = float(m[0][1]); st.session_state.m_a13 = float(m[0][2]); st.session_state.m_b1 = float(m[0][3])
                            st.session_state.m_a21 = float(m[1][0]); st.session_state.m_a22 = float(m[1][1]); st.session_state.m_a23 = float(m[1][2]); st.session_state.m_b2 = float(m[1][3])
                            st.session_state.m_a31 = float(m[2][0]); st.session_state.m_a32 = float(m[2][1]); st.session_state.m_a33 = float(m[2][2]); st.session_state.m_b3 = float(m[2][3])

                        elif metodo == "integracion":
                            st.session_state.nav_selection = "3. Integración (Simpson/Gauss)"
                            st.session_state.int_func = datos['func']
                            st.session_state.int_a = float(datos['a']); st.session_state.int_b = float(datos['b'])

                        elif metodo == "derivada":
                            st.session_state.nav_selection = "4. Derivación Numérica"
                            st.session_state.der_func = datos['func']
                            st.session_state.der_x = float(datos['x']); st.session_state.der_h = float(datos['h'])
                            
                        st.success("✅ ¡Datos detectados! Revisa los campos.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error interpretando datos: {e}")

    st.divider()
    seccion = st.radio("Navegación Manual:", 
        ["1. Raíces (Bisección/Newton)", "2. Sistemas Lineales (Gauss-Seidel)", 
         "3. Integración (Simpson/Gauss)", "4. Derivación Numérica"],
        key="nav_selection")

# --- 4. CALCULADORAS ---

# UNIDAD 1: RAÍCES
if st.session_state.nav_selection == "1. Raíces (Bisección/Newton)":
    st.header("🔍 Raíces")
    tabs = st.tabs(["Bisección", "Newton"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        f_txt = c1.text_input("f(x):", key="bis_func")
        a = c2.number_input("a:", key="bis_a"); b = c2.number_input("b:", key="bis_b")
        if st.button("Calcular Bisección"):
            try:
                x = sp.symbols('x'); f = sp.lambdify(x, sp.sympify(f_txt), "numpy")
                if f(a)*f(b)>=0: st.error("Mismo signo en límites")
                else:
                    res=[]; xr_ant=0
                    for i in range(20):
                        xr=(a+b)/2; fxr=f(xr); err=abs((xr-xr_ant)/xr)*100 if i>0 else 100
                        res.append({"Iter":i, "a":a, "b":b, "xr":xr, "f(xr)":fxr, "Err %":err})
                        if abs(fxr)<0.0001: break
                        if f(a)*fxr<0: b=xr
                        else: a=xr
                        xr_ant=xr
                    st.dataframe(pd.DataFrame(res))
            except Exception as e: st.error(e)
    with tabs[1]:
        c1, c2 = st.columns(2)
        fn = c1.text_input("f(x):", key="newton_func")
        x0 = c2.number_input("x0:", key="newton_x0")
        if st.button("Calcular Newton"):
            try:
                x=sp.symbols('x'); f_e=sp.sympify(fn); f=sp.lambdify(x,f_e); df=sp.lambdify(x,sp.diff(f_e,x))
                res=[]; xi=x0
                for i in range(10):
                    fxi=f(xi); dfxi=df(xi)
                    if dfxi==0: break
                    xn=xi-(fxi/dfxi); err=abs((xn-xi)/xn)*100 if i>0 else 100
                    res.append({"Iter":i, "xi":xi, "f(xi)":fxi, "Err %":err})
                    xi=xn
                st.dataframe(pd.DataFrame(res))
            except Exception as e: st.error(e)

# UNIDAD 2: GAUSS-SEIDEL
elif st.session_state.nav_selection == "2. Sistemas Lineales (Gauss-Seidel)":
    st.header("⛓️ Gauss-Seidel 3x3")
    c1, c2, c3, c4 = st.columns(4)
    a11=c1.number_input("a11", key="m_a11"); a12=c2.number_input("a12", key="m_a12"); a13=c3.number_input("a13", key="m_a13"); b1=c4.number_input("b1", key="m_b1")
    a21=c1.number_input("a21", key="m_a21"); a22=c2.number_input("a22", key="m_a22"); a23=c3.number_input("a23", key="m_a23"); b2=c4.number_input("b2", key="m_b2")
    a31=c1.number_input("a31", key="m_a31"); a32=c2.number_input("a32", key="m_a32"); a33=c3.number_input("a33", key="m_a33"); b3=c4.number_input("b3", key="m_b3")
    
    if st.button("Iterar"):
        x1,x2,x3=0.0,0.0,0.0; res=[]
        for k in range(5):
            try:
                x1=(b1-a12*x2-a13*x3)/a11; x2=(b2-a21*x1-a23*x3)/a22; x3=(b3-a31*x1-a32*x2)/a33
                res.append({"Iter":k+1, "x1":x1, "x2":x2, "x3":x3})
            except: pass
        st.dataframe(pd.DataFrame(res))

# UNIDAD 3: INTEGRACIÓN
elif st.session_state.nav_selection == "3. Integración (Simpson/Gauss)":
    st.header("∫ Integración")
    fi = st.text_input("f(x):", key="int_func")
    c1, c2 = st.columns(2)
    ai = c1.number_input("a:", key="int_a"); bi = c2.number_input("b:", key="int_b")
    met = st.selectbox("Método", ["Gauss-Legendre (2pt)", "Simpson 1/3"])
    if st.button("Integrar"):
        try:
            x=sp.symbols('x'); f=sp.lambdify(x,sp.sympify(fi),"numpy")
            if "Gauss" in met:
                t=0.577350269; dx=(bi-ai)/2; avg=(bi+ai)/2
                st.success(f"Resultado: {dx*(f(avg-dx*t)+f(avg+dx*t)):.6f}")
            else:
                h=(bi-ai)/10; xv=np.linspace(ai,bi,11); yv=f(xv)
                res=(h/3)*(yv[0]+yv[-1]+4*sum(yv[1:-1:2])+2*sum(yv[2:-2:2]))
                st.success(f"Resultado: {res:.6f}")
        except Exception as e: st.error(e)

# UNIDAD 4: DERIVADA
elif st.session_state.nav_selection == "4. Derivación Numérica":
    st.header("∂ Derivada")
    fd = st.text_input("f(x):", key="der_func")
    c1, c2 = st.columns(2)
    xd = c1.number_input("x:", key="der_x"); hd = c2.number_input("h:", key="der_h")
    if st.button("Derivar"):
        try:
            x=sp.symbols('x'); f=sp.lambdify(x,sp.sympify(fd),"numpy")
            st.metric("Resultado", f"{(f(xd+hd)-f(xd-hd))/(2*hd):.5f}")
        except Exception as e: st.error(e)