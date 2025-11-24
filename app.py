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

# Función para inicializar variables de sesión si no existen
def init_session():
    defaults = {
        "nav_selection": "1. Raíces (Bisección/Newton)", # Página actual
        # Datos Bisección
        "bis_func": "0.95*x**3 - 5.9*x**2 + 10.9*x - 6",
        "bis_a": 3.0,
        "bis_b": 4.0,
        # Datos Newton
        "newton_func": "0.95*x**3 - 5.9*x**2 + 10.9*x - 6",
        "newton_x0": 3.5,
        # Datos Sistemas (Gauss)
        "m_a11": 5.0, "m_a12": 1.0, "m_a13": -15.0, "m_b1": 5.0,
        "m_a21": 13.0, "m_a22": 3.0, "m_a23": 9.0, "m_b2": 30.0,
        "m_a31": 1.0, "m_a32": 4.0, "m_a33": 2.0, "m_b3": 15.0,
        # Datos Integración
        "int_func": "(5*x**3 + x) / sqrt(3*x**2 + 5)",
        "int_a": 1.0, "int_b": 5.0,
        # Datos Derivada
        "der_func": "sqrt(5*x**3 + 1)",
        "der_x": 1.3, "der_h": 0.1
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# --- 2. LÓGICA DE IA (DETECTIVE DE DATOS) ---
def analizar_imagen_con_ia(api_key, image):
    genai.configure(api_key=api_key)
    # Usamos flash-latest para rapidez
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = """
    Analiza esta imagen matemática. Tu trabajo es EXTRAER los datos para un programa en Python.
    Devuelve SOLAMENTE un JSON válido (sin texto extra).
    
    Reglas de formato:
    1. Detecta el método: "biseccion", "newton", "gauss_seidel", "integracion", "derivada".
    2. Convierte fórmulas a sintaxis Python: 
       - Potencias: x^3 -> x**3
       - Raíces: √x -> sqrt(x)
       - Seno/Coseno: sin(x) -> np.sin(x)
       - Euler: e^x -> np.exp(x)
    
    Estructura JSON requerida según el método detectado:
    - Si es Bisección: {"metodo": "biseccion", "func": "...", "a": float, "b": float}
    - Si es Newton: {"metodo": "newton", "func": "...", "x0": float}
    - Si es Sistemas 3x3: {"metodo": "gauss_seidel", "matrix": [[a11, a12, a13, b1], [a21, a22, a23, b2], [a31, a32, a33, b3]]}
    - Si es Integración: {"metodo": "integracion", "func": "...", "a": float, "b": float}
    - Si es Derivada: {"metodo": "derivada", "func": "...", "x": float, "h": float}
    """
    
    try:
        response = model.generate_content([prompt, image])
        # Limpiar respuesta para obtener solo JSON
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        st.sidebar.error(f"Error al procesar: {e}")
        return None

# --- 3. BARRA LATERAL (SCANNER Y MENÚ) ---
with st.sidebar:
    st.title("🤖 Auto-Scanner")
    
    # API Key
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API Key Conectada")
    else:
        api_key = st.text_input("API Key Google", type="password")

    # Uploader Inteligente
    uploaded_file = st.file_uploader("📷 Escanear Ejercicio", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and api_key:
        if st.button("🚀 Extraer Datos y Llenar"):
            with st.spinner("Leyendo examen..."):
                img = Image.open(uploaded_file)
                datos = analizar_imagen_con_ia(api_key, img)
                
                if datos:
                    try:
                        metodo = datos.get("metodo")
                        
                        if metodo == "biseccion":
                            st.session_state.nav_selection = "1. Raíces (Bisección/Newton)"
                            st.session_state.bis_func = datos['func']
                            st.session_state.bis_a = float(datos['a'])
                            st.session_state.bis_b = float(datos['b'])
                            st.toast("✅ Datos de Bisección cargados!")
                            
                        elif metodo == "newton":
                            st.session_state.nav_selection = "1. Raíces (Bisección/Newton)"
                            st.session_state.newton_func = datos['func']
                            st.session_state.newton_x0 = float(datos['x0'])
                            st.toast("✅ Datos de Newton cargados!")

                        elif metodo == "gauss_seidel":
                            st.session_state.nav_selection = "2. Sistemas Lineales (Gauss-Seidel)"
                            m = datos['matrix']
                            # Mapear matriz a variables planas
                            st.session_state.m_a11 = float(m[0][0]); st.session_state.m_a12 = float(m[0][1]); st.session_state.m_a13 = float(m[0][2]); st.session_state.m_b1 = float(m[0][3])
                            st.session_state.m_a21 = float(m[1][0]); st.session_state.m_a22 = float(m[1][1]); st.session_state.m_a23 = float(m[1][2]); st.session_state.m_b2 = float(m[1][3])
                            st.session_state.m_a31 = float(m[2][0]); st.session_state.m_a32 = float(m[2][1]); st.session_state.m_a33 = float(m[2][2]); st.session_state.m_b3 = float(m[2][3])
                            st.toast("✅ Matriz cargada!")

                        elif metodo == "integracion":
                            st.session_state.nav_selection = "3. Integración (Simpson/Gauss)"
                            st.session_state.int_func = datos['func']
                            st.session_state.int_a = float(datos['a'])
                            st.session_state.int_b = float(datos['b'])
                            st.toast("✅ Integral cargada!")

                        elif metodo == "derivada":
                            st.session_state.nav_selection = "4. Derivación Numérica"
                            st.session_state.der_func = datos['func']
                            st.session_state.der_x = float(datos['x'])
                            st.session_state.der_h = float(datos['h'])
                            st.toast("✅ Derivada cargada!")
                            
                        st.rerun() # Recargar página con nuevos datos
                        
                    except Exception as e:
                        st.error(f"La IA leyó mal los datos: {e}")

    st.divider()
    
    # Navegación Manual (conectada al session_state)
    seccion = st.radio(
        "Ir a:",
        ["1. Raíces (Bisección/Newton)", 
         "2. Sistemas Lineales (Gauss-Seidel)", 
         "3. Integración (Simpson/Gauss)", 
         "4. Derivación Numérica"],
        key="nav_selection"
    )

# --- 4. CONTENIDO PRINCIPAL (CALCULADORAS) ---

# ==========================================
# UNIDAD 1: RAÍCES
# ==========================================
if st.session_state.nav_selection == "1. Raíces (Bisección/Newton)":
    st.header("🔍 Raíces de Ecuaciones")
    tabs = st.tabs(["Bisección", "Newton-Raphson"])
    
    with tabs[0]: # BISECCIÓN
        c1, c2 = st.columns(2)
        # Usamos KEY para vincular con session_state
        f_txt = c1.text_input("Función f(x):", key="bis_func")
        a_val = c2.number_input("Límite a:", key="bis_a")
        b_val = c2.number_input("Límite b:", key="bis_b")
        tol = st.number_input("Tolerancia:", value=0.0001, format="%.5f")
        
        if st.button("Calcular Bisección"):
            try:
                x = sp.symbols('x')
                f = sp.lambdify(x, sp.sympify(f_txt), "numpy") # Soporte para np.
                if f(a_val)*f(b_val) >= 0: st.error("Error: Mismo signo en los límites.")
                else:
                    datos = []
                    xr_ant = 0; a, b = a_val, b_val
                    for i in range(20):
                        xr = (a + b) / 2
                        fxr = f(xr)
                        err = abs((xr - xr_ant)/xr)*100 if i>0 else 100
                        datos.append({"Iter": i, "a": a, "b": b, "xr": xr, "f(xr)": fxr, "Error %": err})
                        if abs(fxr) < tol: break
                        if f(a)*fxr < 0: b = xr
                        else: a = xr
                        xr_ant = xr
                    st.dataframe(pd.DataFrame(datos).style.format(precision=5))
            except Exception as e: st.error(f"Error: {e}")

    with tabs[1]: # NEWTON
        c1, c2 = st.columns(2)
        f_new = c1.text_input("Función f(x):", key="newton_func")
        x0_val = c2.number_input("Valor inicial x0:", key="newton_x0")
        
        if st.button("Calcular Newton"):
            try:
                x = sp.symbols('x')
                f_expr = sp.sympify(f_new)
                df_expr = sp.diff(f_expr, x)
                f = sp.lambdify(x, f_expr); df = sp.lambdify(x, df_expr)
                
                datos = []
                xi = x0_val
                for i in range(10):
                    fxi = f(xi); dfxi = df(xi)
                    if dfxi == 0: st.error("Derivada 0"); break
                    xi_new = xi - (fxi/dfxi)
                    err = abs((xi_new - xi)/xi_new)*100 if i>0 else 100
                    datos.append({"Iter": i, "xi": xi, "f(xi)": fxi, "f'(xi)": dfxi, "Error %": err})
                    xi = xi_new
                st.dataframe(pd.DataFrame(datos).style.format(precision=5))
            except Exception as e: st.error(e)

# ==========================================
# UNIDAD 2: SISTEMAS (GAUSS)
# ==========================================
elif st.session_state.nav_selection == "2. Sistemas Lineales (Gauss-Seidel)":
    st.header("⛓️ Sistemas 3x3 (Gauss-Seidel)")
    st.info("La IA llena estos campos automáticamente si subes la foto.")
    
    c1, c2, c3, c4 = st.columns(4)
    # Inputs vinculados a SESSION STATE
    a11 = c1.number_input("a11", key="m_a11"); a12 = c2.number_input("a12", key="m_a12"); a13 = c3.number_input("a13", key="m_a13"); b1 = c4.number_input("b1", key="m_b1")
    a21 = c1.number_input("a21", key="m_a21"); a22 = c2.number_input("a22", key="m_a22"); a23 = c3.number_input("a23", key="m_a23"); b2 = c4.number_input("b2", key="m_b2")
    a31 = c1.number_input("a31", key="m_a31"); a32 = c2.number_input("a32", key="m_a32"); a33 = c3.number_input("a33", key="m_a33"); b3 = c4.number_input("b3", key="m_b3")
    
    if st.button("Iterar Gauss-Seidel"):
        x1, x2, x3 = 0.0, 0.0, 0.0
        res = []
        for k in range(5):
            try:
                x1 = (b1 - a12*x2 - a13*x3) / a11
                x2 = (b2 - a21*x1 - a23*x3) / a22
                x3 = (b3 - a31*x1 - a32*x2) / a33
                res.append({"Iter": k+1, "x1": x1, "x2": x2, "x3": x3})
            except: st.error("División por cero"); break
        st.dataframe(pd.DataFrame(res).style.format(precision=4))

# ==========================================
# UNIDAD 3: INTEGRACIÓN
# ==========================================
elif st.session_state.nav_selection == "3. Integración (Simpson/Gauss)":
    st.header("∫ Integración")
    f_int = st.text_input("Función:", key="int_func")
    c1, c2 = st.columns(2)
    a_int = c1.number_input("Límite a:", key="int_a")
    b_int = c2.number_input("Límite b:", key="int_b")
    
    metodo = st.selectbox("Método", ["Gauss-Legendre (2 puntos)", "Simpson 1/3"])
    
    if st.button("Calcular Área"):
        try:
            x = sp.symbols('x')
            f = sp.lambdify(x, sp.sympify(f_int), "numpy")
            
            if metodo == "Gauss-Legendre (2 puntos)":
                t0, t1 = -0.577350269, 0.577350269
                dx = (b_int - a_int)/2; avg = (b_int + a_int)/2
                res = dx * (f(dx*t0 + avg) + f(dx*t1 + avg))
                st.success(f"Resultado Gauss: {res:.6f}")
                
            elif metodo == "Simpson 1/3":
                n=10; h = (b_int - a_int)/n
                x_vals = np.linspace(a_int, b_int, n+1); y_vals = f(x_vals)
                res = (h/3) * (y_vals[0] + y_vals[-1] + 4*sum(y_vals[1:-1:2]) + 2*sum(y_vals[2:-2:2]))
                st.success(f"Resultado Simpson: {res:.6f}")
        except Exception as e: st.error(e)

# ==========================================
# UNIDAD 4: DERIVADA
# ==========================================
elif st.session_state.nav_selection == "4. Derivación Numérica":
    st.header("∂ Derivación")
    f_der = st.text_input("Función:", key="der_func")
    c1, c2 = st.columns(2)
    xi = c1.number_input("Punto x:", key="der_x")
    h = c2.number_input("Paso h:", key="der_h")
    tipo = st.selectbox("Orden", ["2da Derivada (Central)", "1ra Derivada (Central)"])
    
    if st.button("Calcular"):
        try:
            x = sp.symbols('x')
            f = sp.lambdify(x, sp.sympify(f_der), "numpy")
            if "2da" in tipo:
                res = (f(xi+h) - 2*f(xi) + f(xi-h))/(h**2)
            else:
                res = (f(xi+h) - f(xi-h))/(2*h)
            st.metric("Resultado", f"{res:.5f}")
        except Exception as e: st.error(e)