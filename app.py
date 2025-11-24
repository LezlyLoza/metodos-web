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
        # Variables de datos
        "bis_func": "0.95*x**3 - 5.9*x**2 + 10.9*x - 6", "bis_a": 3.0, "bis_b": 4.0,
        "newton_func": "0.95*x**3 - 5.9*x**2 + 10.9*x - 6", "newton_x0": 3.5,
        "m_a11": 5.0, "m_a12": 1.0, "m_a13": -15.0, "m_b1": 5.0,
        "m_a21": 13.0, "m_a22": 3.0, "m_a23": 9.0, "m_b2": 30.0,
        "m_a31": 1.0, "m_a32": 4.0, "m_a33": 2.0, "m_b3": 15.0,
        "int_func": "(5*x**3 + x) / sqrt(3*x**2 + 5)", "int_a": 1.0, "int_b": 5.0,
        "der_func": "sqrt(5*x**3 + 1)", "der_x": 1.3, "der_h": 0.1,
        # NUEVO: Guardar las opciones detectadas (a, b, c) de la foto
        "opciones_detectadas": {} 
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# --- FUNCIÓN DE COMPARACIÓN INTELIGENTE ---
def mostrar_inciso_correcto(resultado_calculado):
    opciones = st.session_state.opciones_detectadas
    if not opciones:
        return # No hay opciones para comparar
    
    st.divider()
    st.markdown("### 🎯 Veredicto de la IA + Python")
    
    mejor_opcion = None
    menor_diferencia = float('inf')
    
    # Buscar cuál opción está más cerca del resultado calculado
    for letra, valor in opciones.items():
        try:
            diff = abs(resultado_calculado - float(valor))
            if diff < menor_diferencia:
                menor_diferencia = diff
                mejor_opcion = letra
        except:
            pass # Ignorar si la opción no es número

    if mejor_opcion:
        # Mostrar en grande y verde
        st.success(f"✅ La respuesta correcta es la opción **{mejor_opcion.upper()}** ({opciones[mejor_opcion]})")
        st.caption(f"Calculado: {resultado_calculado:.6f} | Diferencia: {menor_diferencia:.6f}")
    else:
        st.warning("No pude coincidir el resultado con las opciones de la imagen.")

# --- 2. LÓGICA DE IA (DETECTIVE) ---
def analizar_imagen_con_ia(api_key, image):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = """
    Analiza esta imagen de examen.
    1. Extrae los datos matemáticos.
    2. IMPORTANTE: Extrae las opciones de respuesta (a, b, c, d) si existen.
    3. Devuelve SOLO JSON.
    
    Formato JSON:
    {
        "metodo": "biseccion" | "newton" | "gauss_seidel" | "integracion" | "derivada",
        "func": "ecuacion en python",
        ... (datos especificos del metodo),
        "opciones": {"a": 1.23, "b": 1.24, "c": 1.25}  <-- SOLO NUMEROS
    }
    
    Si es Bisección: {"metodo": "biseccion", "func": "...", "a": float, "b": float, "opciones": {...}}
    Si es Newton: {"metodo": "newton", "func": "...", "x0": float, "opciones": {...}}
    Si es Gauss: {"metodo": "gauss_seidel", "matrix": [...], "opciones": {...}}
    Si es Integración: {"metodo": "integracion", "func": "...", "a": float, "b": float, "opciones": {...}}
    Si es Derivada: {"metodo": "derivada", "func": "...", "x": float, "h": float, "opciones": {...}}
    """
    
    try:
        response = model.generate_content([prompt, image])
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        st.sidebar.error(f"Error IA: {e}")
        return None

# --- 3. BARRA LATERAL ---
with st.sidebar:
    st.title("🤖 Auto-Scanner Pro")
    
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API Key Conectada")
    else:
        api_key = st.text_input("API Key Google", type="password")

    uploaded_file = st.file_uploader("📷 Subir Examen", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and api_key:
        if st.button("🚀 Escanear y Preparar"):
            with st.spinner("Analizando opciones e inciso correcto..."):
                img = Image.open(uploaded_file)
                datos = analizar_imagen_con_ia(api_key, img)
                
                if datos:
                    try:
                        metodo = datos.get("metodo")
                        st.session_state.opciones_detectadas = datos.get("opciones", {})
                        
                        if metodo == "biseccion":
                            st.session_state.nav_selection = "1. Raíces (Bisección/Newton)"
                            st.session_state.bis_func = datos['func']
                            st.session_state.bis_a = float(datos['a'])
                            st.session_state.bis_b = float(datos['b'])
                            
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
                            st.session_state.int_a = float(datos['a'])
                            st.session_state.int_b = float(datos['b'])

                        elif metodo == "derivada":
                            st.session_state.nav_selection = "4. Derivación Numérica"
                            st.session_state.der_func = datos['func']
                            st.session_state.der_x = float(datos['x'])
                            st.session_state.der_h = float(datos['h'])
                            
                        st.toast("✅ ¡Datos y Opciones extraídos!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error procesando datos: {e}")

    st.divider()
    seccion = st.radio("Ir a:", ["1. Raíces (Bisección/Newton)", "2. Sistemas Lineales (Gauss-Seidel)", "3. Integración (Simpson/Gauss)", "4. Derivación Numérica"], key="nav_selection")

    # Mostrar las opciones que la IA vió (para verificar)
    if st.session_state.opciones_detectadas:
        st.info(f"Opciones detectadas en foto: {st.session_state.opciones_detectadas}")

# --- 4. CALCULADORAS ---

# UNIDAD 1: RAÍCES
if st.session_state.nav_selection == "1. Raíces (Bisección/Newton)":
    st.header("🔍 Raíces de Ecuaciones")
    tabs = st.tabs(["Bisección", "Newton-Raphson"])
    
    with tabs[0]: # BISECCIÓN
        f_txt = st.text_input("Función f(x):", key="bis_func")
        c1, c2 = st.columns(2)
        a_val = c1.number_input("Límite a:", key="bis_a")
        b_val = c2.number_input("Límite b:", key="bis_b")
        tol = st.number_input("Tolerancia:", value=0.0001, format="%.5f")
        
        if st.button("Calcular Bisección"):
            try:
                x = sp.symbols('x')
                f = sp.lambdify(x, sp.sympify(f_txt), "numpy")
                if f(a_val)*f(b_val) >= 0: st.error("Error: Mismo signo.")
                else:
                    xr_ant = 0; a, b = a_val, b_val
                    xr_final = 0
                    datos = []
                    for i in range(20):
                        xr = (a + b) / 2
                        fxr = f(xr)
                        datos.append({"Iter": i, "xr": xr, "f(xr)": fxr})
                        if abs(fxr) < tol: break
                        if f(a)*fxr < 0: b = xr
                        else: a = xr
                        xr_final = xr
                    
                    # MAGIA: MOSTRAR INCISO CORRECTO
                    mostrar_inciso_correcto(xr_final)
                    
                    st.dataframe(pd.DataFrame(datos))
            except Exception as e: st.error(e)

    with tabs[1]: # NEWTON
        f_new = st.text_input("Función f(x):", key="newton_func")
        x0_val = st.number_input("Valor inicial x0:", key="newton_x0")
        if st.button("Calcular Newton"):
            try:
                x = sp.symbols('x')
                f_expr = sp.sympify(f_new); df_expr = sp.diff(f_expr, x)
                f = sp.lambdify(x, f_expr); df = sp.lambdify(x, df_expr)
                xi = x0_val
                for i in range(10):
                    xi_new = xi - (f(xi)/df(xi))
                    xi = xi_new
                
                # MAGIA: MOSTRAR INCISO CORRECTO
                mostrar_inciso_correcto(xi)
                st.write(f"Raíz final: {xi}")
            except Exception as e: st.error(e)

# UNIDAD 2: SISTEMAS
elif st.session_state.nav_selection == "2. Sistemas Lineales (Gauss-Seidel)":
    st.header("⛓️ Gauss-Seidel")
    c1, c2, c3, c4 = st.columns(4)
    a11 = c1.number_input("a11", key="m_a11"); a12 = c2.number_input("a12", key="m_a12"); a13 = c3.number_input("a13", key="m_a13"); b1 = c4.number_input("b1", key="m_b1")
    a21 = c1.number_input("a21", key="m_a21"); a22 = c2.number_input("a22", key="m_a22"); a23 = c3.number_input("a23", key="m_a23"); b2 = c4.number_input("b2", key="m_b2")
    a31 = c1.number_input("a31", key="m_a31"); a32 = c2.number_input("a32", key="m_a32"); a33 = c3.number_input("a33", key="m_a33"); b3 = c4.number_input("b3", key="m_b3")
    
    iters = st.number_input("Iteraciones a comprobar:", value=3)

    if st.button("Calcular Gauss-Seidel"):
        x1, x2, x3 = 0.0, 0.0, 0.0
        for k in range(int(iters)):
            x1 = (b1 - a12*x2 - a13*x3) / a11
            x2 = (b2 - a21*x1 - a23*x3) / a22
            x3 = (b3 - a31*x1 - a32*x2) / a33
        
        # MAGIA: Comparamos x1 (puedes ajustar para comparar x2 o x3 si la pregunta es específica)
        st.write(f"Resultados iteración {iters}: x1={x1:.4f}, x2={x2:.4f}, x3={x3:.4f}")
        
        # Intentamos buscar coincidencia con cualquiera de los valores
        opciones = st.session_state.opciones_detectadas
        if opciones:
            st.divider()
            st.markdown("### 🎯 Coincidencias encontradas")
            found = False
            for let, val in opciones.items():
                # Revisa si la opción coincide con x1, x2 o x3
                if abs(x1 - val) < 0.1 or abs(x2 - val) < 0.1 or abs(x3 - val) < 0.1:
                    st.success(f"✅ La opción **{let.upper()}** ({val}) coincide con uno de los resultados.")
                    found = True
            if not found: st.warning("No encontré coincidencia exacta con las opciones.")

# UNIDAD 3: INTEGRACIÓN
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
            res = 0
            if metodo == "Gauss-Legendre (2 puntos)":
                t = 0.577350269; dx = (b_int - a_int)/2; avg = (b_int + a_int)/2
                res = dx * (f(dx*(-t) + avg) + f(dx*t + avg))
            else:
                h = (b_int - a_int)/10; xv = np.linspace(a_int, b_int, 11); yv = f(xv)
                res = (h/3) * (yv[0] + yv[-1] + 4*sum(yv[1:-1:2]) + 2*sum(yv[2:-2:2]))
            
            # MAGIA: MOSTRAR INCISO CORRECTO
            mostrar_inciso_correcto(res)
            
        except Exception as e: st.error(e)

# UNIDAD 4: DERIVADA
elif st.session_state.nav_selection == "4. Derivación Numérica":
    st.header("∂ Derivación")
    f_der = st.text_input("Función:", key="der_func")
    c1, c2 = st.columns(2)
    xi = c1.number_input("Punto x:", key="der_x")
    h = c2.number_input("Paso h:", key="der_h")
    
    if st.button("Calcular Derivada"):
        try:
            x = sp.symbols('x')
            f = sp.lambdify(x, sp.sympify(f_der), "numpy")
            # Calculamos ambas (1ra y 2da) para ver cuál pide el examen
            res1 = (f(xi+h) - f(xi-h))/(2*h)
            res2 = (f(xi+h) - 2*f(xi) + f(xi-h))/(h**2)
            
            st.write(f"1ra Derivada: {res1:.5f}")
            st.write(f"2da Derivada: {res2:.5f}")
            
            # MAGIA: Comparar con ambas
            opciones = st.session_state.opciones_detectadas
            if opciones:
                st.divider()
                match = False
                for let, val in opciones.items():
                    if abs(res1 - val) < 0.01:
                        st.success(f"✅ Opción **{let.upper()}** es la correcta (corresponde a 1ra Derivada).")
                        match = True
                    elif abs(res2 - val) < 0.01:
                        st.success(f"✅ Opción **{let.upper()}** es la correcta (corresponde a 2da Derivada).")
                        match = True
                if not match: st.warning("No encontré coincidencias.")
                
        except Exception as e: st.error(e)