import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import google.generativeai as genai
from PIL import Image

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Metodos Numéricos 2025", layout="wide", page_icon="🧮")
st.title("🧮 Suite de Métodos Numéricos - Examen Final")
st.markdown("Selecciona el tipo de ejercicio en el menú de la izquierda.")

# --- BARRA LATERAL (MENÚ) ---
seccion = st.sidebar.radio(
    "Ir a la Unidad:",
    ["1. Raíces (Bisección/Newton)", 
     "2. Sistemas Lineales (Gauss-Seidel)", 
     "3. Integración (Simpson/Gauss)", 
     "4. Derivación Numérica",
     "5. Series de Taylor",
     "📸 Resolver con Foto (IA)"]
)

# ==========================================
# UNIDAD 1: RAÍCES DE ECUACIONES
# ==========================================
if seccion == "1. Raíces (Bisección/Newton)":
    st.header("🔍 Raíces de Ecuaciones")
    metodo = st.selectbox("Método", ["Método de Bisección", "Newton-Raphson"])

    col1, col2 = st.columns(2)
    with col1:
        funcion_str = st.text_input("Función f(x):", "0.95*x**3 - 5.9*x**2 + 10.9*x - 6")
        st.caption("Nota: Usa ** para potencias (ej: x**2)")
    
    if metodo == "Método de Bisección":
        with col2:
            a = st.number_input("Límite a:", value=3.0)
            b = st.number_input("Límite b:", value=4.0)
        tol = st.number_input("Tolerancia:", value=0.0001, format="%.5f")
        
        if st.button("Calcular Bisección"):
            try:
                x_sym = sp.symbols('x')
                f = sp.lambdify(x_sym, sp.sympify(funcion_str))
                
                if f(a)*f(b) >= 0:
                    st.error("⚠️ Error: f(a) y f(b) tienen el mismo signo. El método no garantiza convergencia.")
                else:
                    datos = []
                    xr_ant = 0
                    for i in range(50):
                        xr = (a + b) / 2
                        fxr = f(xr)
                        error = abs((xr - xr_ant)/xr)*100 if i > 0 else 100
                        datos.append({"Iter": i, "a": a, "b": b, "xr": xr, "f(xr)": fxr, "Error %": error})
                        
                        if abs(fxr) < tol: break
                        if f(a)*fxr < 0: b = xr
                        else: a = xr
                        xr_ant = xr
                    
                    st.dataframe(pd.DataFrame(datos).style.format(precision=5))
                    st.success(f"Raíz aproximada: {xr:.5f}")
            except Exception as e: st.error(f"Error: {e}")

    elif metodo == "Newton-Raphson":
        with col2:
            x0 = st.number_input("Valor inicial (x0):", value=3.5)
        tol = st.number_input("Tolerancia:", value=0.0001, format="%.5f")
        
        if st.button("Calcular Newton-Raphson"):
            try:
                x_sym = sp.symbols('x')
                f_expr = sp.sympify(funcion_str)
                df_expr = sp.diff(f_expr, x_sym) # Derivada automática
                f = sp.lambdify(x_sym, f_expr)
                df = sp.lambdify(x_sym, df_expr)
                
                datos = []
                xi = x0
                for i in range(20):
                    fxi = f(xi)
                    dfxi = df(xi)
                    if dfxi == 0: 
                        st.error("Derivada cero. El método falla."); break
                    
                    xi_new = xi - (fxi/dfxi)
                    error = abs((xi_new - xi)/xi_new)*100 if i > 0 else 100
                    
                    datos.append({"Iter": i, "xi": xi, "f(xi)": fxi, "f'(xi)": dfxi, "Error %": error})
                    xi = xi_new
                    if error < tol: break
                
                st.info(f"Derivada calculada: {df_expr}")
                st.dataframe(pd.DataFrame(datos).style.format(precision=5))
                st.success(f"Raíz aproximada: {xi:.5f}")
            except Exception as e: st.error(f"Error: {e}")

# ==========================================
# UNIDAD 2: SISTEMAS LINEALES
# ==========================================
elif seccion == "2. Sistemas Lineales (Gauss-Seidel)":
    st.header("⛓️ Sistemas de Ecuaciones (Gauss-Seidel)")
    st.markdown("Resuelve sistemas de 3x3. **Nota:** Verifica si es diagonalmente dominante.")
    
    st.write("Ingrese los coeficientes de la Matriz A y el vector B:")
    c1, c2, c3, c4 = st.columns(4)
    # Fila 1
    a11 = c1.number_input("a11", 5.0); a12 = c2.number_input("a12", 1.0); a13 = c3.number_input("a13", -15.0); b1 = c4.number_input("b1", 5.0)
    # Fila 2
    a21 = c1.number_input("a21", 13.0); a22 = c2.number_input("a22", 3.0); a23 = c3.number_input("a23", 9.0); b2 = c4.number_input("b2", 30.0)
    # Fila 3
    a31 = c1.number_input("a31", 1.0); a32 = c2.number_input("a32", 4.0); a33 = c3.number_input("a33", 2.0); b3 = c4.number_input("b3", 15.0)
    
    iters = st.slider("Número de iteraciones:", 1, 10, 3)
    
    if st.button("Iterar Gauss-Seidel"):
        # Valores iniciales
        x1, x2, x3 = 0.0, 0.0, 0.0
        resultados = []
        
        for k in range(iters):
            try:
                # Seidel actualiza inmediatamente los valores
                x1 = (b1 - a12*x2 - a13*x3) / a11
                x2 = (b2 - a21*x1 - a23*x3) / a22
                x3 = (b3 - a31*x1 - a32*x2) / a33
                
                resultados.append({
                    "Iter": k+1,
                    "x1": x1, "x2": x2, "x3": x3
                })
            except ZeroDivisionError:
                st.error("Error: División por cero en la diagonal. Reordena las filas para que los números mayores estén en la diagonal.")
                break
                
        st.dataframe(pd.DataFrame(resultados).style.format(precision=4))

# ==========================================
# UNIDAD 3: INTEGRACIÓN NUMÉRICA
# ==========================================
elif seccion == "3. Integración (Simpson/Gauss)":
    st.header("∫ Integración Numérica")
    metodo_int = st.selectbox("Método", ["Simpson 1/3", "Gauss-Legendre (2 puntos)"])
    
    func_int = st.text_input("Función a integrar:", "(5*x**3 + x) / sqrt(3*x**2 + 5)")
    st.caption("Recuerda usar `sqrt()` para raíces.")
    
    col_a, col_b = st.columns(2)
    a_int = col_a.number_input("Límite inferior (a):", 1.0)
    b_int = col_b.number_input("Límite superior (b):", 5.0)
    
    if metodo_int == "Simpson 1/3":
        n = st.number_input("Número de segmentos (n debe ser par):", value=10, step=2)
        if st.button("Calcular Área"):
            try:
                x = sp.symbols('x')
                f = sp.lambdify(x, sp.sympify(func_int), "numpy")
                
                h = (b_int - a_int) / n
                x_vals = np.linspace(a_int, b_int, n+1)
                y_vals = f(x_vals)
                
                suma = y_vals[0] + y_vals[-1] + 4*sum(y_vals[1:-1:2]) + 2*sum(y_vals[2:-2:2])
                resultado = (h/3) * suma
                
                st.success(f"Resultado Simpson 1/3: **{resultado:.6f}**")
            except Exception as e: st.error(e)
            
    elif metodo_int == "Gauss-Legendre (2 puntos)":
        st.caption("Usa la transformación estándar al intervalo [-1, 1]")
        if st.button("Calcular Gauss"):
            try:
                x_sym = sp.symbols('x')
                f_expr = sp.sympify(func_int)
                f = sp.lambdify(x_sym, f_expr, "numpy")
                
                # Constantes para 2 puntos
                c0, c1 = 1, 1
                t0, t1 = -0.577350269, 0.577350269
                
                # Transformación de variables
                dx = (b_int - a_int)/2
                avg = (b_int + a_int)/2
                
                x0_real = dx*t0 + avg
                x1_real = dx*t1 + avg
                
                val0 = f(x0_real)
                val1 = f(x1_real)
                
                integral = dx * (c0*val0 + c1*val1)
                
                st.write(f"t0 = -0.5774 -> x0 = {x0_real:.4f} -> f(x0) = {val0:.4f}")
                st.write(f"t1 = +0.5774 -> x1 = {x1_real:.4f} -> f(x1) = {val1:.4f}")
                st.success(f"Resultado Gauss-Legendre: **{integral:.6f}**")
            except Exception as e: st.error(e)

# ==========================================
# UNIDAD 4: DERIVACIÓN NUMÉRICA
# ==========================================
elif seccion == "4. Derivación Numérica":
    st.header("∂ Derivación Numérica")
    st.markdown("Calcula la derivada usando diferencias centrales.")
    
    func_der = st.text_input("Función:", "sqrt(5*x**3 + 1)")
    
    c1, c2, c3 = st.columns(3)
    xi_der = c1.number_input("Punto x:", 1.3)
    h_der = c2.number_input("Paso h:", 0.1)
    orden = c3.selectbox("Tipo", ["Primera Derivada (Central)", "Segunda Derivada (Central)"])
    
    if st.button("Calcular Derivada"):
        try:
            x = sp.symbols('x')
            f = sp.lambdify(x, sp.sympify(func_der), "numpy")
            
            val_x = f(xi_der)
            val_x_plus = f(xi_der + h_der)
            val_x_minus = f(xi_der - h_der)
            
            if orden == "Primera Derivada (Central)":
                res = (val_x_plus - val_x_minus) / (2*h_der)
                st.info("Fórmula: [f(x+h) - f(x-h)] / 2h")
                st.write(f"f(x+h) = {val_x_plus:.6f}")
                st.write(f"f(x-h) = {val_x_minus:.6f}")
            else:
                res = (val_x_plus - 2*val_x + val_x_minus) / (h_der**2)
                st.info("Fórmula: [f(x+h) - 2f(x) + f(x-h)] / h²")
                st.write(f"f(x+h) = {val_x_plus:.6f}")
                st.write(f"f(x)   = {val_x:.6f}")
                st.write(f"f(x-h) = {val_x_minus:.6f}")
            
            st.metric(label="Resultado Numérico", value=f"{res:.5f}")
            
        except Exception as e: st.error(e)

# ==========================================
# UNIDAD 5: SERIES DE TAYLOR
# ==========================================
elif seccion == "5. Series de Taylor":
    st.header("📈 Series de Taylor/Maclaurin")
    f_taylor = st.text_input("Función:", "6*x**5 - 3*x**3 + 2")
    n_terms = st.slider("Número de términos:", 1, 8, 3)
    
    if st.button("Generar Serie"):
        x = sp.symbols('x')
        try:
            expr = sp.sympify(f_taylor)
            serie = sp.series(expr, x, 0, n_terms).removeO()
            st.subheader("Polinomio Resultante:")
            st.latex(sp.latex(serie))
            st.write(f"Expresión plana: {serie}")
        except Exception as e: st.error(e)

# ==========================================
# NUEVA SECCIÓN: IA VISUAL (GEMINI)
# ==========================================
elif seccion == "📸 Resolver con Foto (IA)":
    st.header("🤖 Detective de Ejercicios")
    st.markdown("Sube una foto de tu examen y la IA detectará el método y lo resolverá.")
    
    # Campo para la API Key
    api_key = st.text_input("Pega tu Google API Key:", type="password", help="Obtén tu clave en aistudio.google.com")
    
    # Subir imagen
    uploaded_file = st.file_uploader("Sube la imagen del problema", type=["jpg", "png", "jpeg", "webp"])

    if uploaded_file is not None and api_key:
        try:
            # Configurar la IA
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            image = Image.open(uploaded_file)
            st.image(image, caption='Imagen subida', width=300)
            
            if st.button("🔍 Analizar y Resolver"):
                with st.spinner('La IA está pensando... (esto tarda unos segundos)'):
                    prompt = """
                    Actúa como un profesor experto en Métodos Numéricos.
                    1. Analiza la imagen y DETECTA qué método numérico se pide (Bisección, Newton, Gauss, Derivada, etc.).
                    2. Extrae los datos numéricos (función, límites, etc.).
                    3. RESUELVE el ejercicio paso a paso.
                    4. Si es un método iterativo, genera una TABLA en formato Markdown.
                    5. Dame el resultado final claro.
                    """
                    response = model.generate_content([prompt, image])
                    st.markdown(response.text)
                    
        except Exception as e:
            st.error(f"Error de conexión con la IA: {e}")
            st.warning("Verifica que tu API Key sea correcta y tenga permisos.")
    
    elif uploaded_file and not api_key:
        st.warning("⚠️ Necesitas ingresar tu API Key para procesar la imagen.")