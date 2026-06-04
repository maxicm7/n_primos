import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==============================================================================
# --- 1. MOTOR MATEMÁTICO OPTIMIZADO (Vectorizado con NumPy) ---
# ==============================================================================

@st.cache_data
def analizar_homeostasis_vectorizada(N: int):
    """
    Calcula la homeostasis del sistema numérico usando operaciones vectorizadas.
    Complejidad: O(N log log N) en lugar de O(N * sqrt(N)).
    """
    # 1. Criba de Eratóstenes vectorizada para encontrar primos
    sieve = np.ones(N + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if sieve[i]:
            sieve[i*i : N+1 : i] = False
            
    primes = np.where(sieve)[0]
    num_primos = len(primes)
    
    # 2. Cálculo vectorizado de la "Latencia" (Atraso)
    last_prime_idx = np.zeros(N + 1, dtype=int)
    last_prime_idx[primes] = primes
    
    # np.maximum.accumulate propaga el último primo visto hacia la derecha
    last_prime_seen = np.maximum.accumulate(last_prime_idx)
    
    # La latencia es la distancia desde el último primo
    latency = np.arange(N + 1) - last_prime_seen
    latency[primes] = 0 
    
    # 3. Métricas del Sistema
    total_atrasos = int(np.sum(latency))
    max_brecha = int(np.max(latency))
    avg_brecha = float(np.mean(latency[latency > 0])) if np.any(latency > 0) else 0.0
    densidad_primos = num_primos / N
    
    # 4. Construcción del DataFrame
    if N <= 5000:
        df = pd.DataFrame({
            "Número": np.arange(1, N + 1),
            "Tipo": np.where(sieve[1:], "MANIFESTACIÓN (n)", "LATENCIA (k)"),
            "Atraso_Individual": latency[1:],
            "Riesgo_Fallo": (latency[1:] ** 2) / 100
        })
    else:
        # Muestreo inteligente para datasets grandes
        step = max(1, N // 2000)
        indices = np.arange(1, N + 1, step)
        df = pd.DataFrame({
            "Número": indices,
            "Tipo": np.where(sieve[indices], "MANIFESTACIÓN (n)", "LATENCIA (k)"),
            "Atraso_Individual": latency[indices],
            "Riesgo_Fallo": (latency[indices] ** 2) / 100
        })

    return df, total_atrasos, num_primos, max_brecha, avg_brecha, densidad_primos, latency[1:]

# ==============================================================================
# --- 2. CONFIGURACIÓN Y ESTÉTICA STREAMLIT ---
# ==============================================================================

st.set_page_config(
    page_title="PIP-Riemann Homeostasis", 
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para un look "Científico/Futurista"
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #00d4ff; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.1rem; color: #a0a0a0; text-align: center; margin-bottom: 2rem;}
    .metric-card {background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #00d4ff;}
    div[data-testid="stMetricValue"] {font-size: 2rem;}
    /* Mejora visual para los tabs */
    button[data-baseweb="tab"] {font-size: 1rem; padding: 10px 20px;}
    button[data-baseweb="tab"][aria-selected="true"] {background-color: #00d4ff !important; color: #000 !important;}
    .formula-box {background-color: #2a2a2a; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffa500; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏛️ Protocolo de Ingeniería Probabilística: PIP-Riemann</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis de Homeostasis Estocástica y Válvulas de Escape en la Recta Numérica</div>', unsafe_allow_html=True)

# ==============================================================================
# --- 3. SIDEBAR (CONTROLES) ---
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Parámetros del Universo")
    
    st.subheader("📊 Análisis Numérico")
    N_input = st.slider(
        "Límite Superior (N)", 
        min_value=10, max_value=100000, value=1000, step=10,
        help="Límite de la recta numérica a analizar. Valores > 5,000 usarán muestreo inteligente."
    )
    
    st.divider()
    
    st.subheader("🎯 Estructura del Sistema")
    st.markdown("Define la composición del sistema para calcular la Constante Estructural K")
    
    elementos_sistema = st.number_input(
        "Total de Elementos del Sistema",
        min_value=2, max_value=1000, value=46, step=1,
        help="Cantidad total de elementos en el pool del sistema (ej: 46 bolas en lotería)"
    )
    
    manifestaciones = st.number_input(
        "Elementos con Atraso 0 (Manifestaciones)",
        min_value=1, max_value=elementos_sistema-1, value=6, step=1,
        help="Elementos que se manifiestan y resetean el sistema (ej: 6 números extraídos)"
    )
    
    # Cálculo dinámico de K
    K_dinamica = elementos_sistema - manifestaciones
    
    st.divider()
    
    st.markdown("### 📐 Constante Estructural K")
    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(r"K = E_{total} - M")
    st.markdown(f"**K = {elementos_sistema} - {manifestaciones} = {K_dinamica}**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.info(f"""
    **Interpretación:**
    - **E_total:** {elementos_sistema} elementos en el pool
    - **M (Manifestaciones):** {manifestaciones} elementos con atraso 0
    - **K:** {K_dinamica} representa la tensión estructural base del sistema
    """)
    
    st.divider()
    st.caption("v2.2 | K Dinámica | Motor Vectorizado NumPy")

# ==============================================================================
# --- 4. PROCESAMIENTO Y MÉTRICAS ---
# ==============================================================================

df_resultado, total_atrasos, num_primos, max_brecha, avg_brecha, densidad, latencias_array = analizar_homeostasis_vectorizada(N_input)

# Constante C ahora usa K dinámica
constante_C = total_atrasos + K_dinamica
entropia_sistema = float(np.std(latencias_array))

st.divider()

# Dashboard de Métricas
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🟢 Manifestaciones (n)", f"{num_primos:,}", delta=f"{densidad:.2%} densidad")
with col2:
    st.metric("🔴 Σ Atrasos del Sistema", f"{total_atrasos:,}", delta=f"Max: {max_brecha}")
with col3:
    st.metric("⚖️ Constante K (Dinámica)", f"{K_dinamica}", help=f"{elementos_sistema} - {manifestaciones}")
with col4:
    st.metric("🌊 Constante C (Saturación)", f"{constante_C:,}", help=f"ΣAtrasos + K")

# ==============================================================================
# --- 5. VISUALIZACIONES AVANZADAS ---
# ==============================================================================

tab1, tab2, tab3 = st.tabs(["📈 Mapa de Tensión", "📊 Distribución de Latencia", "📄 Registro de Datos"])

# --- TAB 1: MAPA DE TENSIÓN (CON MEJORAS VISUALES) ---
with tab1:
    st.subheader("Acumulación de Presión y Reseteo Homeostático")
    
    df_primos = df_resultado[df_resultado["Tipo"] == "MANIFESTACIÓN (n)"]
    
    # Gráfico de área para mostrar la "acumulación" de tensión
    fig_area = px.area(
        df_resultado, x="Número", y="Atraso_Individual",
        title="Curva de Presión del Sistema (Diente de Sierra)",
        labels={"Atraso_Individual": "Nivel de Tensión / Latencia"},
        color_discrete_sequence=["#ff4b4b"]
    )
    
    # Lógica inteligente para mostrar los marcadores de primos
    if len(df_primos) <= 500:
        # Mostrar todos los primos como marcadores visibles
        fig_area.add_trace(go.Scatter(
            x=df_primos["Número"],
            y=df_primos["Atraso_Individual"],
            mode='markers',
            name='Válvula de Escape (Primo)',
            marker=dict(size=8, color='#00d4ff', line=dict(width=1, color='white'))
        ))
    else:
        # Para N grande, muestrear para evitar saturación visual
        step = max(1, len(df_primos) // 200)
        primos_muestra = df_primos.iloc[::step]
        
        fig_area.add_trace(go.Scatter(
            x=primos_muestra["Número"],
            y=primos_muestra["Atraso_Individual"],
            mode='markers',
            name=f'Válvula de Escape (Muestra 1:{step})',
            marker=dict(size=6, color='#00d4ff', line=dict(width=1, color='white'))
        ))
        
        # Anotación explicativa sobre el muestreo
        fig_area.add_annotation(
            x=0.5, y=1.08,
            xref="paper", yref="paper",
            text=f"⚠️ Mostrando 1 de cada {step} primos para claridad visual (Total: {len(df_primos):,} primos)",
            showarrow=False,
            font=dict(size=11, color="#888")
        )
    
    # Añadir líneas verticales tenues en los primos (solo si hay pocos)
    if len(df_primos) <= 200:
        for _, row in df_primos.iterrows():
            fig_area.add_vline(
                x=row["Número"], 
                line_width=1, 
                line_dash="dash", 
                line_color="#00d4ff",
                opacity=0.25
            )
    
    fig_area.update_layout(
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Recta Numérica (N)",
        yaxis_title="Unidades de Atraso",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80)
    )
    st.plotly_chart(fig_area, use_container_width=True)

# --- TAB 2: DISTRIBUCIÓN DE LATENCIA ---
with tab2:
    st.subheader("Distribución Estadística de las Brechas (Prime Gaps)")
    st.markdown("""
    Esta gráfica muestra la frecuencia con la que el sistema acumula *X* unidades de atraso 
    antes de liberar tensión. Sigue una **distribución exponencial decreciente**, característica 
    de los números primos y coherente con el modelo de fiabilidad de Weibull.
    """)
    
    latencias_no_cero = latencias_array[latencias_array > 0]
    
    fig_hist = px.histogram(
        x=latencias_no_cero,
        nbins=min(50, max_brecha + 1),
        title="Frecuencia de Niveles de Latencia",
        labels={"x": "Unidades de Atraso Acumulado", "count": "Frecuencia de Ocurrencia"},
        color_discrete_sequence=["#00d4ff"]
    )
    fig_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Métricas adicionales del histograma
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.metric("Brecha Promedio", f"{avg_brecha:.2f}")
    with col_h2:
        st.metric("Brecha Máxima Observada", f"{max_brecha}")
    with col_h3:
        st.metric("Moda (Latencia más frecuente)", f"{int(np.bincount(latencias_no_cero.astype(int)).argmax())}")

# --- TAB 3: REGISTRO DE DATOS ---
with tab3:
    st.subheader("Registro de Transiciones Energéticas")
    st.info(f"Mostrando {len(df_resultado):,} filas. Para N > 5,000 se aplica muestreo representativo.")
    
    # Estilizar el dataframe con degradado de color en el riesgo
    styled_df = df_resultado.style.background_gradient(
        cmap="Reds", subset=["Riesgo_Fallo"]
    ).format({"Riesgo_Fallo": "{:.2f}"})
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500
    )

# ==============================================================================
# --- 6. FUNDAMENTOS ACADÉMICOS ---
# ==============================================================================

with st.expander("📚 Ver Fundamentos Matemáticos y Teoría del Modelo"):
    st.latex(r"C = \sum_{i=1}^{N} A_i + K")
    
    st.markdown("### 🎯 Fórmula de la Constante K (Dinámica):")
    st.latex(r"K = E_{total} - M")
    st.markdown(f"""
    Donde:
    - **E_total = {elementos_sistema}**: Total de elementos en el pool del sistema
    - **M = {manifestaciones}**: Elementos que se manifiestan con atraso 0
    - **K = {K_dinamica}**: Tensión estructural base del sistema
    """)
    
    st.markdown("""
    ### Desglose de la Ecuación de Homeostasis:
    - **$A_i$ (Atraso Individual):** Representa la distancia métrica desde el último número primo. 
      En teoría de números, esto se correlaciona con la *brecha entre primos* ($p_{n+1} - p_n$).
    - **$K$ (Constante Estructural Dinámica):** Ya no es un valor fijo, sino que se calcula según 
      la estructura del sistema que se analiza. Representa la tensión base inherente al sistema.
    - **Manifestación (n):** Cuando $A_i = 0$, el sistema alcanza un estado de simetría (número primo), 
      liberando toda la tensión acumulada (Reset).
    - **Riesgo de Fallo (Weibull):** Modelado como $R \\propto A_i^2$. La probabilidad de colapso 
      informativo no es lineal, sino cuadrática respecto al tiempo sin una "válvula de escape" prima.
    
    ### Ejemplo Práctico (Lotería):
    - **Pool de elementos:** 46 bolas
    - **Bolas extraídas (manifestaciones):** 6
    - **K = 46 - 6 = 40** (tensión estructural del sistema)
    """)
