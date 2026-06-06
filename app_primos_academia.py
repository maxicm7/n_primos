import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

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
    button[data-baseweb="tab"] {font-size: 1rem; padding: 10px 20px;}
    button[data-baseweb="tab"][aria-selected="true"] {background-color: #00d4ff !important; color: #000 !important;}
    .formula-box {background-color: #2a2a2a; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffa500; margin: 1rem 0;}
    .citation-box {background-color: #1a1a1a; padding: 0.8rem; border-radius: 0.3rem; border-left: 3px solid #888; margin: 0.5rem 0; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏛️ Protocolo de Ingeniería Probabilística: PIP-Riemann</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Modelo Heurístico de Homeostasis Estocástica en la Distribución de Números Primos</div>', unsafe_allow_html=True)

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
    - **K:** {K_dinamica} representa el umbral de saturación homeostática
    """)
    
    st.divider()
    st.caption("v3.0 | Modelo Heurístico Riguroso | Motor Vectorizado NumPy")

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

tab1, tab2, tab3, tab4 = st.tabs(["📈 Mapa de Tensión", "📊 Distribución de Latencia", "📄 Registro de Datos", "🎓 Fundamentos Académicos"])

# --- TAB 1: MAPA DE TENSIÓN ---
with tab1:
    st.subheader("Acumulación de Presión y Reseteo Homeostático")
    
    df_primos = df_resultado[df_resultado["Tipo"] == "MANIFESTACIÓN (n)"]
    
    fig_area = px.area(
        df_resultado, x="Número", y="Atraso_Individual",
        title="Curva de Presión del Sistema (Diente de Sierra)",
        labels={"Atraso_Individual": "Nivel de Tensión / Latencia"},
        color_discrete_sequence=["#ff4b4b"]
    )
    
    if len(df_primos) <= 500:
        fig_area.add_trace(go.Scatter(
            x=df_primos["Número"],
            y=df_primos["Atraso_Individual"],
            mode='markers',
            name='Válvula de Escape (Primo)',
            marker=dict(size=8, color='#00d4ff', line=dict(width=1, color='white'))
        ))
    else:
        step = max(1, len(df_primos) // 200)
        primos_muestra = df_primos.iloc[::step]
        
        fig_area.add_trace(go.Scatter(
            x=primos_muestra["Número"],
            y=primos_muestra["Atraso_Individual"],
            mode='markers',
            name=f'Válvula de Escape (Muestra 1:{step})',
            marker=dict(size=6, color='#00d4ff', line=dict(width=1, color='white'))
        ))
        
        fig_area.add_annotation(
            x=0.5, y=1.08,
            xref="paper", yref="paper",
            text=f"⚠️ Mostrando 1 de cada {step} primos para claridad visual (Total: {len(df_primos):,} primos)",
            showarrow=False,
            font=dict(size=11, color="#888")
        )
    
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
        color_discrete_sequence=["#00d4ff"],
        histnorm='probability'
    )
    
    # Superponer curva teórica exponencial
    x_range = np.linspace(1, max_brecha, 100)
    lambda_param = 1 / avg_brecha
    y_exp = lambda_param * np.exp(-lambda_param * x_range)
    
    fig_hist.add_trace(go.Scatter(
        x=x_range,
        y=y_exp,
        mode='lines',
        name=f'Curva Teórica Exponencial (λ={lambda_param:.3f})',
        line=dict(color='orange', width=3, dash='dash')
    ))
    
    fig_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Métricas adicionales del histograma
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.metric("Brecha Promedio (μ)", f"{avg_brecha:.2f}")
    with col_h2:
        st.metric("Brecha Máxima Observada", f"{max_brecha}")
    with col_h3:
        st.metric("Moda (Latencia más frecuente)", f"{int(np.bincount(latencias_no_cero.astype(int)).argmax())}")
    
    # Test de bondad de ajuste
    st.subheader("🔬 Validación Estadística")
    st.markdown("Test de Kolmogorov-Smirnov para verificar si los datos siguen una distribución exponencial:")
    
    ks_stat, ks_pvalue = stats.kstest(latencias_no_cero, 'expon', args=(0, avg_brecha))
    
    col_ks1, col_ks2, col_ks3 = st.columns(3)
    with col_ks1:
        st.metric("Estadístico KS", f"{ks_stat:.4f}")
    with col_ks2:
        st.metric("P-valor", f"{ks_pvalue:.4f}")
    with col_ks3:
        if ks_pvalue > 0.05:
            st.success("✅ No se rechaza H₀: Los datos siguen distribución exponencial")
        else:
            st.warning("⚠️ Se rechaza H₀: Los datos NO siguen distribución exponencial")

# --- TAB 3: REGISTRO DE DATOS ---
with tab3:
    st.subheader("Registro de Transiciones Energéticas")
    st.info(f"Mostrando {len(df_resultado):,} filas. Para N > 5,000 se aplica muestreo representativo.")
    
    styled_df = df_resultado.style.background_gradient(
        cmap="Reds", subset=["Riesgo_Fallo"]
    ).format({"Riesgo_Fallo": "{:.2f}"})
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500
    )

# --- TAB 4: FUNDAMENTOS ACADÉMICOS (REFORMULADO) ---
with tab4:
    st.header("🎓 Marco Teórico Riguroso")
    
    st.subheader("1. Modelo Heurístico de Homeostasis Numérica")
    st.markdown("""
    **Definición Formal:**
    
    Proponemos un **modelo heurístico** (no una demostración formal) que interpreta la distribución 
    de números primos como un sistema homeostático donde:
    
    - Los **números compuestos** representan estados de latencia acumulada
    - Los **números primos** actúan como eventos de reset que liberan la tensión del sistema
    - La **brecha entre primos** $g_i = p_{i+1} - p_i$ modela el "tiempo de fallo" entre resets
    """)
    
    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(r"C(N) = \sum_{i=1}^{N} A_i + K")
    st.markdown("""
    Donde:
    - $A_i$: Atraso individual (distancia desde el último primo)
    - $K$: Umbral de saturación homeostática (constante estructural dinámica)
    - $C(N)$: Constante de saturación del sistema hasta N
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("2. Distribución de Brechas: Conexión con la Literatura Existente")
    
    st.markdown("### 2.1. Brechas Ordinarias: Modelo Exponencial/Weibull")
    st.markdown("""
    Las brechas pequeñas y medianas entre primos siguen aproximadamente una distribución 
    **exponencial** (caso particular de Weibull con $k=1$):
    """)
    st.latex(r"f(g) = \lambda e^{-\lambda g}, \quad \text{donde } \lambda \approx \frac{1}{\ln N}")
    
    st.markdown("""
    **Referencia:** Esto es consistente con el **Modelo de Cramér (1936)**, que trata los primos 
    como eventos de un proceso de Poisson con intensidad $1/\ln(N)$.
    """)
    
    st.markdown('<div class="citation-box">', unsafe_allow_html=True)
    st.markdown("""
    **Cita:** Cramér, H. (1936). "On the order of magnitude of the difference between consecutive primes". 
    *Acta Arithmetica*, 2, 23-46.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 2.2. Brechas Extremas: Distribución de Gumbel")
    st.markdown("""
    Para las **brechas máximas** (valores extremos), el modelo predictivo es la distribución de 
    **Gumbel** (Valores Extremos Tipo I):
    """)
    st.latex(r"P(G_{max} \leq x) \approx \exp\left(-\exp\left(-\frac{x - \ln^2 N}{\ln N}\right)\right)")
    
    st.markdown("""
    **Referencia:** Este resultado fue demostrado por **Gallagher (1976)** bajo la Hipótesis de 
    Hardy-Littlewood sobre correlaciones de primos.
    """)
    
    st.markdown('<div class="citation-box">', unsafe_allow_html=True)
    st.markdown("""
    **Cita:** Gallagher, P.X. (1976). "On the distribution of primes in short intervals". 
    *Mathematika*, 23(1), 4-12.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("3. La Constante K como Umbral de Saturación")
    st.markdown("""
    La constante $K = E_{total} - M$ representa el **umbral de tolerancia** del sistema:
    
    - Si $\max(g_i) < K$: El sistema opera en **régimen homeostático estable**
    - Si $\max(g_i) \geq K$: El sistema entra en **zona de estrés estructural**
    
    **Ratio de Seguridad:**
    """)
    st.latex(r"R = \frac{\max(g_i)}{K}")
    st.markdown("""
    - $R < 1$: Sistema estable
    - $R \to 1$: Sistema cerca del colapso
    - $R > 1$: Sistema ha superado su capacidad homeostática
    """)
    
    st.divider()
    
    st.subheader("4. Limitaciones y Honestidad Epistemológica")
    
    st.warning("""
    **Este modelo es HEURÍSTICO, no una demostración formal.**
    
    Limitaciones reconocidas:
    1. **No demuestra** la Hipótesis de Riemann ni resuelve problemas abiertos
    2. **No predice mejor** que modelos existentes (Cramér, Granville) en todos los casos
    3. **No establece** una nueva teoría matemática, sino un marco interpretativo
    4. La **convergencia asintótica** mencionada es una analogía, no un teorema demostrado
    """)
    
    st.markdown("""
    **Valor del modelo:**
    - ✅ **Pedagógico:** Ayuda a visualizar patrones de distribución de primos
    - ✅ **Aplicado:** Útil para detección de anomalías en sistemas criptográficos
    - ✅ **Exploratorio:** Herramienta para generar hipótesis testables
    - ❌ **No es:** Una revolución paradigmática en teoría de números
    """)
    
    st.divider()
    
    st.subheader("5. Bibliografía Esencial")
    
    st.markdown("""
    **Fuentes primarias:**
    1. Cramér, H. (1936). "On the order of magnitude of the difference between consecutive primes"
    2. Gallagher, P.X. (1976). "On the distribution of primes in short intervals"
    3. Granville, A. (1995). "Harald Cramér and the distribution of prime numbers"
    4. Hardy, G.H. & Wright, E.M. (1979). *An Introduction to the Theory of Numbers*
    
    **Contexto histórico:**
    - **Hipótesis de Riemann (1859):** Conjetura sobre los ceros de la función zeta
    - **Teorema de los Números Primos (1896):** $\pi(N) \sim N/\ln(N)$
    - **Conjetura de Cramér (1936):** $p_{n+1} - p_n = O(\ln^2 p_n)$
    """)
    
    st.divider()
    
    st.subheader("6. Validación Empírica con tus Datos")
    
    st.markdown(f"""
    **Resultados para N = {N_input:,}:**
    
    | Métrica | Valor Observado | Valor Teórico (Cramér) | Estado |
    |---------|-----------------|------------------------|--------|
    | Densidad de primos | {densidad:.4f} | {1/np.log(N_input):.4f} | {"✅" if abs(densidad - 1/np.log(N_input)) < 0.02 else "⚠️"} |
    | Brecha promedio | {avg_brecha:.2f} | {np.log(N_input):.2f} | {"✅" if abs(avg_brecha - np.log(N_input)) < 2 else "⚠️"} |
    | Brecha máxima | {max_brecha} | ~{np.log(N_input)**2:.0f} | {"✅" if max_brecha < np.log(N_input)**2 else "⚠️"} |
    | Ratio R = max/K | {max_brecha/K_dinamica:.3f} | < 1 | {"✅" if max_brecha < K_dinamica else "⚠️"} |
    """)
