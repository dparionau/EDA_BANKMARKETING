import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN GENERAL Y ESTILO VISUAL
# ==========================================
st.set_page_config(
    page_title="BankMarketing Analytics | Indecopi & Dilic",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados (Color tecnológico Azulino)
st.markdown("""
    <style>
        /* Color primario y fondos */
        :root {
            --primary-blue: #1E3A8A;
            --accent-blue: #2563EB;
            --light-blue: #38BDF8;
            --dark-navy: #0F172A;
            --bg-card: #F8FAFC;
        }

        /* Estilo general del cuerpo */
        .main {
            background-color: #F8FAFC;
        }

        /* Encabezados */
        h1, h2, h3 {
            color: #1E3A8A !important;
            font-family: 'Segoe UI', Roboto, sans-serif;
            font-weight: 700;
        }

        /* Tarjetas de métricas y contenedores */
        .metric-card {
            background-color: #FFFFFF;
            border-left: 5px solid #2563EB;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            margin-bottom: 12px;
        }

        /* Personalización de los Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #E2E8F0;
            border-radius: 6px 6px 0px 0px;
            color: #334155;
            font-weight: 600;
            padding: 10px 16px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #1E3A8A !important;
            color: #FFFFFF !important;
        }

        /* Botones e interacciones */
        .stButton>button {
            background-color: #2563EB;
            color: white;
            border-radius: 6px;
            border: none;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #1D4ED8;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Configuración del estilo de gráficos en tonos azules
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['text.color'] = '#1E293B'
plt.rcParams['axes.labelcolor'] = '#1E293B'
plt.rcParams['xtick.color'] = '#1E293B'
plt.rcParams['ytick.color'] = '#1E293B'

# ==========================================
# CLASE POO: DataAnalyzer / DataProcessor
# ==========================================
class BankMarketingAnalyzer:
    """
    Clase encargada de encapsular la lógica del procesamiento,
    estadísticas descriptivas, clasificación y gráficos para BankMarketing.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Paleta de color corporativo azulino para gráficos
        self.blue_palette = sns.color_palette("Blues_r")
        self.primary_color = "#1E3A8A"
        self.accent_color = "#2563EB"

    def get_info(self):
        """Devuelve un DataFrame con tipos de datos y nulos por columna."""
        info_df = pd.DataFrame({
            'Tipo de Dato': self.df.dtypes.astype(str),
            'Valores Nulos': self.df.isnull().sum(),
            'Valores No Nulos': self.df.notnull().sum(),
            '% Nulos': (self.df.isnull().sum() / len(self.df)) * 100
        })
        return info_df

    def classify_variables(self):
        """Clasifica las variables en numéricas y categóricas."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        return num_cols, cat_cols

    def get_numeric_stats(self):
        """Genera estadísticas descriptivas para variables numéricas."""
        return self.df.describe().T

    def plot_missing(self):
        """Genera un gráfico de barras con el porcentaje de nulos por variable."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        if len(missing) == 0:
            ax.text(0.5, 0.5, '✔ No existen valores faltantes en el Dataset', 
                    ha='center', va='center', fontsize=12, color='#16A34A', fontweight='bold')
            ax.axis('off')
        else:
            missing.plot(kind='bar', color='#EF4444', ax=ax)
            ax.set_ylabel('Cantidad de Nulos')
            ax.set_title('Valores Faltantes por Variable', fontweight='bold', color=self.primary_color)
        sns.despine()
        return fig

    def plot_distribution(self, col):
        """Genera histograma y KDE para una variable numérica."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(self.df[col], kde=True, color=self.accent_color, ax=ax, edgecolor='#1E3A8A')
        ax.set_title(f'Distribución de {col}', fontweight='bold', fontsize=12, color=self.primary_color)
        ax.set_xlabel(col)
        ax.set_ylabel('Frecuencia')
        sns.despine()
        return fig

    def plot_categorical(self, col):
        """Genera un gráfico de barras para la frecuencia de variables categóricas."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        order = self.df[col].value_counts().index
        sns.countplot(data=self.df, x=col, order=order, palette="Blues_r", ax=ax)
        ax.set_title(f'Frecuencia de {col}', fontweight='bold', fontsize=12, color=self.primary_color)
        ax.set_xlabel(col)
        ax.set_ylabel('Conteo')
        plt.xticks(rotation=45, ha='right')
        sns.despine()
        return fig

    def plot_num_vs_cat(self, num_col, cat_col):
        """Genera un boxplot para analizar variable numérica vs categórica."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, palette="Blues", ax=ax)
        ax.set_title(f'Comportamiento de {num_col} según {cat_col}', fontweight='bold', fontsize=12, color=self.primary_color)
        plt.xticks(rotation=45, ha='right')
        sns.despine()
        return fig

    def plot_cat_vs_cat(self, cat_col1, cat_col2):
        """Genera un gráfico de barras agrupadas entre dos variables categóricas."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.countplot(data=self.df, x=cat_col1, hue=cat_col2, palette=["#2563EB", "#93C5FD"], ax=ax)
        ax.set_title(f'Relación Bivariada: {cat_col1} vs {cat_col2}', fontweight='bold', fontsize=12, color=self.primary_color)
        plt.xticks(rotation=45, ha='right')
        sns.despine()
        return fig


# ==========================================
# BARRA LATERAL (SIDEBAR) & NAVEGACIÓN
# ==========================================
st.sidebar.image("https://img.icons8.com/isometric-folders/100/bank.png", width=70)
st.sidebar.title("Bank Analytics")
st.sidebar.markdown("---")

modulo = st.sidebar.radio(
    "Navegación Principal:",
    ["Módulo 1: Home", "Módulo 2: Carga del Dataset", "Módulo 3: Análisis Exploratorio (EDA)", "Módulo 4: Conclusiones"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **DILIC Institute**\nEspecialización en Python for Analytics")

# ==========================================
# MÓDULO 1: HOME
# ==========================================
if modulo == "Módulo 1: Home":
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏦 Business Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #475569;'>Análisis Exploratorio de Campañas de Marketing Bancario</h4>", unsafe_allow_html=True)
    st.markdown("---")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("""
        ### 🎯 Propósito del Proyecto
        Esta plataforma analítica ha sido construida para explorar y diagnosticar el rendimiento de las campañas telefónicas de **depósitos a plazo fijo**. 
        
        A través de técnicas de **EDA (Exploratory Data Analysis)** y visualizaciones cuantitativas, la aplicación facilita la identificación de patrones de comportamiento, niveles de contacto óptimos y la influencia de factores macroeconómicos sobre el éxito comercial.
        
        #### 📌 Objetivos Clave:
        * Identificar variables con mayor tasa de conversión (`y`).
        * Reducir el desgaste en la base de clientes acortando la frecuencia ineficiente de llamadas.
        * Apoyar la toma de decisiones estratégicas de la fuerza comercial.
        """)

    with col_b:
        st.markdown("""
        <div class="metric-card">
            <h4>👨‍💻 Ficha del Proyecto</h4>
            <hr style="margin: 8px 0;">
            <p><b>Autor:</b> Profesional Analista</p>
            <p><b>Programa:</b> Python for Analytics</p>
            <p><b>Institución:</b> DILIC Institute</p>
            <p><b>Docente:</b> MSc. Carlos Carrillo V.</p>
            <p><b>Año:</b> 2026</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Tecnologías e Infraestructura Analítica")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lenguaje Core", "Python 3.10+")
    col2.metric("Framework Web", "Streamlit")
    col3.metric("Procesamiento", "Pandas / NumPy")
    col4.metric("Visualización", "Seaborn / Matplotlib")

# ==========================================
# MÓDULO 2: CARGA DEL DATASET
# ==========================================
elif modulo == "Módulo 2: Carga del Dataset":
    st.title("📂 Gestión & Carga de Datos")
    st.markdown("Suba el archivo original **`BankMarketing.csv`** para activar las capacidades analíticas de la suite.")

    uploaded_file = st.file_uploader("Seleccione o arrastre el archivo CSV aquí", type=["csv"])

    if uploaded_file is not None:
        try:
            try:
                df = pd.read_csv(uploaded_file, sep=';')
                if len(df.columns) <= 1:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=',')
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',')

            st.session_state['df'] = df
            st.success("✅ ¡El archivo se procesó e inicializó con éxito en la memoria local!")

            # Tarjetas con resumen de dimensiones
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-card'><h3>{df.shape[0]:,}</h3><p>Total Registros (Filas)</p></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><h3>{df.shape[1]}</h3><p>Variables (Columnas)</p></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-card'><h3>{df.memory_usage().sum() / 1024:.1f} KB</h3><p>Memoria Consumida</p></div>", unsafe_allow_html=True)

            st.markdown("### 👁️ Vista Previa del Dataset")
            st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
    else:
        st.info("📌 **Instrucción:** Cargue la base de datos `BankMarketing.csv` usando la caja superior.")

# ==========================================
# MÓDULO 3: ANÁLISIS EXPLORATORIO (EDA)
# ==========================================
elif modulo == "Módulo 3: Análisis Exploratorio (EDA)":
    st.title("🔬 Módulo de Análisis Exploratorio (EDA)")

    if 'df' not in st.session_state:
        st.warning("⚠️ **Base de datos no detectada.** Dirígete al **Módulo 2** para realizar la carga del archivo.")
    else:
        df = st.session_state['df']
        analyzer = BankMarketingAnalyzer(df)
        num_cols, cat_cols = analyzer.classify_variables()

        # Tabs estilizados para los 10 Ítems obligatorios
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 1-2. Estructura & Tipos", 
            "📊 3-4. Estadísticas & Nulos", 
            "📈 5-6. Univariado", 
            "🔄 7-8. Bivariado", 
            "🎛️ 9-10. Filtros & Insights"
        ])

        # TAB 1: ESTRUCTURA Y CLASIFICACIÓN
        with tab1:
            st.subheader("Ítem 1: Inspección de Estructura e Info General")
            st.dataframe(analyzer.get_info(), use_container_width=True)

            st.markdown("---")
            st.subheader("Ítem 2: Clasificación de Variables (Función Personalizada)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### 🔢 Variables Numéricas ({len(num_cols)})")
                st.write(num_cols)
            with col2:
                st.markdown(f"#### 🏷️ Variables Categóricas ({len(cat_cols)})")
                st.write(cat_cols)

        # TAB 2: ESTADÍSTICAS Y FALTANTES
        with tab2:
            st.subheader("Ítem 3: Resumen Estadístico Descriptivo")
            st.dataframe(analyzer.get_numeric_stats(), use_container_width=True)

            st.markdown("---")
            st.subheader("Ítem 4: Evaluación de Valores Faltantes (Nulos)")
            col_g, col_t = st.columns([1.2, 1])
            with col_g:
                st.pyplot(analyzer.plot_missing())
            with col_t:
                st.markdown("""
                **Diagnóstico de Faltantes:**
                * La detección de valores `NaN` o desbalance de datos permite garantizar la calidad de la base antes de cruzar variables.
                * En marketing bancario, los valores desconocidos suelen catalogarse como `'unknown'` para preservar la integridad del historial.
                """)

        # TAB 3: ANÁLISIS UNIVARIADO
        with tab3:
            st.subheader("Ítem 5: Distribución de Variables Numéricas")
            selected_num = st.selectbox("Seleccione la variable numérica a analizar:", num_cols, index=0)
            c1, c2 = st.columns([2, 1])
            with c1:
                st.pyplot(analyzer.plot_distribution(selected_num))
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Métricas de {selected_num}</h4>
                    <p><b>Media:</b> {df[selected_num].mean():.2f}</p>
                    <p><b>Mediana:</b> {df[selected_num].median():.2f}</p>
                    <p><b>Desv. Estándar:</b> {df[selected_num].std():.2f}</p>
                    <p><b>Mínimo / Máximo:</b> {df[selected_num].min()} / {df[selected_num].max()}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Ítem 6: Frecuencias de Variables Categóricas")
            selected_cat = st.selectbox("Seleccione la variable categórica:", cat_cols, index=0)
            c3, c4 = st.columns([2, 1])
            with c3:
                st.pyplot(analyzer.plot_categorical(selected_cat))
            with c4:
                st.write(f"**Distribución % de `{selected_cat}`:**")
                st.dataframe(df[selected_cat].value_counts(normalize=True).map("{:.2%}".format))

        # TAB 4: ANÁLISIS BIVARIADO
        with tab4:
            st.subheader("Ítem 7: Análisis Bivariado (Numérico vs Categórico)")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                num_v = st.selectbox("Variable Numérica:", num_cols, index=num_cols.index('duration') if 'duration' in num_cols else 0)
            with col_b2:
                cat_v = st.selectbox("Categoría Objetivo (Target):", cat_cols, index=cat_cols.index('y') if 'y' in cat_cols else 0, key="cat_target")
            
            st.pyplot(analyzer.plot_num_vs_cat(num_v, cat_v))

            st.markdown("---")
            st.subheader("Ítem 8: Análisis Bivariado (Categórico vs Categórico)")
            col_b3, col_b4 = st.columns(2)
            with col_b3:
                cat1 = st.selectbox("Variable Categórica 1:", cat_cols, index=cat_cols.index('education') if 'education' in cat_cols else 0)
            with col_b4:
                cat2 = st.selectbox("Variable Categórica 2:", cat_cols, index=cat_cols.index('y') if 'y' in cat_cols else 0, key="cat_target2")
            
            st.pyplot(analyzer.plot_cat_vs_cat(cat1, cat2))

        # TAB 5: FILTROS DINÁMICOS Y INSIGHTS
        with tab5:
            st.subheader("Ítem 9: Explorador Dinámico Segmentado")
            
            f1, f2 = st.columns(2)
            with f1:
                min_age, max_age = int(df['age'].min()), int(df['age'].max())
                age_sel = st.slider("Rango de Edad (`age`):", min_age, max_age, (25, 60))
            with f2:
                jobs = df['job'].unique().tolist() if 'job' in df.columns else []
                jobs_sel = st.multiselect("Filtrar por Ocupación (`job`):", jobs, default=jobs[:2] if jobs else [])

            chk_show = st.checkbox("Mostrar tabla con los datos filtrados")

            df_sub = df[(df['age'] >= age_sel[0]) & (df['age'] <= age_sel[1])]
            if jobs_sel:
                df_sub = df_sub[df_sub['job'].isin(jobs_sel)]

            st.info(f"🔍 **Subconjunto Filtrado:** {len(df_sub):,} de {len(df):,} registros seleccionados.")
            if chk_show:
                st.dataframe(df_sub, use_container_width=True)

            st.markdown("---")
            st.subheader("Ítem 10: Síntesis y Hallazgos Principales")
            
            h1, h2 = st.columns([1.5, 1])
            with h1:
                st.markdown("""
                * **Efecto Duración (`duration`):** Existe una relación directa entre el tiempo sostenido de la llamada y la aceptación del depósito a plazo.
                * **Saturación Comercial (`campaign`):** Pasados los 4 contactos, la tasa de rechazo aumenta sensiblemente.
                * **Eficacia Histórica (`poutcome`):** La efectividad sobre clientes con conversión previa positiva (`success`) es sustancialmente superior al promedio.
                """)
            with h2:
                if 'y' in df.columns:
                    fig_pie, ax_pie = plt.subplots(figsize=(5, 3.5))
                    df['y'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#2563EB', '#93C5FD'], ax=ax_pie, startangle=90)
                    ax_pie.set_ylabel('')
                    ax_pie.set_title('Ratio de Aceptación Global (y)', fontweight='bold', color='#1E3A8A')
                    st.pyplot(fig_pie)

# ==========================================
# MÓDULO 4: CONCLUSIONES
# ==========================================
elif modulo == "Módulo 4: Conclusiones":
    st.title("📌 Conclusiones Estratégicas")
    st.markdown("Resumen de decisiones analíticas para mejorar la tasa de conversión comercial:")

    st.markdown("""
    <div class="metric-card">
        <h4>1. Gestión Eficiente del Tiempo de Llamada</h4>
        <p>Aumentar el tiempo de conversación en clientes calificados genera mayor conversión que aumentar la cantidad de llamadas breves e ineficientes.</p>
    </div>
    <div class="metric-card">
        <h4>2. Control del Límite de Incursiones</h4>
        <p>Se recomienda establecer un tope máximo de 3 a 4 intentos por cliente durante la misma campaña para mitigar la insatisfacción comercial.</p>
    </div>
    <div class="metric-card">
        <h4>3. Priorización por Perfil Socioeconómico</h4>
        <p>Segmentos con perfiles profesionales definidos (Management, Technicians) registran mayor afinidad a productos de inversión a plazo fijo.</p>
    </div>
    <div class="metric-card">
        <h4>4. Explotación de la Base con Historial Positivo</h4>
        <p>Los clientes con antecedente <i>poutcome = success</i> deben priorizarse en la asignación diaria de llamadas por parte de la fuerza ejecutiva.</p>
    </div>
    <div class="metric-card">
        <h4>5. Alineación con Indicadores Macroeconómicos</h4>
        <p>La variación del índice Euribor a 3 meses incide en la atracción del cliente. Se deben coordinar esfuerzos comerciales en períodos de tasas favorables.</p>
    </div>
    """, unsafe_allow_html=True)
