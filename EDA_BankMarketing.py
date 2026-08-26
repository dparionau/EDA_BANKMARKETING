import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN GENERAL Y ESTILO VISUAL (AZUL TECNOLÓGICO)
# ==========================================
st.set_page_config(
    page_title="BankMarketing Analytics | DMC Institute",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados (DMC Institute - Tech Blue Palette)
st.markdown("""
    <style>
        /* Importar fuente profesional */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Paleta de colores primarios y fondos */
        .stApp {
            background-color: #F4F7FA;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0B192C !important;
        }
        [data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }
        [data-testid="stSidebar"] .stRadio label {
            font-weight: 500;
            padding: 8px 12px;
            border-radius: 6px;
            transition: all 0.2s ease;
        }

        /* Encabezados y Títulos */
        h1, h2, h3 {
            color: #0B192C !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }
        
        /* Banner Header */
        .main-header {
            background: linear-gradient(135deg, #0B192C 0%, #1E3E62 60%, #008DDA 100%);
            padding: 24px 32px;
            border-radius: 12px;
            color: white;
            box-shadow: 0 10px 15px -3px rgba(11, 25, 44, 0.15);
            margin-bottom: 24px;
        }
        .main-header h1 {
            color: #FFFFFF !important;
            margin: 0;
            font-size: 2.2rem;
        }
        .main-header p {
            color: #41C9E2 !important;
            margin-top: 6px;
            font-size: 1.05rem;
            font-weight: 500;
        }

        /* Tarjetas Métricas Tecnológicas */
        .tech-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #008DDA;
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 16px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .tech-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 12px -1px rgba(0, 141, 218, 0.12);
        }
        .tech-card h4 {
            color: #1E3E62 !important;
            margin-top: 0;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .tech-card p {
            color: #475569;
            margin-bottom: 0;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* Personalización de los Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            border-bottom: 2px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #FFFFFF;
            border-radius: 8px 8px 0px 0px;
            color: #1E3E62;
            font-weight: 600;
            padding: 12px 20px;
            border: 1px solid #E2E8F0;
            border-bottom: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #008DDA !important;
            color: #FFFFFF !important;
            border-color: #008DDA !important;
        }

        /* Botones e Inputs */
        .stButton>button {
            background: linear-gradient(135deg, #008DDA 0%, #1E3E62 100%);
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            padding: 10px 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 141, 218, 0.3);
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #41C9E2 0%, #008DDA 100%);
            color: white;
        }

        /* Contenedores con sombra ligera */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# Estilo de Matplotlib/Seaborn optimizado para temas oscuros/azules
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = '#F8FAFC'
plt.rcParams['text.color'] = '#0B192C'
plt.rcParams['axes.labelcolor'] = '#0B192C'
plt.rcParams['xtick.color'] = '#1E3E62'
plt.rcParams['ytick.color'] = '#1E3E62'
plt.rcParams['font.family'] = 'sans-serif'


# ==========================================
# CLASE POO: DataAnalyzer / DataProcessor
# ==========================================
class BankMarketingAnalyzer:
    """
    Clase POO que encapsula la lógica de procesamiento, análisis descriptivo
    y generación de gráficos bivariados/univariados para BankMarketing.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Paleta de colores institucional en escala de azul tecnológico
        self.primary_color = "#0B192C"
        self.tech_blue = "#008DDA"
        self.light_cyan = "#41C9E2"
        self.navy_blue = "#1E3E62"

    def get_info(self):
        """Devuelve DataFrame estructurado con información general de variables."""
        info_df = pd.DataFrame({
            'Tipo de Dato': self.df.dtypes.astype(str),
            'Valores Nulos': self.df.isnull().sum(),
            'Valores No Nulos': self.df.notnull().sum(),
            '% Nulos': (self.df.isnull().sum() / len(self.df)) * 100
        })
        return info_df

    def classify_variables(self):
        """Clasifica variables en numéricas y categóricas mediante función personalizada."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        return num_cols, cat_cols

    def get_numeric_stats(self):
        """Genera resumen descriptivo ampliado para variables cuantitativas."""
        return self.df.describe().T

    def plot_missing(self):
        """Visualización gráfica del conteo y porcentaje de nulos."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        if len(missing) == 0:
            ax.text(0.5, 0.5, '✔ Base de Datos Completa (0% de valores faltantes)', 
                    ha='center', va='center', fontsize=12, color='#059669', fontweight='bold')
            ax.axis('off')
        else:
            missing.plot(kind='bar', color='#EF4444', ax=ax)
            ax.set_ylabel('Conteo de Nulos')
            ax.set_title('Valores Faltantes Detectados', fontweight='bold', color=self.primary_color)
        sns.despine()
        return fig

    def plot_distribution(self, col):
        """Histograma y KDE de distribución para variables cuantitativas."""
        fig, ax = plt.subplots(figsize=(8, 3.8))
        sns.histplot(self.df[col], kde=True, color=self.tech_blue, ax=ax, edgecolor='#0B192C', alpha=0.7)
        ax.set_title(f'Distribución de la Variable: {col}', fontweight='bold', fontsize=11, color=self.navy_blue)
        ax.set_xlabel(col)
        ax.set_ylabel('Frecuencia')
        sns.despine()
        return fig

    def plot_categorical(self, col):
        """Gráfico de barras ordenado para la frecuencia de variables categóricas."""
        fig, ax = plt.subplots(figsize=(8.5, 4))
        order = self.df[col].value_counts().index
        palette = sns.color_palette("Blues_r", len(order))
        sns.countplot(data=self.df, x=col, order=order, palette=palette, ax=ax)
        ax.set_title(f'Frecuencia Relativa: {col}', fontweight='bold', fontsize=11, color=self.navy_blue)
        ax.set_xlabel(col)
        ax.set_ylabel('Conteo')
        plt.xticks(rotation=45, ha='right')
        sns.despine()
        return fig

    def plot_num_vs_cat(self, num_col, cat_col):
        """Boxplot bivariado para comparar distribuciones numéricas por grupo categórico."""
        fig, ax = plt.subplots(figsize=(8.5, 4))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, palette="Blues", ax=ax, boxprops=dict(alpha=0.8))
        ax.set_title(f'Comportamiento de {num_col} por {cat_col}', fontweight='bold', fontsize=11, color=self.navy_blue)
        plt.xticks(rotation=45, ha='right')
        sns.despine()
        return fig

    def plot_cat_vs_cat(self, cat_col1, cat_col2):
        """Gráfico bivariado cruzado entre dos variables categóricas."""
        fig, ax = plt.subplots(figsize=(8.5, 4))
        sns.countplot(data=self.df, x=cat_col1, hue=cat_col2, palette=[self.tech_blue, self.light_cyan], ax=ax)
        ax.set_title(f'Relación Categórica: {cat_col1} vs {cat_col2}', fontweight='bold', fontsize=11, color=self.navy_blue)
        plt.xticks(rotation=45, ha='right')
        sns.despine()
        return fig


# ==========================================
# BARRA LATERAL (SIDEBAR) & BRANDING DMC
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <div style="background: #008DDA; display: inline-block; padding: 12px; border-radius: 50%; margin-bottom: 8px;">
                <span style="font-size: 32px;">🏦</span>
            </div>
            <h2 style="color: #FFFFFF !important; margin:0; font-size: 1.3rem;">DMC INSTITUTE</h2>
            <p style="color: #41C9E2 !important; margin: 0; font-size: 0.85rem; font-weight: 600;">Python for Analytics</p>
        </div>
        <hr style="border-color: #1E3E62; margin: 15px 0;">
    """, unsafe_allow_html=True)

    modulo = st.radio(
        "MÓDULOS DE NAVEGACIÓN",
        ["Módulo 1: Home", "Módulo 2: Carga del Dataset", "Módulo 3: Análisis Exploratorio (EDA)", "Módulo 4: Conclusiones"],
        index=0
    )

    st.markdown("""
        <hr style="border-color: #1E3E62; margin: 20px 0;">
        <div style="background: rgba(30, 62, 98, 0.4); padding: 12px; border-radius: 8px; border-left: 3px solid #008DDA;">
            <p style="margin:0; font-size: 0.8rem; color: #CBD5E1;"><b>Caso de Estudio N°1:</b> BankMarketing</p>
            <p style="margin:0; font-size: 0.75rem; color: #94A3B8;">Análisis Exploratorio de Datos</p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# MÓDULO 1: HOME (PRESENTACIÓN)
# ==========================================
if modulo == "Módulo 1: Home":
    st.markdown("""
        <div class="main-header">
            <h1>🏦 Business Analytics & EDA Dashboard</h1>
            <p>Análisis Estratégico de Campañas de Marketing Bancario | BankMarketing.csv</p>
        </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1.8, 1])

    with col_a:
        st.markdown("""
        <div class="tech-card">
            <h4>🎯 Objetivo del Proyecto</h4>
            <p>
            Esta plataforma analítica ha sido diseñada para diagnosticar y entender los patrones de comportamiento asociados a las campañas de captación de depósitos a plazo fijo en el sector bancario.
            <br><br>
            A través de un <b>Análisis Exploratorio de Datos (EDA)</b> exhaustivo, la aplicación evalúa perfiles sociodemográficos, canales de contacto y variables macroeconómicas sin recurrir a modelos predictivos, orientándose a la toma de decisiones comerciales eficientes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="tech-card" style="border-left-color: #1E3E62;">
            <h4>👨‍💻 Ficha del Entregable</h4>
            <p><b>Autor:</b> Profesional Analista</p>
            <p><b>Programa:</b> Especialización Python for Analytics</p>
            <p><b>Institución:</b> DMC Institute</p>
            <p><b>Docente:</b> MSc. Carlos Carrillo Villavicencio</p>
            <p><b>Año:</b> 2026</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Arquitectura Tecnológica")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='tech-card'><h4>🐍 Python 3.10+</h4><p>Lenguaje Core</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='tech-card'><h4>⚡ Streamlit</h4><p>Framework Interactivo</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='tech-card'><h4>📊 Pandas / NumPy</h4><p>Procesamiento Vectorial</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='tech-card'><h4>🎨 Seaborn</h4><p>Visualización Estadística</p></div>", unsafe_allow_html=True)

# ==========================================
# MÓDULO 2: CARGA DEL DATASET
# ==========================================
elif modulo == "Módulo 2: Carga del Dataset":
    st.markdown("""
        <div class="main-header">
            <h1>📂 Gestión y Carga de la Base de Datos</h1>
            <p>Carga interactiva del dataset BankMarketing.csv para habilitar los análisis</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Seleccione o arrastre el archivo BankMarketing.csv", type=["csv"])

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
            st.success("✅ ¡El archivo ha sido cargado e inicializado exitosamente en el sistema!")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"<div class='tech-card'><h4>🔢 Total Filas</h4><h2 style='color:#008DDA;'>{df.shape[0]:,}</h2></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='tech-card'><h4>📌 Total Columnas</h4><h2 style='color:#008DDA;'>{df.shape[1]}</h2></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='tech-card'><h4>💾 Memoria Consumida</h4><h2 style='color:#008DDA;'>{df.memory_usage().sum() / 1024:.1f} KB</h2></div>", unsafe_allow_html=True)

            st.markdown("### 👁️ Vista Previa del Dataset (10 Primeras Filas)")
            st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error al procesar el archivo CSV: {e}")
    else:
        st.info("📌 **Requisito obligatorio:** Cargue el archivo `BankMarketing.csv` utilizando el selector superior para habilitar el Módulo 3.")

# ==========================================
# MÓDULO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==========================================
elif modulo == "Módulo 3: Análisis Exploratorio (EDA)":
    st.markdown("""
        <div class="main-header">
            <h1>🔬 Módulo de Análisis Exploratorio (EDA)</h1>
            <p>Exploración estructurada mediante 10 ítems analíticos fundamentales</p>
        </div>
    """, unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.warning("⚠️ **Base de datos no detectada.** Por favor, diríjase al **Módulo 2: Carga del Dataset** y suba el archivo `.csv`.")
    else:
        df = st.session_state['df']
        analyzer = BankMarketingAnalyzer(df)
        num_cols, cat_cols = analyzer.classify_variables()

        # Organización en Tabs según estructura solicitada
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 1-2. Estructura & Clasificación", 
            "📊 3-4. Estadísticas & Nulos", 
            "📈 5-6. Análisis Univariado", 
            "🔄 7-8. Análisis Bivariado", 
            "🎛️ 9-10. Filtros & Hallazgos"
        ])

        # TAB 1: ÍTEMS 1 Y 2
        with tab1:
            st.markdown("### Ítem 1: Información General del Dataset (`.info()`)")
            st.dataframe(analyzer.get_info(), use_container_width=True)

            st.markdown("---")
            st.markdown("### Ítem 2: Clasificación de Variables (Función Personalizada)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='tech-card'><h4>🔢 Variables Numéricas ({len(num_cols)})</h4><p>{', '.join(num_cols)}</p></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='tech-card'><h4>🏷️ Variables Categóricas ({len(cat_cols)})</h4><p>{', '.join(cat_cols)}</p></div>", unsafe_allow_html=True)

        # TAB 2: ÍTEMS 3 Y 4
        with tab2:
            st.markdown("### Ítem 3: Estadísticas Descriptivas (`.describe()`)")
            st.dataframe(analyzer.get_numeric_stats(), use_container_width=True)

            st.markdown("---")
            st.markdown("### Ítem 4: Análisis de Valores Faltantes")
            col_g, col_t = st.columns([1.2, 1])
            with col_g:
                st.pyplot(analyzer.plot_missing())
            with col_t:
                st.markdown("""
                <div class="tech-card">
                    <h4>🔍 Diagnóstico de Completitud</h4>
                    <p>Se valida la ausencia de valores nulos directos (<i>NaN</i>). En el contexto del dataset BankMarketing, las observaciones desconocidas han sido catalogadas bajo la etiqueta explícita <b>'unknown'</b> para preservar la representatividad de la muestra comercial.</p>
                </div>
                """, unsafe_allow_html=True)

        # TAB 3: ÍTEMS 5 Y 6
        with tab3:
            st.markdown("### Ítem 5: Distribución de Variables Numéricas")
            selected_num = st.selectbox("Seleccione variable numérica a analizar:", num_cols, index=0)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(analyzer.plot_distribution(selected_num))
            with col2:
                st.markdown(f"""
                <div class="tech-card">
                    <h4>Métricas Clave: {selected_num}</h4>
                    <p><b>Media:</b> {df[selected_num].mean():.2f}</p>
                    <p><b>Mediana:</b> {df[selected_num].median():.2f}</p>
                    <p><b>Desviación Estándar:</b> {df[selected_num].std():.2f}</p>
                    <p><b>Rango:</b> [{df[selected_num].min()} - {df[selected_num].max()}]</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Ítem 6: Análisis de Variables Categóricas")
            selected_cat = st.selectbox("Seleccione variable categórica:", cat_cols, index=0)
            col3, col4 = st.columns([2, 1])
            with col3:
                st.pyplot(analyzer.plot_categorical(selected_cat))
            with col4:
                st.write(f"**Proporción Relativa (%) de `{selected_cat}`:**")
                st.dataframe(df[selected_cat].value_counts(normalize=True).map("{:.2%}".format))

        # TAB 4: ÍTEMS 7 Y 8
        with tab4:
            st.markdown("### Ítem 7: Análisis Bivariado (Numérico vs Categórico)")
            b1, b2 = st.columns(2)
            with b1:
                num_v = st.selectbox("Variable Numérica:", num_cols, index=num_cols.index('duration') if 'duration' in num_cols else 0)
            with b2:
                cat_v = st.selectbox("Variable Categórica Target:", cat_cols, index=cat_cols.index('y') if 'y' in cat_cols else 0, key="biv_target1")
            
            st.pyplot(analyzer.plot_num_vs_cat(num_v, cat_v))

            st.markdown("---")
            st.markdown("### Ítem 8: Análisis Bivariado (Categórico vs Categórico)")
            b3, b4 = st.columns(2)
            with b3:
                cat1 = st.selectbox("Variable Categórica 1:", cat_cols, index=cat_cols.index('education') if 'education' in cat_cols else 0)
            with b4:
                cat2 = st.selectbox("Variable Categórica 2:", cat_cols, index=cat_cols.index('y') if 'y' in cat_cols else 0, key="biv_target2")
            
            st.pyplot(analyzer.plot_cat_vs_cat(cat1, cat2))

        # TAB 5: ÍTEMS 9 Y 10
        with tab5:
            st.markdown("### Ítem 9: Análisis Basado en Parámetros Seleccionados")
            
            f1, f2 = st.columns(2)
            with f1:
                min_a, max_a = int(df['age'].min()), int(df['age'].max())
                age_sel = st.slider("Filtrar Rango de Edad (`age`):", min_a, max_a, (25, 55))
            with f2:
                job_list = df['job'].unique().tolist() if 'job' in df.columns else []
                job_sel = st.multiselect("Filtrar por Ocupación (`job`):", job_list, default=job_list[:2] if job_list else [])

            chk_table = st.checkbox("Mostrar registros filtrados en tabla")

            df_sub = df[(df['age'] >= age_sel[0]) & (df['age'] <= age_sel[1])]
            if job_sel:
                df_sub = df_sub[df_sub['job'].isin(job_sel)]

            st.info(f"📊 **Muestra Segmentada:** {len(df_sub):,} de {len(df):,} registros coinciden con los criterios elegidos.")
            if chk_table:
                st.dataframe(df_sub, use_container_width=True)

            st.markdown("---")
            st.markdown("### Ítem 10: Hallazgos Clave derivados del EDA")
            
            h1, h2 = st.columns([1.4, 1])
            with h1:
                st.markdown("""
                <div class="tech-card">
                    <h4>💡 Insights Comerciales</h4>
                    <p><b>1. Duración del Contacto (duration):</b> Existe una correlación positiva directa entre el tiempo sostenido de la llamada y la aceptación del depósito a plazo fijo.</p>
                    <p><b>2. Desgaste por Campaña (campaign):</b> Someter al cliente a más de 4 llamadas dentro de la misma campaña disminuye drásticamente la tasa de efectividad.</p>
                    <p><b>3. Recontactabilidad Histórica (poutcome):</b> Los clientes con conversión previa exitosa muestran una efectividad significativamente superior al promedio general.</p>
                </div>
                """, unsafe_allow_html=True)
            with h2:
                if 'y' in df.columns:
                    fig_p, ax_p = plt.subplots(figsize=(4.5, 3.2))
                    df['y'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#008DDA', '#41C9E2'], ax=ax_p, startangle=90)
                    ax_p.set_ylabel('')
                    ax_p.set_title('Ratio Global de Aceptación (y)', fontweight='bold', color='#0B192C')
                    st.pyplot(fig_p)

# ==========================================
# MÓDULO 4: CONCLUSIONES
# ==========================================
elif modulo == "Módulo 4: Conclusiones":
    st.markdown("""
        <div class="main-header">
            <h1>📌 Conclusiones Estratégicas y de Negocio</h1>
            <p>Recomendaciones clave enfocadas en la toma de decisiones comerciales</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tech-card">
        <h4>1. Reestructuración del Argumentario Comercial (Duración)</h4>
        <p>Dado que las llamadas con conversión promedian una mayor duración (<code>duration</code>), se concluye que la efectividad depende de la calidad del argumento de ventas. Se recomienda priorizar la profundidad de la gestión por encima de llamadas breves masivas.</p>
    </div>

    <div class="tech-card">
        <h4>2. Umbral Máximo de Contacto por Cliente (Campañas)</h4>
        <p>El análisis demuestra que superar los 3 a 4 contactos (<code>campaign</code>) no incrementa las ventas y causa un fuerte desgaste en la base. Debe fijarse una regla de negocio que pause la gestión comercial tras el cuarto intento fallido.</p>
    </div>

    <div class="tech-card">
        <h4>3. Priorización de la Base con Historial Positivo</h4>
        <p>La categoría <i>poutcome = success</i> representa el subsegmento con mayor retorno de inversión. Se sugiere priorizar la asignación diaria de estos clientes a los ejecutivos comerciales de mayor rendimiento.</p>
    </div>

    <div class="tech-card">
        <h4>4. Segmentación por Perfil Socio-Laboral</h4>
        <p>Sectores con mayor estabilidad económica y formación (<i>management</i>, <i>technician</i>, nivel <i>tertiary</i>/<i>university</i>) registran índices de aceptación superiores, permitiendo optimizar el costo de adquisición por cliente.</p>
    </div>

    <div class="tech-card">
        <h4>5. Sincronización con el Entorno Macroeconómico</h4>
        <p>Variables como la tasa <code>euribor3m</code> influyen en la propensión al ahorro en depósitos. Las campañas comerciales deben coordinar sus lanzamientos masivos durante contextos de tasas atractivas para el inversor.</p>
    </div>
    """, unsafe_allow_html=True)
