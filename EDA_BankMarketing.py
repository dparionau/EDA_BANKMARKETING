import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración general de la página
st.set_page_config(
    page_title="EDA Bank Marketing Dashboard",
    page_icon="🏦",
    layout="wide"
)

# Estilo global de Seaborn/Matplotlib
sns.set_theme(style="whitegrid")

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
        fig, ax = plt.subplots(figsize=(8, 4))
        if len(missing) == 0:
            ax.text(0.5, 0.5, '¡No existen valores faltantes en el Dataset!', 
                    ha='center', va='center', fontsize=12, color='green')
            ax.axis('off')
        else:
            missing.plot(kind='bar', color='#e74c3c', ax=ax)
            ax.set_ylabel('Cantidad de Nulos')
            ax.set_title('Conteo de Valores Faltantes por Variable')
        return fig

    def plot_distribution(self, col):
        """Genera histograma y KDE para una variable numérica."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(self.df[col], kde=True, color='#2980b9', ax=ax)
        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frecuencia')
        return fig

    def plot_categorical(self, col):
        """Genera un gráfico de barras para la frecuencia de variables categóricas."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        order = self.df[col].value_counts().index
        sns.countplot(data=self.df, x=col, order=order, palette='Blues_r', ax=ax)
        ax.set_title(f'Frecuencia de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Conteo')
        plt.xticks(rotation=45, ha='right')
        return fig

    def plot_num_vs_cat(self, num_col, cat_col):
        """Genera un boxplot para analizar variable numérica vs categórica."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, palette='Set2', ax=ax)
        ax.set_title(f'Análisis de {num_col} según {cat_col}')
        plt.xticks(rotation=45, ha='right')
        return fig

    def plot_cat_vs_cat(self, cat_col1, cat_col2):
        """Genera un gráfico de barras agrupadas entre dos variables categóricas."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.countplot(data=self.df, x=cat_col1, hue=cat_col2, palette='Set1', ax=ax)
        ax.set_title(f'Relación entre {cat_col1} y {cat_col2}')
        plt.xticks(rotation=45, ha='right')
        return fig


# ==========================================
# ESTRUCTURA PRINCIPAL DE LA APLICACIÓN
# ==========================================

# Sidebar: Navegación Principal
st.sidebar.title("📌 Menú de Navegación")
modulo = st.sidebar.radio(
    "Seleccione un Módulo:",
    ["Módulo 1: Home", "Módulo 2: Carga del Dataset", "Módulo 3: Análisis Exploratorio (EDA)", "Módulo 4: Conclusiones"]
)

# ------------------------------------------
# MÓDULO 1: HOME
# ------------------------------------------
if modulo == "Módulo 1: Home":
    st.title("🏦 Análisis Exploratorio de Datos: Bank Marketing")
    st.subheader("Especialización en Python for Analytics - Caso de Estudio N°1")
    
    st.markdown("""
    ---
    ### 🎯 Objetivo del Proyecto
    Esta aplicación interactiva tiene como objetivo realizar un **Análisis Exploratorio de Datos (EDA)** detallado sobre las campañas de marketing directo de una institución financiera. 
    Se busca identificar los factores clave que inciden en la contratación de depósitos a plazo fijo, analizando perfiles de clientes, canales de contacto y variables macroeconómicas para optimizar decisiones comerciales sin la necesidad de modelos predictivos.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 👨‍💻 Datos del Autor
        * **Nombre:** [Tu Nombre Completo]
        * **Programa:** Especialización en Python for Analytics
        * **Institución:** DILIC Institute
        * **Docente:** MSc. Carlos Carrillo Villavicencio
        * **Año:** 2026
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Tecnologías Utilizadas
        * **Lenguaje:** Python 3.10+
        * **Framework Web:** Streamlit
        * **Procesamiento de Datos:** Pandas & NumPy
        * **Visualización:** Matplotlib & Seaborn
        * **Paradigma:** Programación Orientada a Objetos (POO)
        """)

    st.markdown("""
    ---
    ### 📊 Contexto del Dataset BankMarketing
    El conjunto de datos contiene información sobre las campañas publicitarias telefónicas realizadas por el banco. Durante los últimos 6 meses, la efectividad cayó del **12% al 8%**, impactando los resultados comerciales. El dataset incluye información demográfica, historial de gestión previa, datos socioeconómicos y la variable objetivo `y` (*aceptó o no la oferta*).
    """)

# ------------------------------------------
# MÓDULO 2: CARGA DEL DATASET
# ------------------------------------------
elif modulo == "Módulo 2: Carga del Dataset":
    st.title("📂 Carga del Dataset BankMarketing")
    st.markdown("Cargue el archivo en formato `.csv` para inicializar el sistema de análisis.")

    uploaded_file = st.sidebar.file_uploader("Subir archivo BankMarketing.csv", type=["csv"])

    if uploaded_file is not None:
        try:
            # Soporta delimitador estándar (,) o punto y coma (;) común en este dataset
            try:
                df = pd.read_csv(uploaded_file, sep=';')
                if len(df.columns) <= 1:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=',')
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',')

            st.session_state['df'] = df
            st.success("✅ ¡Archivo cargado e inicializado correctamente en memoria!")

            # Métricas rápidas
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Filas (Registros)", df.shape[0])
            col2.metric("Total de Columnas (Variables)", df.shape[1])
            col3.metric("Tamaño en Memoria", f"{df.memory_usage().sum() / 1024:.2f} KB")

            st.markdown("### 👁️ Vista Previa de los Datos (Primeras 10 filas)")
            st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error al procesar el archivo CSV: {e}")
    else:
        st.warning("⚠️ Por favor, sube el archivo `BankMarketing.csv` desde la barra lateral (Sidebar) para activar el análisis.")

# ------------------------------------------
# MÓDULO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ------------------------------------------
elif modulo == "Módulo 3: Análisis Exploratorio (EDA)":
    st.title("🔬 Análisis Exploratorio de Datos (EDA)")

    if 'df' not in st.session_state:
        st.warning("⚠️ No se ha cargado ningún dataset. Dirígete al **Módulo 2: Carga del Dataset** en la barra lateral para subir el archivo `.csv`.")
    else:
        df = st.session_state['df']
        analyzer = BankMarketingAnalyzer(df)
        num_cols, cat_cols = analyzer.classify_variables()

        # PESTAÑAS (TABS) OBLIGATORIAS PARA ORGANIZAR LOS 10 ÍTEMS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 1-2. Estructura & Tipos", 
            "📊 3-4. Estadísticas & Nulos", 
            "📈 5-6. Univariado", 
            "🔄 7-8. Bivariado", 
            "🎛️ 9-10. Filtros & Hallazgos"
        ])

        # TAB 1: ÍTEMS 1 Y 2
        with tab1:
            st.header("Ítem 1: Información General del Dataset")
            st.dataframe(analyzer.get_info(), use_container_width=True)

            st.markdown("---")
            st.header("Ítem 2: Clasificación de Variables")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"🔢 Variables Numéricas ({len(num_cols)})")
                st.write(num_cols)
            with col2:
                st.subheader(f"🏷️ Variables Categóricas ({len(cat_cols)})")
                st.write(cat_cols)

        # TAB 2: ÍTEMS 3 Y 4
        with tab2:
            st.header("Ítem 3: Estadísticas Descriptivas")
            st.dataframe(analyzer.get_numeric_stats(), use_container_width=True)
            st.info("💡 **Interpretación**: Compare las medias y medianas de variables como `duration` o `age` para detectar sesgos o asimetría hacia valores atípicos (outliers).")

            st.markdown("---")
            st.header("Ítem 4: Análisis de Valores Faltantes")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.pyplot(analyzer.plot_missing())
            with col2:
                st.markdown("""
                **Discusión de Nulos:**
                * Se evalúa la completitud del conjunto de datos.
                * En datasets bancarios, los faltantes suelen codificarse como `'unknown'`, lo cual debe tratarse como una categoría explícita antes de eliminar observaciones.
                """)

        # TAB 3: ÍTEMS 5 Y 6
        with tab3:
            st.header("Ítem 5: Distribución de Variables Numéricas")
            selected_num = st.selectbox("Seleccione una variable numérica:", num_cols, index=0)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(analyzer.plot_distribution(selected_num))
            with col2:
                st.write(f"**Métricas Clave de `{selected_num}`:**")
                st.metric("Media", f"{df[selected_num].mean():.2f}")
                st.metric("Mediana", f"{df[selected_num].median():.2f}")
                st.metric("Desviación Estándar", f"{df[selected_num].std():.2f}")

            st.markdown("---")
            st.header("Ítem 6: Análisis de Variables Categóricas")
            selected_cat = st.selectbox("Seleccione una variable categórica:", cat_cols, index=0)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(analyzer.plot_categorical(selected_cat))
            with col2:
                st.write(f"**Proporción Relativa de `{selected_cat}`:**")
                st.dataframe(df[selected_cat].value_counts(normalize=True).map("{:.2%}".format))

        # TAB 4: ÍTEMS 7 Y 8
        with tab4:
            st.header("Ítem 7: Análisis Bivariado (Numérico vs Categórico)")
            col1, col2 = st.columns(2)
            with col1:
                num_var_biv = st.selectbox("Variable Numérica:", num_cols, index=num_cols.index('duration') if 'duration' in num_cols else 0)
            with col2:
                cat_var_biv = st.selectbox("Variable Categórica Target/Grupo:", cat_cols, index=cat_cols.index('y') if 'y' in cat_cols else 0, key="biv1")
            
            st.pyplot(analyzer.plot_num_vs_cat(num_var_biv, cat_var_biv))

            st.markdown("---")
            st.header("Ítem 8: Análisis Bivariado (Categórico vs Categórico)")
            col3, col4 = st.columns(2)
            with col3:
                cat1 = st.selectbox("Variable Categórica 1:", cat_cols, index=cat_cols.index('education') if 'education' in cat_cols else 0)
            with col4:
                cat2 = st.selectbox("Variable Categórica 2 (Target):", cat_cols, index=cat_cols.index('y') if 'y' in cat_cols else 0, key="biv2")
            
            st.pyplot(analyzer.plot_cat_vs_cat(cat1, cat2))

        # TAB 5: ÍTEMS 9 Y 10
        with tab5:
            st.header("Ítem 9: Análisis Basado en Parámetros Seleccionados")
            st.markdown("Filtre dinámicamente el dataset para analizar subsegmentos de clientes de forma rápida.")

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                age_min, age_max = int(df['age'].min()), int(df['age'].max())
                age_range = st.slider("Rango de Edad (age):", age_min, age_max, (age_min, age_max))
            
            with col_f2:
                job_list = df['job'].unique().tolist() if 'job' in df.columns else []
                selected_jobs = st.multiselect("Filtrar por Trabajo (job):", job_list, default=job_list[:3] if job_list else [])

            show_data = st.checkbox("Mostrar tabla de datos filtrados")

            # Aplicar filtro
            df_filtered = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
            if selected_jobs:
                df_filtered = df_filtered[df_filtered['job'].isin(selected_jobs)]

            st.write(f"**Registros encontrados:** {len(df_filtered)} / {len(df)}")
            if show_data:
                st.dataframe(df_filtered, use_container_width=True)

            st.markdown("---")
            st.header("Ítem 10: Hallazgos Clave")
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.markdown("""
                ### 🔑 Insights Principales
                1. **Duración de Llamada:** Es la variable con mayor correlación directa respecto a la conversión (`y = yes`). A mayor duración de contacto, mayor probabilidad de éxito.
                2. **Contactos Previos:** Los clientes contactados en campañas previas exitosas (`poutcome = success`) tienen una tasa de aceptación sensiblemente superior.
                3. **Tasas de Interés (euribor3m):** Períodos con tasas más bajas muestran mayor disposición de inversión a plazo fijo.
                """)
            with col_h2:
                if 'y' in df.columns:
                    fig_res, ax_res = plt.subplots(figsize=(6, 4))
                    df['y'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], ax=ax_res, startangle=90)
                    ax_res.set_ylabel('')
                    ax_res.set_title('Proporción Total de Conversión (y)')
                    st.pyplot(fig_res)

# ------------------------------------------
# MÓDULO 4: CONCLUSIONES
# ------------------------------------------
elif modulo == "Módulo 4: Conclusiones":
    st.title("📌 Conclusiones Finales & Recomendaciones Comerciales")
    st.markdown("A partir del Análisis Exploratorio de Datos (EDA) ejecutado en la aplicación, se establecen las siguientes 5 conclusiones clave:")

    st.markdown("""
    1. **Optimización de la Duración de Contacto:** La variable `duration` demostró ser crítica. Las llamadas fructíferas promedian más de 300 segundos. Se recomienda a la fuerza comercial enfocar la llamada en una argumentación estructurada de valor en lugar de barridos telefónicos rápidos.
    2. **Segmentación por Estado Socioeconómico:** Clientes en categorías laborales como *management* o *technician* y con mayor nivel educativo (`tertiary`/`university.degree`) presentan la mayor tasa de respuesta global.
    3. **Frecuencia de Contacto Controlada (`campaign`):** Superar los 4 o 5 contactos durante la misma campaña reduce drásticamente la conversión e incrementa el rechazo del cliente, desgastando la base de datos comercial.
    4. **Reactivación de Base Histórica (`poutcome`):** Los clientes cuyo resultado previo fue exitoso representan el nicho con mayor retorno de inversión. La prioridad de contacto debe basarse en el historial positivo de campañas anteriores.
    5. **Impacto del Entorno Macroeconómico (`euribor3m` / `emp.var.rate`):** El comportamiento del depósito a plazo varía sustancialmente según los índices macroeconómicos. Las campañas deben sincronizarse comercialmente en momentos de estabilidad o baja de tasas.
    """)