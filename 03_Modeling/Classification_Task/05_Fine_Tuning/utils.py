import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


################################
#Helper functions

def merge_data(data, panel_data, universitario, tecnologico, tecnico):
    """
    Hace merge con la informacion complementaria.

    La informacion complementaria viene de filtrar la base maestra por:
    1. CUPO == 'ACTIVO'
    2. PERIODO_ULTIMO_ESTADO == 'ESTADO_20251_INICIO'
    3. No permitir nulos en:
     - "DOCUMENTO",
     - "PERIODOS_BD_SNIES",
     - "CREDITOS_PROGRAMA",
     - "ACU_CRED_CONSUMIDOS",
     - "ACU_CRED_APROBADOS",
     - "PORCENTAJE_CRED_PERDIDOS"
    
    """
    universitario_merge = pd.merge(
        panel_data['universitario'],
        universitario,
        on="DOCUMENTO")

    tecnologico_merge = pd.merge(
        panel_data['tecnologico'],
        tecnologico,
        on="DOCUMENTO")
    
    tecnico_merge = pd.merge(
        panel_data['tecnico'],
        tecnico,
        on="DOCUMENTO")

    

    data = {
        "universitario":universitario_merge,
        "tecnologico":tecnologico_merge,        
        "tecnico":tecnico_merge
    }
    
    return data

def add_rango_t(df):
    """
    Agrega una columna 'rango_t' al DataFrame,
    que representa el número de semestres observados
    para cada individuo (DOCUMENTO).

    Parámetros:
    ------------
    df : pandas.DataFrame
        Datos panel con columnas 'DOCUMENTO' y 't'.

    Retorna:
    ------------
    pandas.DataFrame con nueva columna 'rango_t'.
    """
    if "rango_t" in df.columns:
        return df
    
    rango_t = (
        df.groupby("DOCUMENTO")["semestre"]
        .agg(lambda x: x.max() - x.min() + 1)
        .rename("rango_t")
    )
    df = df.merge(rango_t, on="DOCUMENTO", how="left")
    return df

def calcular_semestre(
    df: pd.DataFrame,
    columna_tiempo: str = "periodo_orden"
):
    """
    Calcula el semestre actual de cada estudiante a partir de la columna 't',
    asignando 1 al primer registro, 2 al segundo, y así sucesivamente.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas 'DOCUMENTO' y 't'.

    Retorna
    -------
    pandas.DataFrame
        Mismo DataFrame con una nueva columna 'semestre'.
    """
    df = df.copy()
    df = df.sort_values(["DOCUMENTO", columna_tiempo])
    df["semestre"] = df.groupby("DOCUMENTO").cumcount() + 1
    return df


def flag_non_monotonic(df):
    """
    Crea una columna 'non_monotonic' que indica (True/False) si el pct_perd_acum
    de un individuo NO es monótonamente creciente a lo largo de t.

    Parámetros:
    ------------
    df : pandas.DataFrame
        Datos panel con columnas 'DOCUMENTO', 't' y 'pct_perd_acum'.

    Retorna:
    ------------
    pandas.DataFrame con nueva columna 'non_monotonic'.
    """
    if "non_monotonic" in df.columns:
        return df
        
    flags = (
        df.sort_values(["DOCUMENTO", "semestre"])
          .groupby("DOCUMENTO")[["pct_perd_acum"]]
          .apply(lambda g: not g["pct_perd_acum"].is_monotonic_increasing)
          .rename("non_monotonic")
    )

    df = df.merge(flags, on="DOCUMENTO", how="left")
    return df

def clip_perd_acum(
    row,
    epsilon=0.01,
    col_perdida_acum="pct_perd_acum",
    col_semestre="semestre",
    col_creditos_periodo="CREDITOS_PERIODO",
    col_total_periodos="PERIODOS_BD_SNIES"
):
    """
    Ajusta (clip) el valor acumulado de créditos perdidos de un estudiante si excede lo factible.

    Ejemplo de idea:
    ----------------
    Si un programa tiene 153 créditos y dura 9 periodos, se espera que el estudiante inscriba
    17 créditos por semestre. En el primer semestre, la pérdida máxima factible no puede ser
    mayor a 17/153 ≈ 0.11 (es decir, 11%).

    Regla de decisión:
    ------------------
    Si pct_perd_acum + epsilon > (CREDITOS_PERIODO * semestre) / (CREDITOS_PERIODO * PERIODOS_BD_SNIES),
    entonces pct_perd_acum se ajusta (clip) al valor factible.

    Parámetros
    ----------
    row : pandas.Series
        Fila del DataFrame sobre la cual se aplicará la función.
    epsilon : float, default=0.01
        Margen de tolerancia para permitir pequeños excesos antes de hacer el clip.
    col_perdida_acum : str
        Nombre de la columna con el % acumulado de créditos perdidos (en proporción 0–1).
    col_semestre : str
        Nombre de la columna con el semestre actual del estudiante.
    col_creditos_periodo : str
        Nombre de la columna con el número esperado de créditos por periodo.
    col_total_periodos : str
        Nombre de la columna con el número total de periodos en el programa.

    Retorna
    -------
    float
        Valor ajustado de pct_perd_acum (si corresponde), o el valor original.
    """
    # cálculo de pérdida máxima factible en este semestre
    maxima_perdida_factible = (
        row[col_creditos_periodo] * row[col_semestre]
    ) / (row[col_creditos_periodo] * row[col_total_periodos])

    valor = row[col_perdida_acum]
    if pd.notna(valor) and valor > epsilon +  maxima_perdida_factible:
        return maxima_perdida_factible
    return valor

def add_pct_perdida_previa(df, id_col="DOCUMENTO", semestre_col="semestre",
                           perdida_col="pct_perd_acum_clip",
                           new_col="pct_perdida_previa"):
    """
    Agrega columna con el % de pérdida del semestre anterior para cada individuo.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas de identificación, semestre y pérdida acumulada.
    id_col : str, default="DOCUMENTO"
        Nombre de la columna de identificación de cada estudiante.
    semestre_col : str, default="semestre"
        Columna que indica el número de semestre.
    perdida_col : str, default="pct_perd_acum_clip"
        Columna que contiene el % de pérdida acumulada en el semestre actual.
    new_col : str, default="pct_perdida_previa"
        Nombre de la nueva columna a crear.
    
    Returns
    -------
    pd.DataFrame
        DataFrame con columna adicional `pct_perdida_previa`.
    """
    df = df.copy()

    # Ordenamos por individuo y semestre
    df = df.sort_values([id_col, semestre_col])

    # Calculamos el valor del semestre anterior (shift dentro de cada individuo)
    df[new_col] = df.groupby(id_col)[perdida_col].shift(1)

    # Si es semestre 1 (o no existe semestre anterior), asignamos 0
    df[new_col] = df[new_col].fillna(0)

    return df



#################################
#GRAFICAS

import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. Promedio por grupo (líneas)
# ===============================
def plot_avg_pct_perd_by_group(df, group_col):
    """
    Grafica la evolución del promedio de pct_perd_acum a lo largo de t,
    separado por un grupo específico (ej. 'SEXO', 'NIVEL_SISBEN_4', 'MODALIDAD').

    Parámetros:
    ------------
    df : pandas.DataFrame
        Datos panel.
    group_col : str
        Columna para agrupar (ej. 'SEXO').
    """
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df,
        x='semestre',
        y='pct_perd_acum',
        hue=group_col,
        estimator='mean',
        errorbar='sd'
    )
    plt.title(f"Evolución promedio de créditos perdidos por {group_col}")
    plt.xlabel("Semestre")
    plt.ylabel("% Créditos perdidos acumulados")
    plt.legend(title=group_col)
    plt.grid(True, alpha=0.3)
    plt.show()

# ===============================
# 2. Distribuciones por semestre
# ===============================
def plot_pct_perd_distribution(df, kind='box'):
    """
    Muestra la distribución de pct_perd_acum por semestre (t),
    separado por NIVEL_FORMACION, usando boxplot o violinplot.
    También agrega una línea horizontal en y=0.1.

    Parámetros:
    ------------
    df : pandas.DataFrame
        Datos panel.
    kind : str, default='box'
        Tipo de gráfico: 'box' o 'violin'.
    """
    plt.figure(figsize=(12,6))
    if kind == 'box':
        sns.boxplot(data=df, x='semestre', y='pct_perd_acum', hue='NIVEL_FORMACION')
    elif kind == 'violin':
        sns.violinplot(data=df, x='semestre', y='pct_perd_acum', hue='NIVEL_FORMACION', split=True, inner='quartile')
    else:
        raise ValueError("kind debe ser 'box' o 'violin'")

    # Línea horizontal en y=0.1
    plt.axhline(y=0.1, color='red', linestyle='--', linewidth=1)

    plt.title("Distribución de % de créditos perdidos acumulados por semestre y nivel de formación")
    plt.xlabel("Semestre (t)")
    plt.ylabel("% Créditos perdidos acumulados")
    plt.legend(title="Nivel de formación", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()

# ===============================
# 3. Heatmap de cohortes
# ===============================
def plot_heatmap_cohort(df, cohort_col):
    """
    Genera un heatmap del promedio de pct_perd_acum según semestre (t)
    y la variable de cohorte (ej. 'NIVEL_FORMACION').

    Parámetros:
    ------------
    df : pandas.DataFrame
        Datos panel.
    cohort_col : str
        Columna para usar como cohorte (ej. 'NIVEL_FORMACION').
    """
    pivot = df.pivot_table(
        values='pct_perd_acum',
        index=cohort_col,
        columns='semestre',
        aggfunc='mean'
    )

    plt.figure(figsize=(12,8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd")
    plt.title(f"Heatmap de promedio % créditos perdidos por {cohort_col} y semestre")
    plt.xlabel("Semestre (t)")
    plt.ylabel(cohort_col)
    plt.show()

# ===============================
# 4. Trayectorias individuales
# ===============================
def plot_individual_trajectories(df, sample_size=50, random_state=42):
    """
    Grafica las trayectorias individuales del pct_perd_acum a lo largo de t,
    para cada DOCUMENTO. Si hay demasiados individuos, se toma una muestra.
    
    
    Parámetros:
    ------------
    df : pandas.DataFrame
    Datos panel con columnas 'DOCUMENTO', 't' y 'pct_perd_acum'.
    sample_size : int, default=50
    Número máximo de individuos a graficar.
    random_state : int, default=42
    Semilla para la aleatoriedad en el muestreo.
    """
    documentos = df["DOCUMENTO"].unique()
    if len(documentos) > sample_size:
        np.random.seed(random_state)
        documentos = np.random.choice(documentos, size=sample_size, replace=False)
        subset = df[df["DOCUMENTO"].isin(documentos)]
    
    
    plt.figure(figsize=(12,6))
    for doc_id, sub in subset.groupby("DOCUMENTO"):
        plt.plot(sub["semestre"], sub["pct_perd_acum"], alpha=0.5)
        
    
    plt.title("Trayectorias individuales de % de créditos perdidos acumulados")
    plt.xlabel("Semestre (t)")
    plt.ylabel("% Créditos perdidos acumulados")
    plt.grid(True, alpha=0.3)
    plt.show()


from PyPDF2 import PdfMerger

def merge_pdfs(pdf_paths, output_path="resultados_completos.pdf"):
    """
    Une múltiples archivos PDF en un solo documento.

    Parameters
    ----------
    pdf_paths : list of str
        Lista con las rutas de los PDFs a concatenar (en orden).
    output_path : str
        Ruta de salida del PDF combinado.
    """
    merger = PdfMerger()
    for pdf in pdf_paths:
        merger.append(pdf)
    merger.write(output_path)
    merger.close()
    print(f"✅ PDF combinado guardado en: {output_path}")

