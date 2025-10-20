import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd


def calcular_estado_por_periodo(
    df: pd.DataFrame,
    estado: str = "GRADUADO"
) -> pd.DataFrame:
    """
    Calcula el número total de estudiantes que alcanzan un estado específico 
    (por defecto, "GRADUADO") en cada periodo, considerando todos los programas
    del conjunto de datos de Jóvenes a la E.

    Para cada estudiante, se identifica el primer periodo en el que aparece el estado
    indicado y se contabiliza en dicho periodo.

    Args:
        df (pd.DataFrame):
            DataFrame con las columnas de estado por periodo, por ejemplo:
            'ESTADO_20212_CIERRE', 'ESTADO_20221_CIERRE', ..., 'ESTADO_20252_INICIO'.
        estado (str, opcional):
            Estado que se desea contar. Por defecto es "GRADUADO".

    Returns:
        pd.DataFrame:
            DataFrame con dos columnas:
            - **Periodo**: periodo en formato numérico (por ejemplo, 20212).
            - **<estado>**: número de estudiantes cuyo primer registro del estado
              indicado ocurrió en ese periodo.

    Ejemplo:
        >>> df_result = calcular_estado_por_periodo(df, estado="GRADUADO")
        >>> print(df_result)
            Periodo  GRADUADO
        0    20212         45
        1    20221         60
        2    20222         78
    """

    # --- Columnas esperadas (ajustar si cambia la estructura del dataset) ---
    cols: List[str] = [
        "ESTADO_20212_CIERRE",
        "ESTADO_20221_CIERRE",
        "ESTADO_20222_CIERRE",
        "ESTADO_20231_CIERRE",
        "ESTADO_20232_CIERRE",
        "ESTADO_20241_CIERRE",
        "ESTADO_20242_CIERRE",
        "ESTADO_20251_CIERRE",
        "ESTADO_20252_CIERRE",
        "ESTADO_20252_INICIO",
    ]

    # --- Inicializar contador por periodo ---
    periodos_count: Dict[str, int] = {col[7:12]: 0 for col in cols}

    # --- Filtrar estudiantes que tienen el estado en algún periodo ---
    df_filtrado = df[df[cols].eq(estado).any(axis=1)]

    # --- Función auxiliar para obtener el primer periodo donde aparece el estado ---
    def extraer_primer_periodo_estado(row: pd.Series, cols: List[str]) -> Optional[str]:
        for col in cols:
            if row[col] == estado:
                return col[7:12]
        return None

    # --- Contar cuántos alcanzan el estado en cada periodo ---
    for _, row in df_filtrado.iterrows():
        periodo = extraer_primer_periodo_estado(row, cols)
        if periodo is not None:
            periodos_count[periodo] += 1

    # --- Convertir resultados a DataFrame ordenado ---
    df_resultado = (
        pd.DataFrame(list(periodos_count.items()), columns=["Periodo", estado])
        .sort_values("Periodo")
        .reset_index(drop=True)
    )

    return df_resultado



def plot_estudiantes_por_periodo(
    df_resultado: pd.DataFrame,
    estado: str = "GRADUADO"
) -> None:
    """
    Genera un gráfico de barras que muestra el número de estudiantes 
    en un estado específico (por defecto, "GRADUADO") por periodo.

    El gráfico utiliza un fondo negro y barras moradas, con etiquetas blancas
    sobre cada barra indicando el valor correspondiente.

    Args:
        df_resultado (pd.DataFrame):
            DataFrame con al menos dos columnas:
            - **Periodo**: identificador del periodo (ej. 20212, 20221, ...).
            - **<estado>**: conteo de estudiantes en ese estado.
        estado (str, opcional):
            Estado a graficar. Debe coincidir con el nombre de la columna del DataFrame.
            Por defecto es "GRADUADO".

    Returns:
        None: La función muestra el gráfico directamente con `plt.show()`.

    Ejemplo:
        >>> df_resultado = calcular_graduados_por_periodo(df, estado="GRADUADO")
        >>> plot_estudiantes_por_periodo(df_resultado, estado="GRADUADO")
    """
    # --- Crear figura y ejes ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # --- Barras principales ---
    bars = ax.bar(df_resultado["Periodo"], df_resultado[estado], color="purple")

    # --- Etiquetas de valores ---
    ax.bar_label(
        bars,
        labels=[f"{v}" for v in df_resultado[estado]],
        label_type="edge",
        color="white",
        fontsize=10
    )

    # --- Título y etiquetas ---
    total = df_resultado[estado].sum()
    ax.set_title(
        f"{estado.capitalize()} por periodo (Total: {total})",
        fontsize=14,
        color="white",
        pad=15
    )
    ax.set_xlabel("Periodo", color="white", fontsize=12)
    ax.set_ylabel(f"Número de {estado.lower()}s", color="white", fontsize=12)

    # --- Estilo de ejes y grilla ---
    ax.tick_params(colors="white")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="white")

    plt.tight_layout()
    plt.show()


def plot_graduacion_stacked(
    tabla_graduacion,
    color_terminacion="deepskyblue",
    subtitulo="Programas Jóvenes a la E – Fuente: Base Maestra",
    path = "../output/"
):
    """
    Genera un gráfico de barras apiladas mostrando GRADUADO y TERMINACION DE MATERIAS.
    Fondo gris elegante, GRADUADO en morado y TERMINACION DE MATERIAS en color configurable.
    Muestra también el total de cada barra (arriba).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#2E2E2E")   # gris elegante (fondo figura)
    ax.set_facecolor("#2E2E2E")          # gris elegante (fondo eje)

    # Seleccionar solo columnas de interés
    df_plot = tabla_graduacion[["Periodo", "GRADUADO", "TERMINACION DE MATERIAS"]]
    df_plot = df_plot.set_index("Periodo")

    # Colores: morado y configurable
    colors = ["#6A5ACD", color_terminacion]

    # Gráfico apilado
    df_plot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        edgecolor="none"
    )

    # Agregar totales encima de las barras
    totals = df_plot.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(
            i, total + 1,  # posición encima de la barra
            f"{total:,}".replace(",", "."),
            ha="center",
            va="bottom",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    total_general = totals.sum()

    # Títulos
    plt.suptitle(
        f"Evolución observada de Graduación y Terminación por periodo (Total: {total_general:,})",
        fontsize=14,
        color="white",
        weight="bold"
    )

    ax.set_title(
        "Corte Inicio 2025-2",
        color='white',
        fontsize=12,
        pad=20
    )

    # Subtítulo opcional
    plt.figtext(
        0.5, -0.02,
        subtitulo,
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
        color="#C0C0C0"
    )

    # Ejes
    ax.set_xlabel("Periodo", color="white", fontsize=12)
    ax.set_ylabel("Número de estudiantes", color="white", fontsize=12)
    ax.tick_params(colors="white")

    # Gridlines suaves
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="white")

    # Quitar bordes superior y derecho
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Leyenda debajo del subtítulo del eje
    ax.legend(
        facecolor="#2E2E2E",
        labelcolor="white",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=2
    )

    plt.tight_layout()
    plt.savefig(path+"evolucion_observada_graduacion.pdf", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()

