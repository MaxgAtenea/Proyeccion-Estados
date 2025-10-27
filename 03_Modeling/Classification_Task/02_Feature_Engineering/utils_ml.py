import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

######################## LIMPIEZA #########################

def combine_categories(df, column, categories_to_combine, new_category="OTRA"):
    """
    Combina categorías (incluyendo NaN si se especifica) en una sola nueva categoría.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna a modificar.
    column : str
        Nombre de la columna.
    categories_to_combine : list
        Lista de categorías (y opcionalmente NaN) a combinar.
    new_category : str, default="OTRA"
        Nombre de la nueva categoría.

    Returns
    -------
    pd.DataFrame
        DataFrame con las categorías combinadas.
    """
    df = df.copy()
    # Reemplazar categorías normales
    df[column] = df[column].replace(categories_to_combine, new_category)
    # Si 'nan' está en la lista → convertir NaN a la nueva categoría
    if any(pd.isna(cat) for cat in categories_to_combine):
        df[column] = df[column].fillna(new_category)
    return df




def categorize_age_ranges(
    df,
    age_col="EDAD_DEL_CALCULO",
    new_col="EDAD_RANGO",
    bins=None,
    labels=None
):
    """
    Limpia y categoriza la edad en rangos definidos.

    Pasos:
    1. Reemplaza "29 O MAS" por 29.
    2. Convierte la columna de edad a numérica.
    3. Clasifica en rangos usando pd.cut.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos.
    age_col : str
        Nombre de la columna con la edad original.
    new_col : str
        Nombre de la nueva columna categórica de rangos.
    bins : list[int]
        Lista de bordes de los intervalos.
    labels : list[str]
        Etiquetas de los rangos. Deben ser len(bins)-1.

    Retorna
    -------
    df : pd.DataFrame
        DataFrame con la nueva columna de rangos.
    """

    # Copia para no alterar el original
    df = df.copy()

    # Reemplazar "29 O MAS" por 29
    df[age_col] = df[age_col].replace("29 O MAS", 29)

    # Convertir a numérico
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

    # Definir bins y labels por defecto si no se pasan
    if bins is None:
        bins = [14, 19, 23, 28, 29]  # incluye los bordes superiores
    if labels is None:
        labels = ["15-19", "20-23", "24-28", "29"]

    # Categorizar con pd.cut
    df[new_col] = pd.cut(df[age_col], bins=bins, labels=labels, include_lowest=True, right=True)

    return df

def remove_outliers_iqr(df, column, k=1.5):
    """
    Elimina outliers de un DataFrame según el rango intercuartílico (IQR).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    column : str
        Nombre de la columna numérica sobre la cual eliminar outliers.
    k : float, opcional (default=1.5)
        Factor multiplicador del IQR para definir los límites.
        Valores más altos eliminan menos puntos, valores más bajos eliminan más.

    Retorna
    -------
    pd.DataFrame
        Nuevo DataFrame sin los outliers.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    return df_filtered

def display_confusion_matrix(
    y_test,
    y_hat,
    labelencoder,
    normalize_parameter= "true",
    normalize_title = "recall",
    savefig_path=None
):

    cm = confusion_matrix(
        y_test,
        y_hat,
        labels=np.arange(len(labelencoder.classes_)),
        normalize=normalize_parameter
    )

    disp = ConfusionMatrixDisplay(cm, display_labels=labelencoder.classes_)
    if normalize_parameter:
        disp.plot(values_format=".3f", cmap="Blues")
        plt.title(f"Matriz de confusión (test) — Normalizada: ({normalize_title})")
    else:
        disp.plot(cmap="Blues")
        plt.title("Matriz de confusión (test)")

    # Guardar si se especifica path
    if savefig_path is not None:
        plt.savefig(savefig_path, format="pdf", bbox_inches="tight")

    plt.show()


import math
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, classification_report

def per_group_confusion_and_metrics(X, y_true, y_pred, group_col, label_encoder=None):
    """
    Devuelve:
      - cms: dict {grupo: confusion_matrix (counts)}
      - metrics_df: DataFrame con métricas por grupo (balanced_accuracy, f1_macro, n)
    Requiere que y_true, y_pred sean codificados con los mismos labels (int).
    Si label_encoder se pasa, sus clases se usan para ordenar columnas/filas.
    """
    df = X.copy()
    df = df.reset_index(drop=True)
    df["_y"] = np.asarray(y_true)
    df["_yhat"] = np.asarray(y_pred)

    grupos = sorted(df[group_col].dropna().unique(), key=lambda x: (str(x)))
    n_classes = len(label_encoder.classes_) if label_encoder is not None else len(np.unique(np.concatenate([y_true, y_pred])))
    labels_idx = np.arange(n_classes)

    cms = {}
    metrics = []
    for g in grupos:
        sub = df[df[group_col] == g]
        if sub.shape[0] == 0:
            continue
        y_t = sub["_y"].to_numpy()
        y_p = sub["_yhat"].to_numpy()

        # Confusion matrix with fixed order of labels (so matrices son comparables)
        cm = confusion_matrix(y_t, y_p, labels=labels_idx)
        cms[g] = cm

        # Métricas
        ba = balanced_accuracy_score(y_t, y_p)
        f1m = f1_score(y_t, y_p, average="macro", zero_division=0)
        support = len(y_t)
        metrics.append({"group": g, "n": support, "balanced_accuracy": ba, "f1_macro": f1m})

    metrics_df = pd.DataFrame(metrics).set_index("group")
    return cms, metrics_df, labels_idx

import textwrap

def plot_confusion_matrices_by_group(
    cms, 
    labels, 
    normalize_mode="recall",   # "recall", "precision", "accuracy" o None
    max_cols=4, 
    cmap="Blues", 
    fmt=".2f", 
    savefig_path=None
):
    """
    Dibuja heatmaps de la matriz de confusión por grupo.
      - cms: dict {group: cm (counts)}
      - labels: list/array con nombres de clases (strings) en el mismo orden que los índices de cm
      - normalize_mode: 
            "recall" -> normaliza por filas (comparables a recall)
            "precision" -> normaliza por columnas (comparables a precision)
            "accuracy" -> divide por total (proporciones globales)
            None -> cuentas absolutas
      - savefig_path: ruta para guardar la figura (ej. 'conf_matrices.png'). 
                      Si None, solo muestra en pantalla.
    """
    groups = list(cms.keys())
    n = len(groups)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()

    vmax = 1.0 if normalize_mode else None

    for ax, g in zip(axes, groups):
        cm = cms[g].astype(float)

        # --- Normalización según el modo ---
        if normalize_mode == "recall":
            row_sums = cm.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = np.divide(cm, row_sums, where=row_sums!=0)
            cm_to_plot = cm_norm
            title = f"{g} (n={int(cm.sum())}) — filas normalizadas (recall)"
        elif normalize_mode == "precision":
            col_sums = cm.sum(axis=0, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = np.divide(cm, col_sums, where=col_sums!=0)
            cm_to_plot = cm_norm
            title = f"{g} (n={int(cm.sum())}) — columnas normalizadas (precision)"
        elif normalize_mode == "accuracy":
            cm_to_plot = cm / cm.sum() if cm.sum() > 0 else cm
            title = f"{g} (n={int(cm.sum())}) — normalizado global (accuracy)"
        else:
            cm_to_plot = cm
            title = f"{g} (n={int(cm.sum())}) — cuentas"

        # --- Plot ---
        sns.heatmap(
            cm_to_plot, annot=True, fmt=fmt if normalize_mode else "d",
            xticklabels=labels, yticklabels=labels,
            cbar=False, ax=ax, vmin=0, vmax=vmax, cmap=cmap
        )
        
        title_wrapped = "\n".join(textwrap.wrap(title, width=35))
        ax.set_title(title_wrapped)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")

    # quitar ejes vacíos
    for ax in axes[len(groups):]:
        fig.delaxes(ax)

    plt.tight_layout()

    # Guardar si se especifica ruta
    if savefig_path is not None:
        plt.savefig(savefig_path, format="pdf", bbox_inches="tight")

    plt.show()


from sklearn.metrics import classification_report

def report_per_group(X, y_true, y_pred, group_col, label_encoder=None):
    """
    Imprime classification_report por cada valor único de group_col.

    Parameters
    ----------
    X : pd.DataFrame
        Conjunto de features que incluye la columna group_col.
    y_true : array-like
        Etiquetas verdaderas alineadas posicionalmente con X.
    y_pred : array-like
        Etiquetas predichas alineadas posicionalmente con X.
    group_col : str
        Nombre de la columna del DataFrame X que define los grupos.
    label_encoder : LabelEncoder, optional
        Si se pasa, se usan sus clases como target_names en el reporte.
    """
    target_names = label_encoder.classes_ if label_encoder is not None else None

    for g in sorted(X[group_col].dropna().unique(), key=lambda x: str(x)):
        mask = X[group_col] == g
        y_t, y_p = y_true[mask], y_pred[mask]
        if len(y_t) == 0:
            continue
        print(f"\n--- {group_col}: {g} (n={len(y_t)}) ---")
        print(classification_report(y_t, y_p, target_names=target_names, zero_division=0))


def check_alignment(X_train, X_test, y_train, y_test, y_pred=None, grupos_train=None, grupos_test=None):
    '''
    Verifica que las matrices X y y estén alineadas para evitar errores en la interpretación de las matrices de confusion
    '''
    errors = []

    # Longitudes
    if len(X_train) != (len(y_train) if not isinstance(y_train, np.ndarray) else len(y_train)):
        errors.append(f"Mismatch train len: X_train={len(X_train)} vs y_train={len(y_train)}")
    if len(X_test) != (len(y_test) if not isinstance(y_test, np.ndarray) else len(y_test)):
        errors.append(f"Mismatch test len: X_test={len(X_test)} vs y_test={len(y_test)}")

    # Índices de X deben ser 0..n-1 si asumes index positional
    if not (X_train.index.equals(pd.RangeIndex(len(X_train)))):
        errors.append("X_train index no es RangeIndex 0..n-1 (usar reset_index(drop=True))")
    if not (X_test.index.equals(pd.RangeIndex(len(X_test)))):
        errors.append("X_test index no es RangeIndex 0..n-1 (usar reset_index(drop=True))")

    # y_pred length
    if y_pred is not None and len(y_pred) != len(y_test):
        errors.append(f"Mismatch y_pred vs y_test: {len(y_pred)} vs {len(y_test)}")

    # grupos lengths si se pasan
    if grupos_train is not None and len(grupos_train) != len(X_train):
        errors.append(f"Mismatch grupos_train vs X_train: {len(grupos_train)} vs {len(X_train)}")
    if grupos_test is not None and len(grupos_test) != len(X_test):
        errors.append(f"Mismatch grupos_test vs X_test: {len(grupos_test)} vs {len(X_test)}")

    if errors:
        print("❌ Problemas detectados:")
        for e in errors:
            print(" -", e)
        return False
    else:
        print("✅ Checks passed: X/y indices y longitudes están alineados para slicing posicional.")
        return True




def report_class_proportions(y, train_idx, test_idx, class_labels=None):
    """
    Muestra las proporciones de clases en Train y Test, con etiquetas originales si se dan.

    Parámetros
    ----------
    y : array-like o pd.Series
        Vector de etiquetas.
    train_idx, test_idx : arrays
        Índices de train y test (ej. generados por GroupShuffleSplit).
    class_labels : array-like o None
        Etiquetas originales de las clases (ej. le.classes_).
    """
    y_series = pd.Series(y)

    def proportions(idx):
        return y_series.iloc[idx].value_counts(normalize=True).sort_index()

    train_props = proportions(train_idx)
    test_props = proportions(test_idx)

    # Si hay etiquetas originales, reemplazar índices por nombres
    if class_labels is not None:
        train_props.index = [f"{i} → {label}" for i, label in zip(train_props.index, class_labels)]
        test_props.index = [f"{i} → {label}" for i, label in zip(test_props.index, class_labels)]

    print("Proporciones de clases en el \033[1mTrain\033[0m")
    print(train_props)
    print("\nProporciones de clases en el \033[1mTest\033[0m")
    print(test_props)

###################################################

import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplots(
    data,
    column,
    by,
    title=None,
    xlabel=None,
    ylabel=None,
    hline=None,
    hue=None,          # optional, in case you ever need grouping
    save_path=None,
    figsize=(10, 8),
    show_table=True,   # <--- nuevo parámetro
):
    """
    Plots boxplots from a DataFrame, optionally with a summary table.

    Parameters
    ----------
    data : DataFrame
        Dataset to plot.
    column : str
        Numerical column to plot.
    by : str
        Categorical column to group by (x-axis).
    title, xlabel, ylabel : str, optional
        Titles and axis labels.
    hline : float or None
        Adds a horizontal reference line if provided.
    hue : str or None
        Optional grouping variable.
    save_path : str or None
        If given, saves the figure instead of only showing.
    figsize : tuple
        Figure size.
    show_table : bool
        If True, shows a table with descriptive stats under the plot.
    """

    # --- Crear figura con subplots: (gráfico arriba, tabla abajo) ---
    fig, (ax_plot, ax_table) = plt.subplots(
        2, 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]}  # más espacio para el gráfico
    )

    # --- Boxplot ---
    sns.boxplot(
        data=data,
        x=by,
        y=column,
        hue=hue,
        width=0.6,
        showfliers=True,
        ax=ax_plot
    )

    # --- Aesthetics ---
    sns.despine()
    ax_plot.grid(axis="y", linestyle="--", alpha=0.5)

    if hline is not None:
        ax_plot.axhline(hline, ls="--", color="red", lw=1)

    ax_plot.set_title(title if title else f"Distribución de {column} por {by}", fontsize=14, fontweight="bold")
    ax_plot.set_xlabel(xlabel if xlabel else by, fontsize=12)
    ax_plot.set_ylabel(ylabel if ylabel else column, fontsize=12)
    ax_plot.set_xticks(ax_plot.get_xticks())
    ax_plot.set_xticklabels(ax_plot.get_xticklabels(), rotation=30, ha="right")


    # --- Legend ---
    if hue:
        ax_plot.legend(title=hue, bbox_to_anchor=(1.05, 0.8), loc="upper left")
    else:
        leg = ax_plot.get_legend()
        if leg:
            leg.remove()

    # --- Tabla ---
    ax_table.axis("off")
    if show_table:
        summary = data.groupby(by)[column].describe().round(2)
        table = ax_table.table(
            cellText=summary.values,
            colLabels=summary.columns,
            rowLabels=summary.index,
            cellLoc="center",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.show()



def save_filtered_data_pkls(filtered_data, base_path="../data"):
    """
    Guarda los DataFrames en filtered_data como archivos .pkl.

    Parameters
    ----------
    filtered_data : dict[str, pd.DataFrame]
        Diccionario con los subconjuntos (ej. {"universitario": df1, "tecnico": df2, ...}).
    base_path : str
        Carpeta donde guardar los .pkl (default=".").

    Returns
    -------
    None
    """
    for key, df in filtered_data.items():
        file_path = f"{base_path}/{key}.pkl"
        df.to_pickle(file_path)
        print(f"✅ Guardado: {file_path}")



def summarize_by_group(df, group_col="CONVOCATORIA", skip_cols=None):
    """
    Print descriptive statistics and % of missing values 
    for each variable in a dataframe grouped by `group_col`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_col : str, default="CONVOCATORIA"
        Column to group by.
    skip_cols : list, optional
        Columns to skip (besides the group column).
    """

    
    if skip_cols is None:
        skip_cols = []
    cols = [c for c in df.columns if c not in ([group_col] + skip_cols)]

    for col in cols:
        print("=" * 50)
        print(f"📊 Variable: {col}")
        print(df.groupby(group_col)[col].describe())

        print("\nMissing values (% by group):")
        missing_pct = (
            df.groupby(group_col)[col]
            .apply(lambda x: x.isna().mean() * 100)
        )
        print(missing_pct.round(2))

def category_distribution_by_convocatoria(df, category_col, normalize=True, decimals=2):
    """
    Calcula la distribución de una variable categórica por convocatoria.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos.
    category_col : str
        Nombre de la variable categórica a analizar.
    normalize : bool, opcional (default=True)
        Si True, devuelve porcentajes. Si False, devuelve conteos absolutos.
    decimals : int, opcional (default=2)
        Número de decimales para redondear los resultados.

    Retorna
    -------
    pd.DataFrame
        Tabla con la distribución de la variable por convocatoria.
    """
    dist = (
        df.groupby("semestre")[category_col]
        .value_counts(normalize=normalize, dropna=False)
        .unstack(fill_value=0)
    )
    if normalize:
        dist *= 100
    # Agregar columna TOTAL (conteo real de personas por convocatoria)
    totals = df.groupby("semestre").size()
    dist["TOTAL"] = totals
    
    return dist.round(decimals)



def missing_percentage(df, group_col, target_col, decimals=2):
    """
    Calculate percentage of missing values in `target_col`
    grouped by `group_col`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_col : str
        Column to group by.
    target_col : str
        Column where NaN percentage is calculated.
    decimals : int, default=2
        Number of decimals to round.

    Returns
    -------
    pd.Series
        Percentage of missing values per group.
    """
    nan_pct = (
        df.groupby(group_col)[target_col]
        .apply(lambda x: x.isna().mean() * 100)
        .round(decimals)
    )
    return nan_pct

def plot_grouped_boxplot(
    df,
    x_col,
    y_col,
    hue_col,
    height=6,
    aspect=1.5,
    title=None,
    ylabel=None,
    xlabel=None,
    hline=None,  # <--- nuevo parámetro para trazar línea horizontal
):
    """
    Genera un boxplot con colores (hue) y leyenda usando seaborn.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos.
    x_col : str
        Variable en el eje X (ej. "CONVOCATORIA").
    y_col : str
        Variable numérica para el eje Y (ej. "PORCENTAJE_CRED_PERDIDOS").
    hue_col : str
        Variable categórica que define los colores (ej. "MODALIDAD").
    height : int, opcional
        Alto de la figura (default=6).
    aspect : float, opcional
        Relación ancho/alto de cada gráfico (default=1.5).
    title : str, opcional
        Título general del gráfico.
    ylabel : str, opcional
        Etiqueta del eje Y.
    xlabel : str, opcional
        Etiqueta del eje X.
    hline : float, opcional
        Si se pasa, traza una línea horizontal en esa coordenada Y.
    """

    g = sns.catplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        kind="box",
        height=height,
        aspect=aspect,
    )

    # etiquetas personalizadas
    g.set_axis_labels(xlabel or x_col, ylabel or y_col)
    g.add_legend(title=hue_col)

    if title:
        plt.title(title, y=1.05)

    # trazar línea horizontal
    if hline is not None:
        for ax in g.axes.flat:
            ax.axhline(hline, ls="--", color="red", lw=1.5)

    plt.show()



def plot_grouped_countplots(
    df,
    x_col,
    hue_cols,
    height=5,
    aspect=3,
    table_cols=None,      # None => todas, [] => ninguna
    normalize=False,      # True => proporciones en vez de conteos
    save_path=None,
    global_title=None,    # <--- título global
    plot_titles=None      # <--- lista/dict de títulos por subplot
):
    """
    Grafica countplots de varias variables (hue_cols) contra x_col,
    con su tabla de distribución debajo.
    """
    n = len(hue_cols)
    fig, axes = plt.subplots(
        nrows=2 * n,
        figsize=(aspect * height, height * n),
        gridspec_kw={'height_ratios': [3, 1] * n}
    )
    if n == 1:
        axes = list(axes)

    for i, hue_col in enumerate(hue_cols):
        ax_plot, ax_table = axes[2*i], axes[2*i + 1]

        # --- Gráfico ---
        if normalize:
            sns.histplot(
                data=df, x=x_col, hue=hue_col,
                stat="probability", multiple="stack",
                discrete=True, shrink=0.8, ax=ax_plot,
                legend=True
            )
            ax_plot.set_ylabel("Proporción")
        else:
            sns.countplot(
                data=df, x=x_col, hue=hue_col,
                ax=ax_plot,
                legend=True
            )
            ax_plot.set_ylabel("Conteo")
        
        # --- Título del subplot ---
        if plot_titles is None:
            ax_plot.set_title(f"{hue_col} por {x_col}")
        elif isinstance(plot_titles, dict):
            ax_plot.set_title(plot_titles.get(hue_col, f"{hue_col} por {x_col}"))
        elif isinstance(plot_titles, (list, tuple)):
            if i < len(plot_titles):
                ax_plot.set_title(plot_titles[i])
            else:
                ax_plot.set_title(f"{hue_col} por {x_col}")

        # Ajustar la leyenda solo si hay elementos
        handles, labels = ax_plot.get_legend_handles_labels()
        if handles:
            ax_plot.legend(handles, labels, title=hue_col,
                           bbox_to_anchor=(1.05, 0.8), loc="upper left")

        # --- Tabla ---
        ax_table.axis("off")
        show_table = (table_cols is None) or (hue_col in table_cols)
        if show_table:
            dist = df.groupby([x_col, hue_col]).size().unstack(fill_value=0)
            if normalize:
                dist = dist.div(dist.sum(axis=1), axis=0).round(3)
            ax_table.table(
                cellText=dist.values,
                colLabels=dist.columns,
                rowLabels=dist.index,
                cellLoc="center", loc="center"
            )

    # --- Título global ---
    if global_title:
        fig.suptitle(global_title, fontsize=16, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()



def plot_multiple_grouped_boxplots(
    df,
    x_col,
    y_col,
    hue_cols,
    hline=None,
    height=5,
    aspect=3,
    table_cols=None,   # (None => todas, [] => ninguna)
    plot_ratio=3,      # altura relativa del gráfico (por defecto 3)
    table_ratio=1,      # altura relativa de la tabla (por defecto 1)
    save_path="../output/distribucion_variables_categoricas.pdf" 
):
    """
    Genera múltiples boxplots (uno por variable categórica en hue_cols) 
    en subplots apilados verticalmente dentro de la misma figura.
    Debajo de cada gráfico se muestra una tabla con la distribución
    porcentual de la variable categórica por x_col (opcional, controlada por table_cols).

    Parámetros (resumen)
    --------------------
    df : pd.DataFrame
    x_col, y_col : str
    hue_cols : list[str]
    hline : float or None
    height, aspect : tamaños
    table_cols : list[str] or None
        Si None -> mostrar tabla para todas las hue_cols.
        Si lista -> mostrar tabla solo para las hue_cols en la lista.
        Si lista vacía -> no mostrar tablas.
    plot_ratio, table_ratio : ints
        Proporción de altura entre gráfico y tabla; por ejemplo (3,1) da más espacio al gráfico.
    """

    n = len(hue_cols)

    # Creamos 2 filas por cada hue_col: (gráfico, tabla)
    fig, axes = plt.subplots(
        nrows=2 * n,
        ncols=1,
        figsize=(aspect * height, height * n),
        gridspec_kw={'height_ratios': [plot_ratio, table_ratio] * n}
    )

    # Cuando n == 1, plt.subplots devuelve un array de shape (2,), convertir a lista para indexar
    if n == 1:
        axes = list(axes)

    for i, hue_col in enumerate(hue_cols):
        ax_plot = axes[2*i]
        ax_table = axes[2*i + 1]

        # --- Boxplot ---
        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax_plot)
        ax_plot.set_title(f"{y_col} por {x_col} y {hue_col}")
        ax_plot.set_xlabel(x_col)
        ax_plot.set_ylabel(y_col)

        if hline is not None:
            ax_plot.axhline(hline, ls="--", color="red", lw=1)

        ax_plot.legend(title=hue_col, bbox_to_anchor=(1.05, 0.8), loc="upper left")

        # --- Tabla (condicional) ---
        show_table = (table_cols is None) or (hue_col in table_cols)
        ax_table.axis("off")
        if show_table:
            dist_table = category_distribution_by_convocatoria(df, hue_col)
            ax_table.table(
                cellText=dist_table.values,
                colLabels=dist_table.columns,
                rowLabels=dist_table.index,
                cellLoc="center",
                loc="center"
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.show()
        
    else:
        plt.show()


def plot_multiple_grouped_distributions(
    df,
    x_col,
    cont_cols,
    height=5,
    aspect=3,
    table_cols=None,   # (None => todas, [] => ninguna)
    plot_ratio=3,
    table_ratio=1,
    save_path="../output/distribucion_variables_continuas.pdf"
):
    """
    Genera múltiples distribuciones (una por variable continua en cont_cols) 
    en subplots apilados verticalmente dentro de la misma figura.
    Debajo de cada gráfico se muestra una tabla con estadísticas descriptivas
    por x_col (opcional, controlada por table_cols).

    Parámetros
    ----------
    df : pd.DataFrame
    x_col : str
        Variable categórica usada para agrupar (ej. "CONVOCATORIA").
    cont_cols : list[str]
        Lista de variables continuas.
    hline : float or None
        Si se pasa, dibuja una línea horizontal (útil si la métrica es comparable).
    height : int
        Alto de cada grupo de (gráfico+tabla).
    aspect : float
        Relación ancho/alto de la figura.
    table_cols : list[str] or None
        Si None -> mostrar tabla para todas las cont_cols.
        Si lista -> mostrar tabla solo para las cont_cols en la lista.
        Si lista vacía -> no mostrar tablas.
    plot_ratio, table_ratio : ints
        Proporción de altura entre gráfico y tabla.
    save_path : str or None
        Ruta donde guardar el PDF. Si None, solo muestra la figura.
    """

    n = len(cont_cols)
    
    fig, axes = plt.subplots(
        nrows=2 * n,
        ncols=1,
        figsize=(aspect * height, height * n),
        gridspec_kw={'height_ratios': [plot_ratio, table_ratio] * n}
    )
    
    if n == 1:
        axes = list(axes)
    
    for i, cont_col in enumerate(cont_cols):
        ax_plot = axes[2 * i]
        ax_table = axes[2 * i + 1]
    
        # --- Distribución ---
        sns.kdeplot(data=df, x=cont_col, hue=x_col, fill=True, ax=ax_plot, common_norm=False)
        
        handles, labels = ax_plot.get_legend_handles_labels()
        if handles:  # solo mostrar leyenda si hay algo
            ax_plot.legend(title=x_col, bbox_to_anchor=(1.05, 0.8), loc="upper left")
            
        ax_plot.set_title(f"Distribución de {cont_col} por {x_col}")
        ax_plot.set_xlabel(cont_col)
        ax_plot.set_ylabel("Densidad")
        
        # --- Tabla (condicional) ---
        show_table = (table_cols is None) or (cont_col in table_cols)
        ax_table.axis("off")
        if show_table:
            dist_table = df.groupby(x_col)[cont_col].describe().round(2)
            ax_table.table(
                cellText=dist_table.values,
                colLabels=dist_table.columns,
                rowLabels=dist_table.index,
                cellLoc="center",
                loc="center"
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.show()
    else:
        plt.show()

import math
def plot_estado_transition_by_semestre(df, semestre_col="semestre"):
    """
    Dibuja heatmaps de la distribución de estado_next según estado, separados por semestre.
    Cada heatmap muestra proporciones (normalizado por fila).
    """
    semestres = sorted(df[semestre_col].dropna().unique())
    n = len(semestres)
    
    fig, axes = plt.subplots(
        nrows=math.ceil(n/2), ncols=2, 
        figsize=(12, 5*math.ceil(n/2))
    )
    axes = axes.flatten()

    for i, sem in enumerate(semestres):
        sub = df[df[semestre_col] == sem]

        tabla = pd.crosstab(
            sub["estado"],
            sub["estado_next"],
            normalize="index"
        ).round(3)

        sns.heatmap(
            tabla, annot=True, fmt=".2f", cmap="Blues", 
            cbar=False, ax=axes[i]
        )
        axes[i].set_title(f"Distribución transición de estados — Semestre {sem}", fontsize=12)
        axes[i].set_ylabel("Estado actual")
        axes[i].set_xlabel("Estado siguiente")

    # quitar ejes sobrantes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

############## SPLIT ##############################

def plot_distribution_with_stats(series, bins=50, height=5, aspect=2):
    """
    Plot the density (histogram + KDE) of a variable, with a table below
    showing its descriptive statistics.
    
    Parameters
    ----------
    series : pd.Series
        Variable to visualize.
    bins : int, optional
        Number of bins for histogram.
    height : int, optional
        Height of the plot.
    aspect : float, optional
        Aspect ratio of the figure.
    """
    stats = series.describe().to_frame(name="Value")
    
    # Create figure with 2 rows: top=plot, bottom=table
    fig, (ax_plot, ax_table) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(aspect * height, height),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # --- Distribution plot ---
    sns.histplot(series, bins=bins, kde=True, ax=ax_plot)
    ax_plot.set_title(f"Distribution of {series.name}")
    ax_plot.set_xlabel(series.name)
    ax_plot.set_ylabel("Frequency")

    # --- Table ---
    ax_table.axis("off")
    table = ax_table.table(
        cellText=stats.values.round(4),
        rowLabels=stats.index,
        colLabels=stats.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    plt.tight_layout()
    plt.show()

# --- Función para splits por semestre ---
def make_splits_by_semester(df, X, y, test_size=0.2, random_state=42):
    splits = {}
    
    for s in df["semestre"].unique():
        # Filtrar por semestre
        data_x = X.query("semestre == @s").copy()
        data_y = y.loc[data_x.index].copy()

        if len(data_y.unique()) < 2:
            stratify = None
        else:
            stratify = data_y

        X_train, X_test, y_train, y_test = train_test_split(
            data_x, data_y,
            test_size=test_size,
            stratify=stratify, #para manejar mejor el desbalance de clases
            random_state=random_state
        )

        splits[s] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
    
    return splits



############## OHE helper functions  ##############################

import pandas as pd
from scipy.sparse import issparse

def get_encoded_dataframe(pipeline, X):
    """
    Return the DataFrame after preprocessing (e.g., OneHotEncoder).
    
    Parameters
    ----------
    pipeline : sklearn.Pipeline
        The fitted pipeline containing the preprocessing step ("preprocess").
    X : pd.DataFrame
        The original features before preprocessing.
    
    Returns
    -------
    pd.DataFrame
        The encoded DataFrame with expanded feature names.
    """
    # Extract preprocessor
    preprocessor = pipeline.named_steps["preprocess"]
    
    # Transform X
    X_transformed = preprocessor.transform(X)
    
    # Handle sparse/dense
    if issparse(X_transformed):
        X_transformed = X_transformed.toarray()
    
    # Get feature names
    encoded_features = preprocessor.get_feature_names_out()
    
    # Return DataFrame
    return pd.DataFrame(
        X_transformed,
        columns=encoded_features,
        index=X.index
    )


def decode_onehot_row(encoded_row, preprocessor, categoricas):
    """
    Decode a single one-hot encoded row back to original categorical values.
    
    Parameters
    ----------
    encoded_row : pd.Series
        A row from the encoded DataFrame (after ColumnTransformer).
    preprocessor : fitted ColumnTransformer
        Your pipeline's preprocessing step (pipeline.named_steps["preprocess"]).
    categoricas : list of str
        The original categorical feature names.
    
    Returns
    -------
    dict
        Mapping of original categorical variables to their decoded values.
    """
    # Get the fitted encoder
    encoder = preprocessor.named_transformers_["cat"]
    categories = encoder.categories_

    decoded = {}
    for col, cats in zip(categoricas, categories):
        # Find the one-hot columns for this variable
        relevant_cols = [c for c in encoded_row.index if c.startswith(f"cat__{col}_")]
        
        # If all 0 → baseline category (the one dropped due to drop="first")
        if all(encoded_row[relevant_cols] == 0):
            decoded[col] = cats[0]  # baseline category
        else:
            # Find the active category
            active_col = [c for c in relevant_cols if encoded_row[c] == 1][0]
            decoded[col] = active_col.split(f"cat__{col}_")[1]
    
    return decoded


############## MODEL EVAL ##############################
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


def print_regression_metrics(y_true, y_pred, dataset_name="Dataset"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"📊 {dataset_name}")
    print(f"   MAE : {mae:.3f}")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   R²  : {r2:.3f}\n")

import matplotlib.pyplot as plt
import seaborn as sns

def plot_error_scatter(
    y_true,
    y_pred,
    title="Error vs True Values",
    xlabel="True Values",
    ylabel="Prediction Error (y_pred - y_true)",
    ax = None
):
    """
    Scatter plot of prediction errors against true values.

    Parameters
    ----------
    y_true : array-like
        Ground truth values (dependent variable).
    y_pred : array-like
        Predicted values from the model.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis (default: "True Values").
    ylabel : str, optional
        Label for the y-axis (default: "Prediction Error").
    """
    errors = y_pred - y_true
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.scatter(x=y_true, y=errors, alpha=0.6, edgecolor="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.3)


def plot_observed_vs_prediction_scatter(
    y_true,
    y_pred,
    title="Observed vs prediction",
    xlabel="Observed Values",
    ylabel="Predicted values",
    ax= None
):
    """
    Scatter plot of prediction errors against true values.

    Parameters
    ----------
    y_true : array-like
        Ground truth values (dependent variable).
    y_pred : array-like
        Predicted values from the model.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis (default: "True Values").
    ylabel : str, optional
        Label for the y-axis (default: "Prediction Error").
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.scatter(x=y_true, y=y_pred, alpha=0.6, edgecolor="none")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.3)



def plot_feature_importance(pipeline, categoricas, numericas, top_n=None, figsize=(10, 6)):
    """
    Genera un gráfico de importancia de variables a partir de un pipeline entrenado 
    con OneHotEncoder y XGBRegressor.

    Parámetros
    ----------
    pipeline : sklearn.Pipeline
        Pipeline entrenado que contiene el preprocesamiento (con OneHotEncoder en 'cat') 
        y un modelo (con atributo feature_importances_).
    categoricas : list[str]
        Lista de columnas categóricas originales.
    numericas : list[str]
        Lista de columnas numéricas originales.
    top_n : int or None, opcional
        Número máximo de variables a mostrar. Si None, muestra todas.
    figsize : tuple, opcional
        Tamaño de la figura para el gráfico.
    """
    # Recuperar nombres de features transformados
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    categorical_features = ohe.get_feature_names_out(categoricas)
    all_features = np.r_[categorical_features, numericas]

    # Importancias desde el modelo entrenado
    importances = pipeline.named_steps["model"].feature_importances_

    # Ordenar de mayor a menor
    sorted_idx = np.argsort(importances)[::-1]

    if top_n is not None:
        sorted_idx = sorted_idx[:top_n]

    # --- Plot ---
    plt.figure(figsize=figsize)
    plt.bar(range(len(sorted_idx)), importances[sorted_idx], align="center")
    plt.xticks(range(len(sorted_idx)), all_features[sorted_idx], rotation=90)
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.show()


import shap

def plot_shap_importance(pipeline, categoricas, numericas, X_sample=None, max_display=15):
    """
    Plot SHAP feature importances (summary plot) for a trained pipeline.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        The trained pipeline with preprocessing + XGBRegressor.
    categoricas : list of str
        List of categorical feature names (before encoding).
    numericas : list of str
        List of numerical feature names.
    X_sample : pd.DataFrame or None
        Optional sample of input data to compute SHAP values.
        If None, will use the full training set (not recommended for large data).
    max_display : int, default=15
        Maximum number of features to display in the SHAP summary plot.
    """
    # 1. Get transformed feature names
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    categorical_features = ohe.get_feature_names_out(categoricas)
    all_features = np.r_[categorical_features, numericas]

    # 2. Get model from pipeline
    model = pipeline.named_steps["model"]

    # 3. Choose a dataset to explain
    if X_sample is None:
        raise ValueError("⚠️ You must pass a sample DataFrame to X_sample (avoid using full training data).")

    # Apply preprocessing only (without model)
    X_transformed = pipeline.named_steps["preprocess"].transform(X_sample)

    # 4. SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)

    # 5. Summary plot
    plt.title("SHAP Feature Importances")
    shap.summary_plot(shap_values, X_transformed, feature_names=all_features, max_display=max_display)



def print_cv_results(search, metric_name="RMSE"):
    """
    Pretty print the results of a GridSearchCV or RandomizedSearchCV object.

    Parameters
    ----------
    search : fitted search object
        The fitted GridSearchCV or RandomizedSearchCV object.
    metric_name : str, default="RMSE"
        The name of the metric to display (e.g. "RMSE", "MAE").
    """
    print("=" * 50)
    print(" Best Cross-Validation Results ")
    print("=" * 50)
    print(f"Best {metric_name}: {-search.best_score_:.4f}")
    print("\nBest Parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    print("=" * 50)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_cv_results(search, param_name, metric_name="RMSE", top_n=20):
    """
    Plot distribution of cross-validation scores as a function of one hyperparameter.
    
    Parameters
    ----------
    search : fitted GridSearchCV or RandomizedSearchCV object
    param_name : str
        The hyperparameter to analyze (e.g., "model__max_depth").
    metric_name : str, default="RMSE"
        Name of the metric (for labeling).
    top_n : int, default=20
        Number of top models to display (sorted by best score).
    """
    # Extract results
    results = pd.DataFrame(search.cv_results_)
    results["rmse"] = -results["mean_test_score"]  # flip sign

    # Sort by performance
    results = results.sort_values("rmse").head(top_n)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=results,
        x=param_name,
        y="rmse",
        size="rmse",  # errors as bubble size
        hue="rmse",
        palette="viridis",
        legend=False
    )
    plt.title(f"CV {metric_name} vs {param_name}")
    plt.ylabel(metric_name)
    plt.xlabel(param_name)
    plt.show()

def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    """
    Evaluate a regression pipeline with R² and predictions.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        Fitted pipeline with a regressor at the end.
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : pd.Series or np.ndarray
        Test target.

    Returns
    -------
    dict
        Dictionary with predictions and R² scores.
    """
    # Predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Scores
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    #  Métricas
    print_regression_metrics(y_train, y_pred_train, "Train")
    print_regression_metrics(y_test, y_pred_test, "Test")
    
    return {
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "train_score": train_score,
        "test_score": test_score
    }


from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

def plot_precision_recall_vs_threshold(pipelines, splits, semester, threshold=None):
    """
    Plotea Precision y Recall en función del threshold y calcula el área bajo la curva PR
    para un semestre específico.

    Parameters
    ----------
    pipelines : dict
        Diccionario {semestre: pipeline entrenado}.
    splits : dict
        Diccionario {semestre: {X_train, X_test, y_train, y_test}}.
    semester : int
        Semestre a evaluar.
    threshold : float, optional
        Si se pasa, se marca en el gráfico el threshold con una línea vertical.

    Returns
    -------
    float
        Área bajo la curva Precision-Recall (PR AUC).
    """
    # --- 1. Seleccionar pipeline y datos del semestre ---
    pipeline = pipelines[semester]
    X_test = splits[semester]["X_test"]
    y_test = splits[semester]["y_test"]

    # --- 2. Probabilidades para la clase positiva ---
    probs_y = pipeline.predict_proba(X_test)[:, 1]

    # --- 3. Curva precision-recall ---
    precision, recall, thresholds = precision_recall_curve(y_test, probs_y)

    # --- 4. Calcular PR AUC ---
    pr_auc = auc(recall, precision)
    print(f"PR AUC (Sem {semester}) = {pr_auc:.4f}")

    # --- 5. Plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "r--", label="Recall")

    if threshold is not None:
        plt.axvline(x=threshold, color="green", linestyle="--", label=f"Threshold = {threshold}")

    plt.title(f"Precision-Recall vs Threshold (Sem {semester})")
    plt.xlabel("Threshold")
    plt.ylabel("Precision / Recall")
    plt.legend(loc="lower left")
    plt.ylim([0, 1])
    plt.grid(alpha=0.3)
    plt.show()

    return pr_auc

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,  roc_curve, roc_auc_score, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_semester(pipelines, splits, semester, threshold=0.5, n_preview=10):
    """
    Evaluate a trained pipeline for a given semester with custom threshold.
    
    Parameters
    ----------
    pipelines : dict
        Dictionary of pipelines trained by semester.
    splits : dict
        Dictionary of train/test splits by semester.
    semester : int
        Semester number to evaluate.
    threshold : float, default=0.5
        Probability cutoff for classification.
    n_preview : int, default=10
        Number of rows to preview in results DataFrame.
    
    Returns
    -------
    dict
        Dictionary with evaluation metrics and preview DataFrame.
    """
    # --- 1. Access data ---
    pipe = pipelines[semester]
    X_test = splits[semester]["X_test"]
    y_test = splits[semester]["y_test"]

    # --- 2. Predictions ---
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # --- 3. ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # --- 4. PR curve ---
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    # --- 5. Plot ROC & PR ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve (Semester {semester})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.title(f"Precision-Recall Curve (Semester {semester})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

    # --- 6. Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix (Semester {semester}, threshold={threshold})")
    plt.show()

    # --- 7. Metrics ---
    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

    print(f"\n--- Metrics for semester {semester} (threshold={threshold}) ---")
    for k, v in metrics.items():
        print(f"{k.capitalize():<10}: {v:.4f}")

    # --- 8. Results preview ---
    results = pd.DataFrame({
        "y_true": y_test.values,
        "y_proba": y_proba,
        "y_pred": y_pred
    })
    print("\nSnippet of observed vs predicted:")
    print(results.head(n_preview))

    return {"metrics": metrics, "results": results}

################### VISUALIZE TREE ###################################

import numpy as np
from xgboost import to_graphviz
from IPython.display import display

def export_xgb_tree(
    pipeline,
    categoricas,
    numericas,
    tree_index=0,
    save_path="xgb_tree.gv",
    fmt="pdf"
):
    """
    Exporta un árbol individual de un modelo XGBoost entrenado dentro de un pipeline.

    Parámetros
    ----------
    pipeline : sklearn.Pipeline
        Pipeline que contiene el modelo XGBoost entrenado en el step "model".
    categoricas : list[str]
        Lista de variables categóricas (usadas en OneHotEncoder).
    numericas : list[str]
        Lista de variables numéricas.
    tree_index : int, opcional
        Índice del árbol a visualizar (por defecto 0).
    save_path : str, opcional
        Ruta base donde guardar el archivo (sin extensión).
    fmt : str, opcional
        Formato de salida ("pdf", "png", "svg", etc.).
    display_inline : bool, opcional
        Si True, muestra el árbol inline en Jupyter Notebook.
    """
    # 1. Extraer booster del pipeline
    model = pipeline.named_steps["model"].get_booster()

    # 2. Recuperar nombres de features
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    categorical_features = ohe.get_feature_names_out(categoricas)
    all_features = np.r_[categorical_features, numericas].tolist()

    # 3. Asociar nombres al booster
    model.feature_names = all_features

    # 4. Exportar árbol
    dot = to_graphviz(model, tree_idx=tree_index)

    # 5. Guardar a archivo
    dot.render("../output/tree/"+save_path, format=fmt)



################### PREDICT NEW DATA

def predict_new_student(pipelines, new_data):
    """
    Realiza la predicción para un nuevo estudiante usando el pipeline 
    correspondiente a su semestre.
    
    Parameters
    ----------
    pipelines : dict
        Diccionario de pipelines entrenados por semestre.
    new_data : pd.DataFrame
        DataFrame con una o más observaciones nuevas, que debe incluir 
        la columna 'semestre'.
    
    Returns
    -------
    pd.DataFrame con columnas:
        - y_pred : predicción binaria
        - y_proba : probabilidad de clase positiva
    """
    # --- 1. Asegurar que tenemos columna semestre ---
    if "semestre" not in new_data.columns:
        raise ValueError("El nuevo dato debe incluir la columna 'semestre'.")

    results = []

    for i, row in new_data.iterrows():
        sem = row["semestre"]

        if sem not in pipelines:
            raise ValueError(f"No existe pipeline entrenado para semestre {sem}.")

        pipe = pipelines[sem]

        # Convertimos la fila a DataFrame de 1 fila
        X_new = row.to_frame().T  

        # Predicción
        y_pred = pipe.predict(X_new)[0]
        y_proba = pipe.predict_proba(X_new)[:, 1][0]

        results.append({
            **row.to_dict(),
            "y_pred": y_pred,
            "y_proba": y_proba
        })

    return pd.DataFrame(results)


########### OPTIMIZAR THRESHOLDS MUTICLASE:
import numpy as np
from sklearn.metrics import f1_score

def predict_with_thresholds(probas, thresholds, class_order):
    """
    Convierte una matriz de probabilidades en predicciones aplicando umbrales por clase.

    Args:
        probas (np.ndarray): shape (n_samples, n_classes). Columnas en el mismo orden que class_order.
        thresholds (dict): {label: threshold} umbral para cada etiqueta en class_order.
                           Si falta una etiqueta usa 0.5 por defecto.
        class_order (list): lista de etiquetas (strings o ints) que ordenan las columnas de probas.

    Returns:
        np.ndarray (int): predicciones como índices (0..k-1) en el mismo ordering que class_order.
    """
    n, k = probas.shape
    preds = np.empty(n, dtype=int)
    for i in range(n):
        row = probas[i]
        passed = [j for j, lab in enumerate(class_order) if row[j] >= thresholds.get(lab, 0.5)]
        if len(passed) == 1:
            preds[i] = passed[0]
        elif len(passed) > 1:
            preds[i] = int(max(passed, key=lambda j: row[j]))  # elegir la con mayor prob entre las que pasaron
        else:
            preds[i] = int(np.argmax(row))  # fallback: argmax
    return preds



from sklearn.metrics import f1_score
import numpy as np

def optimize_thresholds_coordinate_search(probas, y_true, class_names, objective_function, focus_class="Aplazado", n_steps=50):
    """
    Busca los thresholds óptimos para maximizar una métrica compuesta:
        0.7 * f1_score(clase foco) + 0.3 * f1_macro
    
    Parameters
    ----------
    probas : np.ndarray
        Probabilidades predichas (n_samples, n_classes).
    y_true : np.ndarray
        Etiquetas verdaderas (enteros).
    class_names : list of str
        Nombres de clases en el mismo orden que columnas de probas.
    focus_class : str, default="Aplazado"
        Clase de interés principal.
    n_steps : int
        Resolución de búsqueda en thresholds (más alto = más fino, pero más lento).
    
    Returns
    -------
    best_thr : dict
        Threshold óptimo por clase.
    best_score : float
        Valor de la métrica compuesta alcanzada.
    """
    n_classes = probas.shape[1]
    thresholds = np.linspace(0.1, 0.9, n_steps)
    best_score = -np.inf
    best_thr = {c: 0.5 for c in class_names}

    idx_focus = class_names.index(focus_class)

    for thr_focus in thresholds:
        # Copia thresholds por clase (por simplicidad, solo optimizamos la clase foco aquí)
        thr = np.full(n_classes, 0.5)
        thr[idx_focus] = thr_focus

        # Predicciones con estos thresholds
        y_pred = []
        for row in probas:
            # Asignar la primera clase que pase su umbral, si ninguna -> argmax
            chosen = None
            for i, p in enumerate(row):
                if p >= thr[i]:
                    chosen = i
                    break
            if chosen is None:
                chosen = np.argmax(row)
            y_pred.append(chosen)
        y_pred = np.array(y_pred)

        # Calcular métricas
        score = objective_function(y_true, y_pred, class_names, target_label=focus_class)

        if score > best_score:
            best_score = score
            best_thr = {c: thr[i] for i, c in enumerate(class_names)}

    return best_thr, best_score

