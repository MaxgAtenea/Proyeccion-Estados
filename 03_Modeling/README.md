# README.md — Directorio 03_Modeling/
---

## Propósito
Este directorio centraliza **todo el desarrollo analítico para la proyección de estados**, desde el baseline hasta el modelo optimizado, asegurando trazabilidad, claridad y reproducibilidad.  
El objetivo es permitir que cualquier analista pueda **replicar y entender el proceso completo** sin depender de memoria oral o archivos dispersos.

---

## Estructura general

---

## Flujo de trabajo analítico

1. **01_Baseline/**  
   - Punto de partida para medir mejoras futuras.  
   - Contiene el modelo más simple (logístico, regresión, etc.) con métricas básicas.  

2. **02_Feature_Engineering/**  
   - Construcción y documentación de variables predictoras.  
   - Incluye definiciones, transformaciones y justificaciones.  

3. **03_Model_Selection/**  
   - Experimentos de modelos candidatos (Random Forest, XGBoost, etc.).  
   - Contiene resultados de *cross-validation*, métricas comparativas y parámetros.  

4. **04_Best_Model/**  
   - Implementación limpia del modelo final seleccionado.  
   - Guarda el pipeline consolidado y artefactos gráficos del entrenamiento.  

5. **05_Evaluation/**  
   - Validación interna del modelo (test set).  
   - Incluye métricas de desempeño, ROC, matriz de confusión y análisis de overfitting.  

6. **06_Fine_Tuning/**  
   - Ajustes finos del modelo: calibración, umbrales de decisión, interpretabilidad.  
   - Incluye gráficos SHAP y reliabilidad de probabilidades.  

7. **07_Outputs/**  
   - Artefactos finales listos para análisis o proyección.  
   - Sirve como punto de entrada para `05_Proyeccion/`.  

8. **Notes/**  
   - Registro de decisiones, observaciones y versiones del modelo.  
   - Facilita trazabilidad y comunicación dentro del equipo.  
   - *Output esperado:* `Version_Log.md`, `Decisiones_Modelo.csv`.

---

## Matching_Task/
Emparejamiento y panel.  

- **01_Imputacion/** → imputación de variables faltantes antes del emparejamiento.  
- **02_Emparejamiento/** → cálculo de propensity scores y balance.  
- **03_Panel/** → transformación de datos a formato panel y análisis longitudinal.  
- **Notes/** → decisiones metodológicas y observaciones.

---

## Convenciones de trabajo

| Tipo de archivo | Convención | Ejemplo |
|------------------|-------------|----------|
| Notebook | Descriptivo, con fecha o versión | `Model_Experiments_v02.ipynb` |
| Dataset intermedio | Fecha o versión explícita | `base_emparejamientos_{BDM_fecha Base Maestra (YYYmmdd)}_{fecha_actual (YYYmmdd)}.csv` |
| Imágenes o métricas | Nombres claros y métricas incluidas | `ROC_TestSet_AUC_{fecha_actual}.png` |
| Pesos modelos | `.pkl` con prefijo “Model_{caracteristica}” | `Model_optimizado_umbralesprob_aplazados.pkl` |

---

## Buenas prácticas

1. **Un notebook = un propósito.**  
   Cada notebook debe responder una pregunta específica o una etapa del pipeline.

2. **Documentar antes de cerrar.**  
   Al finalizar una sesión, registrar decisiones clave en `Notes/Version_Log.md`.

3. **Guardar outputs intermedios.**  
   No sobrescribir; versionar usando fecha o sufijo (`_v01`, `_v02`).

4. **Separar ejecución de interpretación.**  
   El código vive en notebooks, pero la interpretación va en `Notes/`. Esto es para facilitar encontrar las interpretaciones.

5. **Consistencia visual.**  
   Usar siempre la misma estructura de encabezado en los notebooks:
   - Propósito
   - Inputs
   - Outputs
   - Principales hallazgos
   - Próximos pasos

---

## Trazabilidad recomendada

| Etapa | Input | Output | Conecta con |
|--------|--------|---------|-------------|
| Baseline | `BaseMaestra` | `Metrics_Baseline.csv` | Model_Selection |
| Feature Engineering | `BaseMaestra` | `Feature_List_Final.csv` | Model_Selection |
| Model Selection | `Features Finales` | `Crossvalidation_Scores.csv` | Best_Model |
| Best Model | `Dataset Final` | `Pipeline.ipynb` | Evaluation |
| Evaluation | `Pipeline` | `Classification_Report.csv` | Fine_Tuning |
| Fine Tuning | `Pipeline` | `Optimized_Model.pkl` | Outputs |
| Outputs | `Optimized_Model.pkl` | `Predicciones y métricas finales` | Proyección |

---

## 🧾 Versión del documento
- **Versión:** 1.0  
- **Última actualización:** 2025-10-13   

