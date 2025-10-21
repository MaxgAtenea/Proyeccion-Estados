## 🧾 Estilo de commits del proyecto

Esta tabla define los **formatos estándar de commits** para mantener trazabilidad y claridad en el repositorio.  
Cada commit debe reflejar el propósito del cambio, siguiendo la estructura:

tipo: resumen breve

Ejemplo:  


---

| Tipo de commit | Cuándo usarlo | Ejemplo de mensaje corto | Ejemplo de mensaje extendido | Impacto esperado |
|-----------------|----------------|---------------------------|-------------------------------|------------------|
| **data** | Cuando actualizas, agregas o limpias datos (CSV, Excel, bases maestras) | `data: se actualizan datasets desde OneDrive (cohortes 2020–2025)` | `data: se reemplaza BaseMaestra con versión actualizada desde OneDrive (v2025-10-15)`<br><br>- Se incorporan nuevas cohortes 2024 y 2025.<br>- Se eliminan duplicados en IDs.<br>- Sin cambios en estructura de columnas. | Garantiza trazabilidad y versiones claras de datos base. |
| **eda** | Cuando realizas análisis exploratorios o generas reportes de calidad | `eda: análisis de outliers y correlaciones en variables socioeconómicas` | `eda: se actualizan reportes de exploración para cohorte 2024-2`<br><br>- Se generan gráficos de distribución.<br>- Se documentan valores atípicos y ausentes. | Mejora comprensión y control de calidad de datos. |
| **feature** | Cuando creas o modificas variables predictoras | `feature: se agregan variables de desempeño académico acumulado` | `feature: nuevas variables de retención basadas en créditos acumulados`<br><br>- Se agregan 5 nuevas features.<br>- Se documentan en Feature_List_Final.csv.<br>- Se actualiza FeaturePipeline.ipynb. | Mejora la capacidad explicativa del modelo. |
| **model** | Cuando entrenas, ajustas o cambias modelos predictivos | `model: se entrena XGBoost y se comparan métricas con baseline` | `model: se entrena modelo optimizado con nueva base y features`<br><br>- Se comparan Logistic, RandomForest y XGBoost.<br>- XGBoost mejora recall en 8%.<br>- Próximo paso: calibrar umbral. | Avance técnico en desempeño del modelo. |
| **eval** | Cuando validas, calibras o generas métricas y gráficos | `eval: se genera ROC y matriz de confusión del modelo final` | `eval: calibración del modelo con curva precision-recall`<br><br>- Se ajusta umbral a 0.68 (F1 óptimo).<br>- Se genera Reliability_Diagram.png.<br>- Impacto: reducción de falsos positivos en 8%. | Mejora precisión y confiabilidad del modelo. |
| **docs** | Cuando actualizas documentación, README o reportes | `docs: se actualiza README de 03_Modeling con flujo final` | `docs: se sincronizan presentaciones y bitácoras desde OneDrive`<br><br>- Se actualiza documentación de decisiones.<br>- Se agrega sección de resultados finales. | Facilita comunicación y trazabilidad. |
| **note** | Cuando documentas decisiones, aprendizajes o supuestos | `note: registro de supuestos sobre tasas de abandono` | `note: se documentan supuestos metodológicos de Feature_Engineering`<br><br>- Supuestos sobre imputación de ingreso familiar.<br>- Notas sobre cohortes incluidas/excluidas. | Transparencia analítica y transferencia de conocimiento. |
| **refactor** | Cuando reorganizas código sin cambiar resultados | `refactor: se divide Feature_Engineering en dos notebooks` | `refactor: limpieza y modularización de notebooks`<br><br>- Se reestructura pipeline para claridad.<br>- Sin cambios en resultados.<br>- Facilita mantenibilidad. | Estandarización del código y mejora técnica. |
| **fix** | Cuando corriges errores o inconsistencias menores | `fix: se corrige variable mal nombrada en FeaturePipeline` | `fix: corrección de bug en pipeline de entrenamiento`<br><br>- Se corrige tipo de dato en variable 'promedio_general'.<br>- Se actualizan métricas de validación. | Mantiene integridad del código. |
| **sync** | Cuando sincronizas o reemplazas archivos desde fuentes externas (como OneDrive) | `sync: actualización general desde OneDrive (sin cambios analíticos)` | `sync: actualización de datasets, métricas y presentaciones desde OneDrive`<br><br>- Se sincronizan archivos de datos y reportes.<br>- Sin cambios en código o resultados.<br>- Actualización de respaldo documental. | Sincroniza y actualiza contenido del repositorio. |

---

### 💡 Buenas prácticas rápidas

- **1 commit = 1 propósito.** No mezcles varios temas en un solo commit.  
- **Usa mensajes claros.** Evita “update files” o “minor fix”.  
- **Versiona outputs críticos.** Ejemplo: `Predictions_TrainTest_2025-10-15.csv`.  
- **Haz commits frecuentes y atómicos.** Mejor 5 commits pequeños que uno gigante.  

---
