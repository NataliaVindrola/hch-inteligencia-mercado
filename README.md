# Inteligencia de Mercado — Harina de Carne y Hueso (HCH)

Predicción de precios, comparación de insumos sustitutos y monitoreo de oferta/demanda para el mercado colombiano de Harina de Carne y Hueso (HCH), un insumo proteico clave en la formulación de alimento balanceado para animales.

> 🔗 **Demo en vivo:** [hch-inteligencia-mercado.streamlit.app](https://hch-inteligencia-mercado.streamlit.app)

## Pregunta de negocio

¿Cómo puede una planta de alimentos balanceados anticipar y optimizar la compra de HCH —monitoreando la tendencia de precios de los insumos clave (TRM, maíz y soya Chicago), comparando alternativas de sustitución nutricional entre insumos proteicos, y prediciendo el precio futuro de la HCH— para reducir sobrecostos de compra y riesgo de desabastecimiento?

## Capturas

| Predicción | Comparación de insumos |
|---|---|
| <img width="1365" height="320" alt="newplot (1)" src="https://github.com/user-attachments/assets/f32da0ca-c99c-47ba-9c3f-de6569551a34" /> | <img width="1365" height="360" alt="newplot (2)" src="https://github.com/user-attachments/assets/21c24796-117e-4dc0-ac17-abbde7c25c82" />
|

## Metodología — ASUM-DM

| Fase | Contenido |
|---|---|
| 1. Entendimiento y calidad de datos | 9 fuentes perfiladas (BMC, Comtrade, TRM, CME, RONI, DANE-ESAG), reglas de negocio, matriz de calidad |
| 2. Modelo de datos | Esquema estrella (5 dimensiones + 2 tablas de hechos), ETL Extract-Transform-Load |
| 3. Preparación de datos | Vista `variables_modelo_mensual` — pivote mensual listo para modelar |
| 4. Modelado (Analítica 2.0) | XGBoost + Optuna, forecast directo multi-horizonte con intervalo de confianza, interpretabilidad con SHAP |
| 5. Dashboard (Analítica 1.0 + 2.0) | Streamlit — tendencia, comparación ponderada de insumos, oferta/demanda, predicción |

## Modelo de datos

Esquema estrella con 5 dimensiones (`DimFecha`, `DimInsumo`, `DimFuente`, `DimPais`, `DimIndicador`) y 2 tablas de hechos (`HechosPrecioInsumo`, `HechosIndicadorMacro`), diseñado siguiendo la metodología de Kimball (llaves subrogadas, dimensión fecha conforme, medidas aditivas/no aditivas documentadas).

## Hallazgos clave

- El precio de HCH se explica principalmente por su propia inercia y por el precio de sus sustitutos directos en el mercado local (especialmente sebo de res) — el TRM y los commodities de Chicago influyen, pero de forma secundaria.
- En septiembre de 2023 el precio de HCH cayó -16.9% en un solo mes sin que TRM ni commodities de Chicago se movieran de forma comparable — un choque doméstico específico del mercado de HCH.
- Colombia depende crecientemente de Brasil para las importaciones de HCH (51% → 78.5% entre 2020-2025), concentrando el riesgo de abastecimiento.
- Modelo de predicción (XGBoost + Optuna): **MAPE 4.4%** en validación cruzada walk-forward.

## Stack técnico

- **Datos:** Python (pandas, sqlite3), SQLite
- **Modelado:** XGBoost, Optuna (tuning bayesiano), SHAP (interpretabilidad), scikit-learn
- **Dashboard:** Streamlit, Plotly
- **Automatización:** scripts de actualización diaria vía `jupyter nbconvert` + git

## Estructura del repositorio

```
├── app.py                          # Dashboard Streamlit
├── actualizar_diario.py            # Orquestador de actualización diaria (notebooks + git push)
├── actualizacion_diaria.ipynb      # Extracción: TRM, CME, BMC, RONI, sacrificio bovino/porcino
├── etl_modelo_dimensional.ipynb    # ETL: esquema estrella + vista mensual
├── modelado_xgboost_hch.ipynb      # Pipeline de modelado (feature engineering → forecast)
├── Entendimiento_y_Calidad_Datos_HCH.ipynb   # Fase 1 — perfilamiento y calidad
├── HCH_features.ipynb              # Fuentes complementarias (sacrificio porcino, etc.)
├── BMC_Scraper_Final.ipynb         # Scraper de boletines BMC
└── mercado_hch.db                  # Base de datos SQLite
```

## Cómo correrlo localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/NataliaVindrola/hch-inteligencia-mercado.git
cd hch-inteligencia-mercado

# 2. Instalar dependencias
pip install streamlit plotly xgboost shap scikit-learn pandas numpy optuna statsmodels

# 3. Correr el dashboard
streamlit run app.py
```

### Actualizar los datos

```bash
python actualizar_diario.py
```
> **Nota:** el sacrificio bovino (DANE ESAG) no se actualiza automáticamente — el DANE publica esta serie trimestralmente. Revisar cada ~3 meses en [dane.gov.co — ESAG históricos](https://www.dane.gov.co/index.php/estadisticas-por-tema/agropecuario/encuesta-de-sacrificio-de-ganado/encuesta-de-sacrificio-de-ganado-esag-historicos), descargar el archivo `series-hist-ESAG-*.xls` más reciente y actualizar `RUTA_ESAG` en `actualizacion_diaria.ipynb`.
> 
Corre en orden `actualizacion_diaria.ipynb` → `etl_modelo_dimensional.ipynb`, y sube los cambios a GitHub automáticamente. Pensado para automatizarse con Task Scheduler (Windows) o cron (Linux/Mac).

## Autora

**Natalia Vindrola** — [github.com/NataliaVindrola](https://github.com/NataliaVindrola)

Proyecto de portafolio, con formación en Zootecnia aplicada a Ciencia de Datos.
