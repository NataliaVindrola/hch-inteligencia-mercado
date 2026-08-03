"""
Dashboard HCH — Inteligencia de Mercado
Sección: Predicción (análisis 3.a / 3.b)

Corre con:  streamlit run app.py
Requiere:   mercado_hch.db en la misma carpeta (o ajustar DB_PATH abajo)
"""

import sqlite3
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit

# ============================================================
# CONFIGURACIÓN
# ============================================================
DB_PATH = "mercado_hch.db"
TARGET = "precio_hch_cop_kg"
N_FORECAST = 18

# Hiperparámetros ya encontrados por Optuna en el notebook de modelado
# (Fase 4 — no se vuelven a tunear aquí, eso sería lento para un dashboard en vivo)
PARAMS_XGB = dict(
    n_estimators=281, learning_rate=0.10972207973716684, max_depth=3,
    subsample=0.8312720803599148, colsample_bytree=0.5248127018659656,
    min_child_weight=4, reg_lambda=18.51310142624015, reg_alpha=1.257168269795475,
    random_state=42, verbosity=0,
)

# Features ya seleccionadas en el notebook de modelado (ranking SHAP real)
FEATURES = [
    "lag_1", "sustituto_sebo_res_cop_kg", "sustituto_harina_visceras_cop_kg",
    "ratio_hch_harina_hueso", "ratio_hch_torta_soya_importada", "mom3",
    "ratio_hch_soya", "ratio_hch_harina_pluma_sangre", "ratio_hch_subproductos_carnicos",
    "demanda_concentrado_avicola", "roni_lag3", "demanda_concentrado_porcino",
    "sustituto_harina_sangre_cop_kg", "trm_diff", "mom1", "sacrificio_bovino",
    "mes_sin", "demanda_concentrado_gatos", "trim",
]

SUSTITUTOS = ["torta_soya_importada", "harina_visceras", "harina_pluma_sangre",
              "harina_sangre", "harina_hueso", "sebo_res", "subproductos_carnicos"]

# Traducción de variables técnicas a frases de negocio (ver mockup)
DRIVER_FRASES = {
    "lag_1": "La tendencia reciente del precio sigue siendo el factor más fuerte",
    "sustituto_sebo_res_cop_kg": "El precio del sebo de res (sustituto directo)",
    "sustituto_harina_visceras_cop_kg": "El precio de la harina de vísceras",
    "sustituto_harina_sangre_cop_kg": "El precio de la harina de sangre",
    "ratio_hch_harina_hueso": "Qué tan competitivo está HCH frente a harina de hueso",
    "ratio_hch_torta_soya_importada": "Qué tan competitivo está HCH frente a la torta de soya",
    "ratio_hch_soya": "Qué tan competitivo está HCH frente a la soya de Chicago",
    "ratio_hch_harina_pluma_sangre": "Qué tan competitivo está HCH frente a harina pluma-sangre",
    "ratio_hch_subproductos_carnicos": "Qué tan competitivo está HCH frente a subproductos cárnicos",
    "demanda_concentrado_avicola": "La demanda de concentrado avícola",
    "demanda_concentrado_porcino": "La demanda de concentrado porcino",
    "demanda_concentrado_gatos": "La demanda de concentrado para gatos",
    "roni_lag3": "Las condiciones climáticas (El Niño/La Niña, con 3 meses de rezago)",
    "trm_diff": "El movimiento reciente del dólar (TRM)",
    "mom1": "El impulso del precio en el último mes",
    "mom3": "El impulso del precio en los últimos 3 meses",
    "sacrificio_bovino": "La disponibilidad de materia prima (sacrificio bovino)",
    "mes_sin": "La estacionalidad del mes",
    "trim": "El trimestre del año",
}

st.set_page_config(page_title="HCH — Predicción", layout="wide")


# ============================================================
# CACHÉ QUE SE AUTO-INVALIDA CUANDO CAMBIA mercado_hch.db
# ============================================================
def _db_mtime():
    """Antiguedad del archivo de la base de datos — se usa como llave de
    caché: si el ETL diario actualiza mercado_hch.db, este valor cambia
    y Streamlit vuelve a cargar/entrenar automáticamente, sin reiniciar."""
    return os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0


@st.cache_data(show_spinner="Cargando datos de mercado_hch.db...")
def cargar_datos(_mtime):
    con = sqlite3.connect(DB_PATH)

    df = pd.read_sql("SELECT * FROM variables_modelo_mensual ORDER BY anio, mes", con)
    df["fecha"] = pd.to_datetime(df["anio_mes"] + "-01")

    placeholders = ",".join("?" * len(SUSTITUTOS))
    df_sust = pd.read_sql(f"""
        SELECT f.anio, f.mes, i.codigo_insumo AS producto, AVG(h.precio_cop_ton) AS precio_cop_ton
        FROM HechosPrecioInsumo h
        JOIN DimFecha f ON h.id_fecha = f.id_fecha
        JOIN DimInsumo i ON h.id_insumo = i.id_insumo
        JOIN DimFuente fu ON h.id_fuente = fu.id_fuente
        WHERE i.codigo_insumo IN ({placeholders}) AND fu.nombre_fuente = 'BMC'
        GROUP BY f.anio, f.mes, i.codigo_insumo
    """, con, params=SUSTITUTOS)
    con.close()

    df_sust_pivot = df_sust.pivot(index=["anio", "mes"], columns="producto", values="precio_cop_ton").reset_index()
    df_sust_pivot.columns.name = None
    df = df.merge(df_sust_pivot, on=["anio", "mes"], how="left")

    return df


def feature_engineer(df_util):
    df = df_util.copy()
    df[TARGET] = df["precio_hch_cop_ton"] / 1000

    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["trim"] = df["fecha"].dt.quarter

    df["lag_1"] = df[TARGET].shift(1)
    df["mom1"] = df[TARGET].shift(1) - df[TARGET].shift(2)
    df["mom3"] = df[TARGET].shift(1) - df[TARGET].shift(4)

    df["trm_lag1"] = df["trm"].shift(1)
    df["trm_diff"] = df["trm"] - df["trm"].shift(1)
    df["roni_lag3"] = df["roni"].shift(3)
    df["roni_lag6"] = df["roni"].shift(6)

    df["ratio_hch_soya"] = df["lag_1"] / (df["cme_soya"] / (df["trm"] / 1000)).replace(0, np.nan)
    df["ratio_hch_maiz"] = df["lag_1"] / (df["cme_maiz"] / (df["trm"] / 1000)).replace(0, np.nan)

    for s in SUSTITUTOS:
        if s in df.columns:
            df[f"sustituto_{s}_cop_kg"] = df[s] / 1000
            df[f"ratio_hch_{s}"] = df["lag_1"] / df[f"sustituto_{s}_cop_kg"].replace(0, np.nan)

    return df


@st.cache_resource(show_spinner="Entrenando modelo XGBoost...")
def entrenar_y_pronosticar(_mtime):
    df = cargar_datos(_mtime)
    df_util = df.dropna(subset=["precio_hch_usd_ton"]).reset_index(drop=True)

    # Imputación estacional para columnas con rezago de publicación (DANE)
    for col in df_util.columns:
        if df_util[col].isna().any() and df_util[col].dtype != "object":
            for idx in df_util[df_util[col].isna()].index:
                mes_i = df_util.loc[idx, "mes"]
                promedio_mes = df_util[(df_util["mes"] == mes_i) & (df_util[col].notna())][col].mean()
                if pd.notna(promedio_mes):
                    df_util.loc[idx, col] = promedio_mes

    df_feat = feature_engineer(df_util)
    df_clean = df_feat.dropna(subset=[TARGET] + FEATURES).reset_index(drop=True)

    X = df_clean[FEATURES].values
    y = df_clean[TARGET].values

    # Métricas de validación (walk-forward, igual que en el notebook)
    tscv = TimeSeriesSplit(n_splits=5, test_size=3)
    maes, rmses, mapes = [], [], []
    for tr_idx, te_idx in tscv.split(X):
        m = xgb.XGBRegressor(**PARAMS_XGB)
        m.fit(X[tr_idx], y[tr_idx])
        preds = m.predict(X[te_idx])
        maes.append(np.mean(np.abs(y[te_idx] - preds)))
        rmses.append(np.sqrt(np.mean((y[te_idx] - preds) ** 2)))
        mapes.append(np.mean(np.abs((y[te_idx] - preds) / y[te_idx])) * 100)

    # Forecast directo multi-horizonte (igual lógica que el notebook de modelado)
    X_last = df_clean[FEATURES].iloc[[-1]].values
    fecha_base = df_clean["fecha"].iloc[-1]
    fechas_fut = pd.date_range(start=fecha_base + pd.DateOffset(months=1), periods=N_FORECAST, freq="MS")

    preds_punto, ic_low, ic_high = [], [], []
    ultimo_residuo = np.array([0.0])

    for h in range(1, N_FORECAST + 1):
        y_h = df_feat[TARGET].shift(-h)
        df_h = df_feat[FEATURES].copy()
        df_h["y_h"] = y_h
        df_h = df_h.dropna()
        if len(df_h) < 12:
            continue
        X_h, y_hv = df_h[FEATURES].values, df_h["y_h"].values

        n_splits_h = max(2, min(5, len(df_h) // 8))
        tscv_h = TimeSeriesSplit(n_splits=n_splits_h, test_size=1)
        residuos_h = []
        for tr_idx, te_idx in tscv_h.split(X_h):
            if len(tr_idx) < 8:
                continue
            m = xgb.XGBRegressor(**PARAMS_XGB)
            m.fit(X_h[tr_idx], y_hv[tr_idx])
            residuos_h.extend(y_hv[te_idx] - m.predict(X_h[te_idx]))
        residuos_h = np.array(residuos_h)
        if len(residuos_h) >= 3:
            ultimo_residuo = residuos_h
        else:
            residuos_h = ultimo_residuo

        modelo_h = xgb.XGBRegressor(**PARAMS_XGB)
        modelo_h.fit(X_h, y_hv)
        pred_h = modelo_h.predict(X_last)[0]

        residuos_centrados = residuos_h - residuos_h.mean()
        boot_h = pred_h + np.random.choice(residuos_centrados, size=1000, replace=True)
        p5, p95 = np.percentile(boot_h, [5, 95])
        p5, p95 = min(p5, pred_h), max(p95, pred_h)

        preds_punto.append(pred_h)
        ic_low.append(p5)
        ic_high.append(p95)

    df_forecast = pd.DataFrame({
        "fecha": fechas_fut[:len(preds_punto)], "pred": preds_punto,
        "ic_low": ic_low, "ic_high": ic_high,
    })

    # SHAP sobre modelo final (entrenado con el 100% de los datos)
    modelo_final = xgb.XGBRegressor(**PARAMS_XGB)
    modelo_final.fit(X, y)
    import shap
    explainer = shap.TreeExplainer(modelo_final)
    shap_values = explainer.shap_values(df_clean[FEATURES])
    shap_signed = pd.Series(shap_values[-1], index=FEATURES)  # último mes real, con signo

    metricas = {"mae": np.mean(maes), "rmse": np.mean(rmses), "mape": np.mean(mapes), "mape_std": np.std(mapes)}

    return df_clean, df_forecast, metricas, shap_signed


# ============================================================
# UI
# ============================================================
@st.cache_data(show_spinner="Cargando tendencia de insumos...")
def cargar_tendencia(_mtime):
    con = sqlite3.connect(DB_PATH)

    df_hch = pd.read_sql("""
        SELECT f.fecha, AVG(h.precio_cop_ton)/1000 AS precio_hch_cop_kg
        FROM HechosPrecioInsumo h
        JOIN DimFecha   f  ON h.id_fecha  = f.id_fecha
        JOIN DimInsumo  i  ON h.id_insumo = i.id_insumo
        JOIN DimFuente  fu ON h.id_fuente = fu.id_fuente
        WHERE i.codigo_insumo = 'HCH' AND fu.nombre_fuente = 'BMC'
        GROUP BY f.fecha
    """, con)

    df_ind = pd.read_sql("""
        SELECT f.fecha, di.codigo_indicador, AVG(h.valor) AS valor
        FROM HechosIndicadorMacro h
        JOIN DimFecha     f  ON h.id_fecha     = f.id_fecha
        JOIN DimIndicador di ON h.id_indicador = di.id_indicador
        WHERE di.codigo_indicador IN ('trm', 'cme_soya', 'cme_maiz')
        GROUP BY f.fecha, di.codigo_indicador
    """, con)
    con.close()

    df_ind_pivot = df_ind.pivot(index="fecha", columns="codigo_indicador", values="valor").reset_index()
    # outer: cada serie conserva su propia historia — HCH no se publica todos
    # los días calendario (solo días con rueda BMC), TRM/CME sí son diarios;
    # con "inner" se recortaban las fechas al mínimo común y se perdía rango.
    df = df_hch.merge(df_ind_pivot, on="fecha", how="outer")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


def pagina_prediccion():
    mtime = _db_mtime()
    df_clean, df_forecast, metricas, shap_signed = entrenar_y_pronosticar(mtime)

    st.title("Precio de HCH — próximos meses")
    st.caption(f"Actualizado con el boletín BMC más reciente · datos al {df_clean['anio_mes'].iloc[-1]}")

    MESES_ES = {1:"ene",2:"feb",3:"mar",4:"abr",5:"may",6:"jun",7:"jul",
                8:"ago",9:"sep",10:"oct",11:"nov",12:"dic"}
    def fmt_mes(fecha):
        return f"{MESES_ES[fecha.month]} {fecha.year}"

    precio_actual = df_clean[TARGET].iloc[-1]
    fecha_actual_legible = fmt_mes(df_clean["fecha"].iloc[-1])

    precio_3m = df_forecast["pred"].iloc[:3].mean()
    delta_3m_pct = (precio_3m - precio_actual) / precio_actual * 100
    rango_3m_ini = fmt_mes(df_forecast["fecha"].iloc[0])
    rango_3m_fin = fmt_mes(df_forecast["fecha"].iloc[2])

    if delta_3m_pct > 2:
        st.warning(f"📈 **Se espera un alza de ~{delta_3m_pct:.1f}% en los próximos 3 meses.** "
                   "Si tu formulación lo permite, considera adelantar compra.")
    elif delta_3m_pct < -2:
        st.info(f"📉 **Se espera una baja de ~{abs(delta_3m_pct):.1f}% en los próximos 3 meses.** "
                "Podría convenir posponer compras no urgentes.")
    else:
        st.success("➡️ **El precio se espera estable en los próximos 3 meses.** Sin señal fuerte de compra/espera.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Precio actual ({fecha_actual_legible})",
                f"{precio_actual*1000:,.0f} COP/kg" if precio_actual < 100 else f"{precio_actual:,.0f} COP/kg")
    col2.metric(f"Promedio próximos 3 meses ({rango_3m_ini}–{rango_3m_fin})",
                f"{precio_3m:,.0f} COP/kg", f"{delta_3m_pct:+.1f}%")
    col3.metric(f"Rango probable 90% ({rango_3m_ini}–{rango_3m_fin})",
                f"{df_forecast['ic_low'].iloc[:3].min():,.0f}–{df_forecast['ic_high'].iloc[:3].max():,.0f}")
    confianza = "Alta" if metricas["mape"] < 6 else ("Media" if metricas["mape"] < 10 else "Baja")
    col4.metric("Confianza del pronóstico", confianza, f"MAPE {metricas['mape']:.1f}%")
    st.caption("El rango probable es la envolvente de los intervalos de confianza individuales "
               "de cada uno de los 3 meses (no el precio de un solo mes puntual).")

    st.markdown("##### Evolución esperada (próximos 18 meses)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_forecast["fecha"], y=df_forecast["ic_high"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df_forecast["fecha"], y=df_forecast["ic_low"], fill="tonexty",
                              fillcolor="rgba(239,159,39,0.18)", line=dict(width=0), name="Rango probable"))
    fig.add_trace(go.Scatter(x=df_forecast["fecha"], y=df_forecast["pred"], line=dict(color="#BA7517", width=2),
                              mode="lines+markers", name="Precio esperado"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="COP/kg",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Qué explicó el precio en el último mes conocido")
    st.caption("Esto describe el mes más reciente con dato real — no es la explicación del forecast de arriba, que mira hacia adelante.")
    top_drivers = shap_signed.abs().sort_values(ascending=False).head(5)
    for i, (var, _) in enumerate(top_drivers.items()):
        signo = shap_signed[var]
        icono = "🔺" if signo > 0 else "🔻"
        frase = DRIVER_FRASES.get(var, var)
        sufijo = " → presiona el precio al alza" if signo > 0 else " → presiona el precio a la baja"
        principal = "  `Principal`" if i == 0 else ""
        st.write(f"{icono} {frase}{sufijo}{principal}")

    with st.expander("Detalle técnico del modelo"):
        st.write(f"MAE: {metricas['mae']:.1f} COP/kg · RMSE: {metricas['rmse']:.1f} COP/kg · "
                 f"MAPE: {metricas['mape']:.2f}% ± {metricas['mape_std']:.2f}%")
        st.write(f"Features usadas: {len(FEATURES)} · Meses de entrenamiento: {len(df_clean)}")
        df_forecast_mostrar = df_forecast.copy()
        df_forecast_mostrar[["pred", "ic_low", "ic_high"]] = df_forecast_mostrar[["pred", "ic_low", "ic_high"]].round(0)
        st.dataframe(df_forecast_mostrar)


def pagina_tendencia():
    df = cargar_tendencia(_db_mtime())

    st.title("Medidas macro (TRM, CME)")
    st.caption("Análisis 1.a / 1.b — evolución diaria de HCH junto a TRM y commodities Chicago")

    fecha_min, fecha_max = df["fecha"].min().date(), df["fecha"].max().date()
    rango = st.date_input("Rango de fechas", value=(fecha_min, fecha_max),
                           min_value=fecha_min, max_value=fecha_max, format="YYYY-MM-DD")

    if len(rango) != 2:
        st.info("Selecciona la fecha de inicio y la fecha de fin en el calendario.")
        return
    df_f = df[(df["fecha"].dt.date >= rango[0]) & (df["fecha"].dt.date <= rango[1])].reset_index(drop=True)

    if len(df_f) < 2:
        st.warning("Selecciona un rango con al menos 2 días con dato.")
        return

    series_disponibles = {
        "Precio HCH (COP/kg)": ("precio_hch_cop_kg", "COP/kg", "#D85A30"),
        "TRM (COP/USD)": ("trm", "COP/USD", "#378ADD"),
        "Soya Chicago (USD/ton)": ("cme_soya", "USD/ton", "#1D9E75"),
        "Maíz Chicago (USD/ton)": ("cme_maiz", "USD/ton", "#EF9F27"),
    }
    seleccionadas = st.multiselect("Variables a mostrar", options=list(series_disponibles.keys()),
                                    default=list(series_disponibles.keys()))
    if not seleccionadas:
        st.info("Selecciona al menos una variable.")
        return
    series = {k: series_disponibles[k] for k in seleccionadas}

    # Cada indicador puede tener su propio último día con dato (p.ej. HCH solo
    # publica en días de rueda BMC, TRM/CME son de lunes a viernes) — el
    # cálculo de "actual" y su fecha se hace por serie, no sobre la fila final
    # del dataframe combinado.
    cols = st.columns(len(series))
    for col, (label, (campo, unidad, _)) in zip(cols, series.items()):
        serie_valida = df_f[["fecha", campo]].dropna()
        if len(serie_valida) < 2:
            col.metric(label, "Sin datos en el rango")
            continue
        actual = serie_valida[campo].iloc[-1]
        fecha_actual = serie_valida["fecha"].iloc[-1]
        inicial = serie_valida[campo].iloc[0]
        delta = (actual - inicial) / inicial * 100
        col.metric(f"{label} — {fecha_actual.strftime('%d %b %Y')}", f"{actual:,.0f} {unidad}", f"{delta:+.1f}%")

    st.markdown(f"##### Evolución indexada diaria (base 100 = {rango[0].strftime('%d %b %Y')})")
    fig = go.Figure()
    for label, (campo, unidad, color) in series.items():
        serie_valida = df_f[["fecha", campo]].dropna()
        if serie_valida.empty:
            continue
        base = serie_valida[campo].iloc[0]
        indice = (serie_valida[campo] / base * 100).round(1)
        fig.add_trace(go.Scatter(
            x=serie_valida["fecha"], y=indice, name=label, line=dict(color=color, width=1.5),
            customdata=serie_valida[campo],
            hovertemplate=f"%{{x|%d %b %Y}}<br>{label}: %{{customdata:,.0f}} {unidad}  (índice %{{y}})<extra></extra>",
        ))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       yaxis_title="Índice", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    st.caption("Pasa el mouse sobre la línea para ver el valor real de cada día. "
               "El HCH solo tiene dato en días con rueda BMC; TRM y CME son días hábiles.")

    st.markdown("##### Tabla de valores reales (sin indexar)")
    df_tabla = df_f[["fecha"]].copy()
    for label, (campo, unidad, _) in series.items():
        df_tabla[f"{label}"] = df_f[campo].round(1)
        df_tabla[f"{label} — var. diaria %"] = (df_f[campo].pct_change() * 100).round(2)
    df_tabla = df_tabla.rename(columns={"fecha": "Fecha"})
    df_tabla["Fecha"] = df_tabla["Fecha"].dt.strftime("%Y-%m-%d")
    st.dataframe(df_tabla.sort_values("Fecha", ascending=False), width="stretch", hide_index=True)


@st.cache_data(show_spinner="Cargando precios de sustitutos...")
def cargar_sustitutos(_mtime):
    con = sqlite3.connect(DB_PATH)
    productos = SUSTITUTOS + ["HCH"]
    placeholders = ",".join("?" * len(productos))
    df = pd.read_sql(f"""
        SELECT f.fecha, i.codigo_insumo AS producto, AVG(h.precio_cop_ton)/1000 AS precio_cop_kg
        FROM HechosPrecioInsumo h
        JOIN DimFecha f ON h.id_fecha = f.id_fecha
        JOIN DimInsumo i ON h.id_insumo = i.id_insumo
        JOIN DimFuente fu ON h.id_fuente = fu.id_fuente
        WHERE i.codigo_insumo IN ({placeholders}) AND fu.nombre_fuente = 'BMC'
        GROUP BY f.fecha, i.codigo_insumo
    """, con, params=productos)
    con.close()
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


NOMBRES_SUSTITUTOS = {
    "HCH": "Harina de carne y hueso (HCH)",
    "torta_soya_importada": "Torta de soya importada",
    "harina_visceras": "Harina de vísceras",
    "harina_pluma_sangre": "Harina pluma-sangre",
    "harina_sangre": "Harina de sangre",
    "harina_hueso": "Harina de hueso",
    "sebo_res": "Sebo de res",
    "subproductos_carnicos": "Subproductos cárnicos",
}
COLORES_SUSTITUTOS = ["#D85A30", "#1D9E75", "#D4537E", "#378ADD", "#EF9F27", "#7F77DD", "#888780", "#B57A1D"]


def pagina_sustitutos():
    df = cargar_sustitutos(_db_mtime())

    st.title("Precios de sustitutos en el tiempo")
    st.caption("Apoya análisis 2.a / 2.b — precio diario BMC (por rueda), COP/kg")

    if df.empty:
        st.info("Aún no hay precios de sustitutos cargados en HechosPrecioInsumo.")
        return

    disponibles = [s for s in (["HCH"] + SUSTITUTOS) if s in df["producto"].unique()]
    default_sust = [s for s in SUSTITUTOS if s in disponibles][:2]
    default = (["HCH"] if "HCH" in disponibles else []) + default_sust
    elegidos = st.multiselect("Insumos a mostrar", options=disponibles,
                               format_func=lambda s: NOMBRES_SUSTITUTOS.get(s, s), default=default)

    if not elegidos:
        st.info("Selecciona al menos un insumo.")
        return

    fig = go.Figure()
    fechas_ultimas = []
    for i, s in enumerate(elegidos):
        d = df[df["producto"] == s].sort_values("fecha")
        if d.empty:
            continue
        es_hch = s == "HCH"
        nombre = NOMBRES_SUSTITUTOS.get(s, s)
        fig.add_trace(go.Scatter(
            x=d["fecha"], y=d["precio_cop_kg"], name=nombre,
            line=dict(color=COLORES_SUSTITUTOS[i % len(COLORES_SUSTITUTOS)], width=3 if es_hch else 1.5),
            hovertemplate=f"%{{x|%d %b %Y}}<br>{nombre}: %{{y:,.0f}} COP/kg<extra></extra>",
        ))
        fechas_ultimas.append(f"{nombre}: {d['fecha'].iloc[-1].strftime('%d %b %Y')}")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                       yaxis_title="COP/kg", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    st.caption("Último dato por insumo — " + " · ".join(fechas_ultimas))


@st.cache_data(show_spinner="Cargando oferta y demanda...")
def cargar_oferta_demanda(_mtime):
    con = sqlite3.connect(DB_PATH)
    codigos = ["sacrificio_bovino", "demanda_concentrado_perros", "demanda_concentrado_gatos",
               "demanda_concentrado_avicola", "demanda_concentrado_porcino"]
    placeholders = ",".join("?" * len(codigos))
    df = pd.read_sql(f"""
        SELECT f.anio, f.mes, di.codigo_indicador, AVG(h.valor) AS valor
        FROM HechosIndicadorMacro h
        JOIN DimFecha f ON h.id_fecha = f.id_fecha
        JOIN DimIndicador di ON h.id_indicador = di.id_indicador
        WHERE di.codigo_indicador IN ({placeholders})
        GROUP BY f.anio, f.mes, di.codigo_indicador
    """, con, params=codigos)
    con.close()

    df_pivot = df.pivot(index=["anio", "mes"], columns="codigo_indicador", values="valor").reset_index()
    df_pivot.columns.name = None
    df_pivot["fecha"] = pd.to_datetime(dict(year=df_pivot["anio"], month=df_pivot["mes"], day=1))
    df_pivot = df_pivot.sort_values("fecha").reset_index(drop=True)

    for col in ["demanda_concentrado_perros", "demanda_concentrado_gatos",
                "demanda_concentrado_avicola", "demanda_concentrado_porcino"]:
        if col not in df_pivot.columns:
            df_pivot[col] = np.nan

    # sum(min_count=1): da NaN solo si AMBAS columnas están vacías (evita
    # que "sin dato todavía" se confunda con "demanda = 0" y arrastre la
    # fecha base del índice hasta 2008)
    df_pivot["demanda_mascotas"] = df_pivot[["demanda_concentrado_perros", "demanda_concentrado_gatos"]].sum(axis=1, min_count=1)
    df_pivot["demanda_monogastricos"] = df_pivot[["demanda_concentrado_avicola", "demanda_concentrado_porcino"]].sum(axis=1, min_count=1)
    return df_pivot


def pagina_oferta_demanda():
    df = cargar_oferta_demanda(_db_mtime())

    st.title("Oferta y demanda")
    st.caption("Contexto de mercado para 2.b y 3.b — oferta = sacrificio bovino (DANE-ESAG, actualización "
               "trimestral/manual); demanda = consumo de concentrado terminado (BMC, mensual)")

    if df.empty:
        st.info("Aún no hay datos de oferta/demanda cargados.")
        return

    # Slider de rango por mes (los datos son mensuales, tiene más sentido
    # deslizar entre meses que elegir dos fechas exactas de un calendario).
    df["anio_mes"] = df["fecha"].dt.strftime("%Y-%m")
    meses_disponibles = sorted(df["anio_mes"].unique())
    mes_ini, mes_fin = st.select_slider(
        "Rango de meses", options=meses_disponibles,
        value=(meses_disponibles[0], meses_disponibles[-1]),
    )
    df = df[(df["anio_mes"] >= mes_ini) & (df["anio_mes"] <= mes_fin)].reset_index(drop=True)

    if df.empty:
        st.info("No hay datos en el rango seleccionado.")
        return

    series_def = {
        "Oferta (sacrificio bovino)": ("sacrificio_bovino", "cab./mes", "#378ADD"),
        "Demanda mascotas": ("demanda_mascotas", "ton/mes", "#D4537E"),
        "Demanda monogástricos": ("demanda_monogastricos", "ton/mes", "#EF9F27"),
    }

    # Cada serie se evalúa por separado: si el sacrificio bovino no tiene
    # dato reciente (o directamente no existe) dentro del rango elegido,
    # simplemente no se dibuja — no se recorta la demanda para "esperarlo".
    series = {label: campo_info for label, campo_info in series_def.items()
              if campo_info[0] in df.columns and df[campo_info[0]].notna().sum() >= 2}

    if not series:
        st.info("Se necesitan al menos 2 meses con dato en alguna de las series, dentro del rango elegido.")
        return

    faltantes = [label for label in series_def if label not in series]
    if faltantes:
        st.caption(f"⚠️ Sin datos suficientes en este rango: {', '.join(faltantes)} — se omiten del gráfico.")

    cols = st.columns(len(series))
    for col, (label, (campo, unidad, _)) in zip(cols, series.items()):
        serie_valida = df[["fecha", campo]].dropna()
        ultimo = serie_valida[campo].iloc[-1]
        fecha_ultimo = serie_valida["fecha"].iloc[-1]
        col.metric(f"{label} — {fecha_ultimo.strftime('%b %Y')}", f"{ultimo:,.0f} {unidad}")

    st.markdown(f"##### Evolución indexada (base 100 = primer mes disponible de cada serie, dentro de {mes_ini} a {mes_fin})")
    fig = go.Figure()
    for label, (campo, unidad, color) in series.items():
        serie_valida = df[["fecha", campo]].dropna().sort_values("fecha")
        base = serie_valida[campo].iloc[0]
        indice = (serie_valida[campo] / base * 100).round(1) if base else serie_valida[campo] * 0
        fig.add_trace(go.Scatter(
            x=serie_valida["fecha"], y=indice, name=label, line=dict(color=color, width=2),
            customdata=serie_valida[campo],
            hovertemplate=f"%{{x|%b %Y}}<br>{label}: %{{customdata:,.0f}} {unidad}  (índice %{{y}})<extra></extra>",
        ))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       yaxis_title="Índice", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    st.caption("Cada serie llega hasta su propio último dato disponible dentro del rango elegido. Si el "
               "sacrificio bovino (actualización manual/trimestral) queda rezagado frente a la demanda de "
               "concentrados (mensual vía BMC), su línea simplemente termina antes — las demás siguen hasta "
               "donde efectivamente hay dato.")

    if "sacrificio_bovino" in df.columns and df["sacrificio_bovino"].notna().sum() >= 2:
        with st.expander("Ver oferta con su historia completa (desde 2008, ignora el slider de arriba)"):
            df_completo = cargar_oferta_demanda(_db_mtime())
            df_of_completo = df_completo.dropna(subset=["sacrificio_bovino"]).sort_values("fecha")
            base_larga = df_of_completo["sacrificio_bovino"].iloc[0]
            indice_largo = (df_of_completo["sacrificio_bovino"] / base_larga * 100).round(1)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_of_completo["fecha"], y=indice_largo, line=dict(color="#378ADD", width=1.5),
                customdata=df_of_completo["sacrificio_bovino"],
                hovertemplate="%{x|%b %Y}<br>%{customdata:,.0f} cab./mes  (índice %{y})<extra></extra>",
            ))
            fig2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Índice",
                                title=dict(text=f"Oferta (sacrificio bovino) — base 100 = {df_of_completo['fecha'].iloc[0].strftime('%b %Y')}", font=dict(size=13)))
            st.plotly_chart(fig2, width="stretch")
            st.caption("Aquí sí se ve la tendencia estructural decreciente de largo plazo y la estacionalidad "
                       "(feb=mínimo, dic=máximo) documentadas en Fase 1 — se muestra siempre completa, sin "
                       "aplicar el filtro de meses de arriba.")


# ============================================================
# NAVEGACIÓN
# ============================================================
tab_pred, tab_tend, tab_sust, tab_of = st.tabs(
    ["📈 Predicción", "📊 Medidas macro", "🧪 Sustitutos", "🐄 Oferta y demanda"])
with tab_pred:
    pagina_prediccion()
with tab_tend:
    pagina_tendencia()
with tab_sust:
    pagina_sustitutos()
with tab_of:
    pagina_oferta_demanda()
