# 🐄 Inteligencia de Mercado — Harina de Carne y Hueso (HCH)
## Hallazgos del Análisis de Datos + Propuesta de Fuente Adicional

**Autora:** Natalia Vindrola Muñoz | Zootecnista + M.S. Inteligencia Analítica de Datos  
**Proyecto:** Pet Food Protein Intelligence — Colombia & LATAM  
**Fecha:** Mayo 2026  
**Fuentes analizadas:** UN Comtrade API · DIAN (Importaciones 2026)

---

## 1. Metodología y Fuentes de Datos

Se construyó una base de datos SQLite (`mercado_hch.db`) combinando dos fuentes oficiales:

| Fuente | Descripción | Período | Registros limpios |
|--------|-------------|---------|-------------------|
| **UN Comtrade API** | Estadísticas anuales de importación hacia Colombia por partida arancelaria | 2020–2024 | 881 |
| **DIAN** | Declaraciones reales de importación colombiana | Enero–Febrero 2026 | 69 |

### Partidas arancelarias monitoreadas

| Código | Ingrediente |
|--------|-------------|
| 2301100000 | Harina de carne y hueso (HCH) |
| 2301200000 | Harina de pescado |
| 1208100000 | Harina de soya |
| 2835260000 | Fosfato dicálcico |

---

## 2. Hallazgos Principales

### 2.1 Precios de importación por ingrediente (USD/ton) — Comtrade limpio

| Producto | 2020 | 2021 | 2022 | 2023 | 2024 |
|----------|------|------|------|------|------|
| **HCH** | $620 | $875 | $943 | $884 | $1,151 |
| Harina de pescado | $1,170 | $1,265 | $1,624 | $1,699 | $1,625 |
| Fosfato | $527 | $723 | $1,131 | $993 | N/D |
| Soya nacional | N/D | N/D | N/D | $1,403 | N/D |

### 2.2 Hallazgo #1 — El precio del HCH casi se duplicó en 4 años

El precio de importación del HCH hacia Colombia pasó de **$620 USD/ton en 2020**
a **$1,151 USD/ton en 2024** — un incremento del **86% en cuatro años.**

Se identifican tres momentos clave:

- **2021:** Primer salto significativo (+41%) — recuperación post-COVID y aumento
  global de demanda de proteína animal
- **2022:** Pico de la guerra en Ucrania — encarecimiento de energía y fletes
  impacta los costos de rendering a nivel global
- **2023:** Leve corrección (-6%) — normalización parcial de cadenas de suministro
- **2024:** Nuevo récord histórico ($1,151/ton) — señal de presión estructural,
  no coyuntural

> **Implicación para fabricantes de alimento para mascotas en Colombia:**
> Un ingrediente que representaba ~$620/ton en 2020 hoy cuesta ~$1,151/ton.
> Para una planta que consume 500 toneladas/mes, eso equivale a
> **$265,500 USD adicionales por mes** en costo de materia prima.

### 2.3 Hallazgo #2 — Colombia no importa HCH de forma significativa

Al cruzar los datos de la DIAN (enero–febrero 2026), **no se encontraron
declaraciones de importación de HCH**. Los únicos registros encontrados
corresponden a fosfato dicálcico (69 registros).

**¿Por qué?** Colombia es un país ganadero con una industria frigorífica
de gran escala (Minerva Foods, Frigorífico Guadalupe, Frigorínorte, entre otros).
El HCH es un **subproducto del proceso de faena** — se produce directamente en
los frigoríficos a partir de huesos, recortes y tejidos no comestibles mediante
el proceso de rendering.

Esto significa que:

- El precio del HCH en Colombia **no está determinado por el mercado
  internacional de importaciones** como ocurre con la harina de pescado o el fosfato
- Su costo y disponibilidad dependen directamente de la
  **dinámica de sacrificio de ganado bovino en Colombia**
- Las estadísticas de Comtrade capturan el *precio de referencia internacional*,
  pero no el precio real al que acceden los fabricantes colombianos

### 2.4 Hallazgo #3 — Diferente perfil de riesgo por ingrediente

| Ingrediente | Origen en Colombia | Principal driver de precio |
|-------------|-------------------|---------------------------|
| **HCH** | **Doméstico** (subproducto frigoríficos) | Volumen de sacrificio bovino nacional |
| Harina de pescado | **Importado** (principalmente Perú) | Cuotas de anchoveta, El Niño, TRM |
| Fosfato dicálcico | **Importado** (Marruecos, China, USA) | Precio internacional + TRM + fletes |
| Harina de soya | **Mixto** (producción nacional + imports) | Cosecha colombiana + precios Argentina/Brasil |

> Este hallazgo es crítico para fabricantes como Nestlé Purina en Colombia:
> **cada ingrediente tiene un perfil de riesgo completamente diferente**
> y por lo tanto requiere una estrategia de abastecimiento distinta.

---

## 3. Limitación Identificada y Propuesta de Solución

### 3.1 El problema

Los datos de Comtrade reflejan precios de importación internacionales.
Los datos de DIAN confirman que HCH **no se importa** en cantidad significativa.

Esto genera una brecha analítica: **no tenemos datos del precio real del HCH
producido y comercializado domésticamente en Colombia.**

Sin esa información, el modelo de forecasting predice el precio de referencia
internacional del HCH, pero no el precio que pagan realmente Nestlé Purina,
Italcol o Campollo cuando compran HCH a un frigorífico colombiano.

### 3.2 La solución — Integrar datos de FEDEGAN

**FEDEGAN** (Federación Colombiana de Ganaderos) es la fuente oficial de
estadísticas del sector bovino en Colombia. Publica mensualmente:

- **Volumen de sacrificio bovino** por departamento (cabezas/mes)
- **Precio del ganado en pie** (COP/kg)
- **Inventario ganadero** por región
- **Tendencias estacionales** de sacrificio

#### ¿Por qué FEDEGAN es la fuente correcta para modelar el precio del HCH?

La lógica es directa:

```
Mayor sacrificio bovino
        ↓
Mayor disponibilidad de materia prima para rendering
        ↓
Mayor oferta de HCH en el mercado doméstico
        ↓
Presión a la baja en el precio del HCH
```

Y en sentido contrario:

```
Menor sacrificio (sequía, fiebre aftosa, restricciones sanitarias)
        ↓
Menor disponibilidad de huesos y recortes para rendering
        ↓
Escasez de HCH doméstico
        ↓
Aumento de precio o necesidad de importar a precio internacional
```

Esta relación es la que **ningún analista de datos convencional detectaría**
sin conocimiento del proceso productivo. Es el valor diferencial de
tener formación como Zootecnista combinada con experiencia en frigoríficos
industriales.

### 3.3 Cómo integrar FEDEGAN al modelo

**Fuente de datos:** `https://www.fedegan.org.co/estadisticas/sacrificio`  
**Frecuencia:** Mensual  
**Formato:** Reportes descargables en Excel

```python
# Estructura propuesta para la nueva tabla en mercado_hch.db

fedegan_mensual = {
    "fecha":                  "YYYY-MM-01",   # primer día del mes
    "sacrificio_bovino_cabezas": int,          # cabezas sacrificadas en Colombia
    "departamento":           str,             # o "NACIONAL" para el total
    "precio_ganado_cop_kg":   float,           # precio del ganado en pie
    "fuente":                 "FEDEGAN"
}
```

**Nueva variable en el modelo de forecasting:**

```python
# El sacrificio bovino se usa como regresor externo en Prophet
modelo = Prophet()
modelo.add_regressor('sacrificio_bovino_cabezas')
modelo.fit(df_train)
```

Con esto, el modelo no solo proyecta el precio del HCH por tendencia histórica,
sino que incorpora la **variable causal real** que determina su disponibilidad
y precio en Colombia.

### 3.4 Impacto esperado en la calidad del modelo

| Sin FEDEGAN | Con FEDEGAN |
|-------------|-------------|
| Forecast basado solo en tendencia internacional (Comtrade) | Forecast que incorpora dinámica de oferta doméstica real |
| No captura estacionalidad del sacrificio bovino en Colombia | Captura picos de sacrificio (fin de año, Semana Santa) |
| Precio predicho = referencia internacional | Precio predicho = proxy del precio doméstico colombiano |
| Útil como benchmark | Útil como herramienta de decisión real para procurement |

---

## 4. Estado Actual del Proyecto y Próximos Pasos

```
✅  Pipeline de datos construido (Comtrade + DIAN → SQLite)
✅  881 registros limpios validados
✅  Hallazgos clave identificados (86% de aumento, origen doméstico del HCH)
✅  Brecha analítica identificada y solución propuesta (FEDEGAN)

🔜  Descargar datos históricos de sacrificio FEDEGAN (2020–2026)
🔜  Integrar tabla FEDEGAN a mercado_hch.db
🔜  Construir modelo Prophet con regresor externo de sacrificio bovino
🔜  Generar visualizaciones (EDA + forecast)
🔜  Construir dashboard en Power BI
🔜  Publicar en GitHub con README en inglés y español
```

---

## 5. Por Qué Este Análisis es Relevante para la Industria

Para empresas como **Nestlé Purina, Italcol, Campollo o Cargill Colombia**,
este modelo responde una pregunta que hoy no tiene respuesta estructurada:

> *"¿Cuándo debo pre-negociar contratos de HCH con los frigoríficos,
> y cuándo conviene evaluar ingredientes alternativos como harina de
> vísceras de pollo o harina de pescado?"*

Un analista sin formación en ciencias animales modelaría el HCH como
un commodity importado más. Este análisis demuestra que en Colombia
**es un subproducto doméstico cuya oferta sigue el ciclo ganadero**,
lo que cambia completamente la estrategia de abastecimiento.

Esa es la diferencia entre un análisis de datos genérico y
**inteligencia de mercado con contexto sectorial.**

---

*Documento generado como parte del portafolio de proyectos de Natalia Vindrola Muñoz*  
*github.com/NataliaVindrola · linkedin.com/in/natalia-vindrola-aa3976191*
