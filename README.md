# Revisión y Selección de ETFs

Proyecto en Python y Streamlit para comparar ETFs dentro de un mandato de inversión y generar un primer filtro cuantitativo. La idea es simular una tarea real de un analista: revisar un peer group, comparar cada ETF contra su benchmark, detectar alertas básicas y dejar un memo preliminar para análisis posterior.

No es una herramienta de recomendación de inversión. El resultado sirve para priorizar revisión.

## Qué problema resuelve

Cuando se revisan varios ETFs en Excel, el proceso suele repetir los mismos pasos:

- descargar precios y metadata;
- validar que los tickers tengan datos suficientes;
- calcular retornos, drawdowns y ratios de riesgo;
- comparar cada ETF contra un benchmark razonable;
- revisar costo, AUM, volumen y concentración;
- ordenar candidatos y documentar la lectura.

Este proyecto automatiza ese flujo para que el análisis sea más rápido, consistente y fácil de auditar.

## Flujo del proyecto

```text
Universo de ETFs
-> descarga o lectura de snapshot local
-> validación de datos
-> métricas de performance y riesgo
-> comparación contra benchmark
-> revisión de liquidez y costo
-> alertas cuantitativas
-> score de priorización
-> memo preliminar
-> dashboard en Streamlit
```

## Estructura

```text
.
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   │   ├── etf_universe_master.csv
│   │   └── fund_universe.csv
│   └── processed/
│       ├── prices_master.csv
│       ├── daily_returns_master.csv
│       ├── fund_metadata_master.csv
│       └── master_data_quality_summary.csv
├── scripts/
│   ├── build_master_snapshot.py
│   ├── run_data_snapshot.py
│   └── run_full_analysis.py
├── src/
│   └── fund_selection/
│       ├── alpha_vantage.py
│       ├── benchmark.py
│       ├── data_loader.py
│       ├── liquidity_cost.py
│       ├── memo.py
│       ├── performance.py
│       ├── pipeline.py
│       ├── red_flags.py
│       ├── risk.py
│       └── scoring.py
├── requirements.txt
└── README.md
```

## Fuentes de datos

El proyecto usa fuentes públicas:

- `yfinance`: precios históricos, volumen y metadata disponible.
- Alpha Vantage ETF Profile: respaldo para holdings o metadata faltante.
- Snapshots locales en `data/processed`: evitan depender de la API en cada demo.

La metadata pública puede venir incompleta. Cuando falta información, el modelo no asume que el ETF está bien; lo marca como punto de revisión.

## Métricas principales

Performance:

- retorno total;
- CAGR;
- retorno anualizado;
- mejor y peor mes;
- porcentaje de meses positivos.

Riesgo:

- volatilidad anualizada;
- Sharpe ratio;
- Sortino ratio;
- downside deviation;
- maximum drawdown;
- VaR histórico;
- CVaR histórico.

Benchmark fit:

- beta;
- alpha;
- tracking error;
- information ratio;
- correlación;
- R²;
- exceso de retorno anualizado.

Implementación:

- expense ratio / TER;
- AUM;
- volumen promedio;
- volumen promedio en dólares;
- concentración Top 10;
- penalizaciones por alertas.

## Score de priorización

El score combina cinco bloques:

```text
Score de priorización =
  25% Performance
+ 25% Riesgo
+ 20% Benchmark fit
+ 15% Liquidez
+ 15% Costo
- Penalizaciones por alertas
```

El score no decide una compra. Solo ordena el universo para saber qué ETFs vale la pena revisar primero.

Lecturas usadas en el dashboard:

- `Preferido`: candidato fuerte dentro del peer group.
- `Aprobado`: cumple razonablemente el filtro.
- `En observación`: requiere revisión adicional.
- `No prioritario`: no destaca frente al universo seleccionado.

## Cómo correrlo

Instalar dependencias:

```powershell
py -m pip install -r requirements.txt
```

Crear o actualizar el snapshot de datos:

```powershell
py scripts\run_data_snapshot.py
```

Ejecutar el análisis completo:

```powershell
py scripts\run_full_analysis.py
```

Abrir el dashboard:

```powershell
py -m streamlit run dashboard\app.py
```

## Qué muestra la app

- selección guiada de mandato y ETFs comparables;
- benchmark asignado por ETF;
- ranking de selección;
- comparación de performance;
- tabla ejecutiva;
- análisis individual por ETF;
- drawdown, concentración Top 10 y fuentes;
- memo preliminar generado con reglas.

## Límites

- No reemplaza Bloomberg, FactSet, Morningstar ni el factsheet oficial del emisor.
- No evalúa impuestos, suitability, spreads en vivo ni disponibilidad UCITS/offshore.
- No proyecta retornos futuros.
- La data pública puede tener vacíos o retrasos.

## Cómo lo explicaría en una entrevista

Construí una herramienta de investment analytics que automatiza el primer filtro de ETFs. El proyecto toma un universo comparable, asigna benchmarks por ETF, calcula métricas de performance, riesgo, benchmark fit, liquidez y costo, detecta alertas y genera un memo preliminar. La parte de Python replica un proceso que normalmente se haría en Excel, pero de forma más trazable y reutilizable.
