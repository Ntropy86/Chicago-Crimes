# Roadmap – Chicago Crimes Project

This roadmap captures the planned extensions beyond the currently shipping L1 → L2 → L3 medallion pipeline. Each initiative lists purpose, prerequisites, expected deliverables, and suggested learning focus.

## 1. Hotspot Storytelling (Baseline **shipped**)

- **Delivered**: Streamlit dashboard (`app.py`) and notebook (`notebooks/01_hotspot_storytelling.ipynb`) showing daily trends, H3 hotspot maps, low-confidence flags, and optional clustering overlay.
- **Next Enhancements**: Add drill-down filters (crime type, arrest flag) and narrative export for weekly reporting.
- **Intern Learning Focus**: Capture qualitative insights, evaluate deterministic vs. HDBSCAN differences, and log follow-up questions for stakeholders.

## 2. Forecasting Engine (Post-demo)

- **Goal**: Produce reliable forward-looking crime incident forecasts backed by defensible metrics.
- **Steps**:
  1. Baseline: seasonal naive & SARIMA on monthly counts per district/H3; log MAE/MAPE using rolling-origin evaluation.
  2. Hybrid: augment with LSTM or Temporal Fusion Transformer using L2 features (temporal encodings, arrests, location type). Compare vs. baseline.
  3. Model management: persist experiment configs, metrics, and hold-out predictions in `reports/forecasting/`.
- **Prereqs**: Select target aggregation level (citywide/district/H3 bucket), decide forecast horizon, ensure data splitting respects leakage.
- **Deliverables**: Reproducible training script, validation report, forecast parquet for future months.
- **Intern Learning Focus**: Time-series evaluation, handling exogenous variables, model selection, overfitting detection.

## 3. Population & Census Enrichment (Post-demo enabler)

- **Goal**: Integrate demographic context for both descriptive dashboards and forecasting features.
- **Steps**:
  - Source: Identify ACS tables (population, median income, daytime population, age bands) for Chicago census tracts or community areas.
  - Normalize geography: map tracts/community areas to H3 (resolution mapping or area-weighted interpolation).
  - Persist enriched datasets under `data/external/` with version metadata.
- **Deliverables**: ETL script/notebook, documentation on feature provenance, joinable dataset with keys aligning to L3 outputs.
- **Intern Learning Focus**: Geospatial data wrangling, unit harmonization, metadata hygiene.

## 4. Production-Quality Visual Platform (Mid term)

- **Goal**: Promote notebooks to an interactive dashboard (Streamlit/Bokeh/Plotly Dash) telling an “Insights per Month” story.
- **Prereqs**: Hotspot storytelling assets, stable forecasting outputs, census enrichment.
- **Deliverables**: MVP app with tabs for current hotspots, forecast scenarios, and demographic overlays; deployable locally.
- **Intern Learning Focus**: UX for analytics, transforming exploratory code into maintainable modules, stakeholder storytelling.

## 5. Agentic Exploration (Stretch goal)

- **Goal**: Natural-language interface backed by curated SQL templates and vector search over documentation.
- **Dependencies**: Solid datasets (L1–L3 + enrichment), clear governance, validated visual platform.
- **Deliverables**: Proof-of-concept using LangChain (or alternative) that translates questions into parameterized queries and returns plots/explanations.
- **Intern Learning Focus**: Prompt design, retrieval-augmented generation, safe execution guards.

## Operating Principles

1. **Document reality first** – README and runbooks must reflect shipping code. Use this roadmap to advertise what’s next without overpromising.
2. **Version every experiment** – notebooks/scripts should log input partitions, parameters, metric definitions.
3. **Feedback loops** – after each milestone, capture lessons learned and update this roadmap (scope changes, new dependencies).
4. **Learning Mindset** – treat each initiative as a coaching ladder: read existing pipeline code, replicate outputs, then extend.

_Last updated: 2025-09-26_
