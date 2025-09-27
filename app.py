import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import h3
from pathlib import Path
import sys
from typing import Iterable, Dict

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

L3_BASE = DATA_DIR / 'l3'
L2_BASE = DATA_DIR / 'l2'
AVAILABLE_RES = [7, 8, 9]
CHICAGO_CENTER = {"lat": 41.881832, "lon": -87.623177}

RESOLUTION_GUIDE: Dict[int, Dict[str, str]] = {
    7: {"label": "District-scale", "size": "≈1.2 km across", "story": "Best for citywide comparisons and strategic deployment."},
    8: {"label": "Neighborhood-scale", "size": "≈0.46 km across", "story": "Balances coverage with localized patterns – great for beat discussions."},
    9: {"label": "Street-scale", "size": "≈0.17 km across", "story": "Highlights specific blocks/intersections for tactical response."},
}


@st.cache_data(show_spinner=False)
def load_l3(year: int, month: int, res: int) -> pd.DataFrame:
    path = L3_BASE / f'res={res}' / f'year={year}' / f'month={month:02d}' / f'l3-aggregates-{res}-{year}-{month:02d}.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Missing L3 partition: {path}')
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data(show_spinner=False)
def load_l2(year: int, month: int) -> pd.DataFrame:
    path = L2_BASE / f'year={year}' / f'month={month:02d}' / f'features-{year}-{month:02d}.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Missing L2 partition: {path}')
    keep_cols = [
        'datetime', 'primary_type', 'street_norm', 'community_area_id',
        'district_id', 'ward_id', 'h3_r7', 'h3_r8', 'h3_r9'
    ]
    df = pd.read_parquet(path)
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def summarise_month(l3_df: pd.DataFrame, res: int) -> pd.DataFrame:
    h3_col = f'h3_r{res}'
    summary_cols = ['n_crimes', 'n_arrests', 'low_conf', 'smoothed_rate', 'pooled_smoothed']
    month_summary = (
        l3_df.assign(low_conf=lambda d: d['low_conf'].astype(bool))
             .groupby(h3_col, dropna=True)[summary_cols]
             .agg({
                 'n_crimes': 'sum',
                 'n_arrests': 'sum',
                 'low_conf': 'mean',
                 'smoothed_rate': 'mean',
                 'pooled_smoothed': 'mean'
             })
             .reset_index()
    )
    month_summary.rename(columns={h3_col: 'h3_id', 'low_conf': 'low_conf_share'}, inplace=True)
    month_summary['low_conf_share'] = month_summary['low_conf_share'].round(3)
    month_summary['lat'] = month_summary['h3_id'].apply(lambda h: h3.cell_to_latlng(h)[0])
    month_summary['lon'] = month_summary['h3_id'].apply(lambda h: h3.cell_to_latlng(h)[1])
    return month_summary


def top_value(df: pd.DataFrame, h3_col: str, value_col: str, alias: str) -> pd.DataFrame:
    clean = df.dropna(subset=[h3_col, value_col]).copy()
    if clean.empty:
        return pd.DataFrame(columns=[h3_col, alias, f'{alias}_count'])
    counts = (
        clean.groupby([h3_col, value_col])
             .size()
             .reset_index(name='count')
             .sort_values(['count', value_col], ascending=[False, True])
    )
    top = counts.drop_duplicates(h3_col)
    top.rename(columns={value_col: alias, 'count': f'{alias}_count'}, inplace=True)
    return top[[h3_col, alias, f'{alias}_count']]


def build_context(l2_df: pd.DataFrame, res: int) -> pd.DataFrame:
    h3_col = f'h3_r{res}'
    context_frames = []
    context_frames.append(top_value(l2_df, h3_col, 'primary_type', 'common_crime'))
    context_frames.append(top_value(l2_df, h3_col, 'street_norm', 'signature_street'))
    context_frames.append(top_value(l2_df, h3_col, 'district_id', 'district_id'))
    context_frames.append(top_value(l2_df, h3_col, 'ward_id', 'ward_id'))
    context_frames.append(top_value(l2_df, h3_col, 'community_area_id', 'community_area_id'))

    context = None
    for frame in context_frames:
        if context is None:
            context = frame
        else:
            context = context.merge(frame, on=h3_col, how='outer')
    if context is None:
        context = pd.DataFrame(columns=[h3_col])
    context.rename(columns={h3_col: 'h3_id'}, inplace=True)
    context.fillna({'common_crime': 'UNKNOWN', 'signature_street': 'UNKNOWN'}, inplace=True)
    return context


def focus_share(l2_df: pd.DataFrame, res: int, selected_types: Iterable[str]) -> pd.DataFrame:
    h3_col = f'h3_r{res}'
    base = l2_df.dropna(subset=[h3_col]).copy()
    if base.empty:
        return pd.DataFrame(columns=['h3_id', 'incident_total', 'focus_count', 'focus_share'])
    total = base.groupby(h3_col).size().reset_index(name='incident_total')
    if selected_types:
        focus = base[base['primary_type'].isin(selected_types)].groupby(h3_col).size().reset_index(name='focus_count')
    else:
        focus = total.rename(columns={'incident_total': 'focus_count'})
    merged = total.merge(focus, on=h3_col, how='left').fillna({'focus_count': 0})
    merged['focus_share'] = (merged['focus_count'] / merged['incident_total']).replace([pd.NA, float('inf')], 0).fillna(0).round(3)
    merged.rename(columns={h3_col: 'h3_id'}, inplace=True)
    return merged


def build_geojson(summary: pd.DataFrame) -> Dict:
    features = []
    for _, row in summary.iterrows():
        boundary = h3.cell_to_boundary(row['h3_id'], geo_json=True)
        features.append({
            'type': 'Feature',
            'id': row['h3_id'],
            'properties': {
                'h3_id': row['h3_id']
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [boundary]
            }
        })
    return {'type': 'FeatureCollection', 'features': features}


def plot_hotspot_map(summary: pd.DataFrame, res: int, year: int, month: int):
    geojson = build_geojson(summary)
    color_max = summary['n_crimes'].max() or 1
    fig = px.choropleth_mapbox(
        summary,
        geojson=geojson,
        locations='h3_id',
        color='n_crimes',
        color_continuous_scale='YlOrRd',
        range_color=(0, color_max),
        hover_data={
            'n_crimes': True,
            'n_arrests': True,
            'smoothed_rate': ':.2f',
            'low_conf_share': ':.2f',
            'focus_share': ':.2f',
            'common_crime': True,
            'signature_street': True,
            'district_id': True,
            'ward_id': True,
            'community_area_id': True,
        },
        featureidkey='properties.h3_id',
        title=f'Hotspots (r{res}) – {year}-{month:02d}',
        mapbox_style='open-street-map',
        center=CHICAGO_CENTER,
        zoom=10.5,
        opacity=0.75,
    )
    fig.update_layout(margin={'r': 0, 't': 50, 'l': 0, 'b': 0})
    return fig


def h3_story(res: int) -> str:
    guide = RESOLUTION_GUIDE.get(res)
    if not guide:
        return f'Resolution r{res}.'
    return f"Resolution r{res} — {guide['label']} ({guide['size']}). {guide['story']}"


def narrative_bullets(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "- No incidents recorded for this slice."
    lines = []
    for _, row in summary.sort_values('n_crimes', ascending=False).head(3).iterrows():
        street = row.get('signature_street') or 'unknown streets'
        crime = row.get('common_crime') or 'mixed offenses'
        share = row.get('focus_share')
        focus_txt = f", focus selection share {share:.0%}" if share not in (None, 0) else ""
        lines.append(
            f"- Hex `{row['h3_id']}` near **{street}** logged **{int(row['n_crimes'])} incidents**; most common offense: **{crime}**{focus_txt}."
        )
    return "\n".join(lines)


def main():
    st.set_page_config(page_title='Chicago Crime Hotspots', layout='wide')
    st.title('Chicago Crime Hotspot Storytelling')

    col1, col2, col3 = st.columns(3)
    with col1:
        res = st.selectbox('H3 resolution', AVAILABLE_RES, index=AVAILABLE_RES.index(9))
    years = sorted(int(p.name.split('=')[1]) for p in (L3_BASE / f'res={res}').glob('year=*') if p.is_dir())
    if not years:
        st.error('No L3 data found for the selected resolution.')
        st.stop()
    with col2:
        year = st.selectbox('Year', years, index=len(years) - 1)
    months = sorted(int(p.name.split('=')[1]) for p in (L3_BASE / f'res={res}' / f'year={year}').glob('month=*') if p.is_dir())
    if not months:
        st.error('No months available for selected year/resolution.')
        st.stop()
    with col3:
        month = st.selectbox('Month', months, index=len(months) - 1, format_func=lambda m: f'{m:02d}')

    st.info(h3_story(res))

    l3_df = load_l3(year, month, res)
    l2_df = load_l2(year, month)

    primary_types = sorted([pt for pt in l2_df['primary_type'].dropna().unique()])
    default_focus = []
    focus_types = st.multiselect('Focus on specific crime types (optional)', primary_types, default=default_focus)

    summary = summarise_month(l3_df, res)
    context = build_context(l2_df, res)
    share = focus_share(l2_df, res, focus_types)
    summary = summary.merge(context, on='h3_id', how='left').merge(share, on='h3_id', how='left')
    summary['focus_share'] = summary['focus_share'].fillna(1.0 if not focus_types else 0.0)

    total_incidents = int(summary['n_crimes'].sum())
    top_hex = summary['n_crimes'].max() if not summary.empty else 0
    avg_focus = summary['focus_share'].replace([np.inf, -np.inf], np.nan).fillna(0).mean()

    m1, m2, m3 = st.columns(3)
    m1.metric('Total incidents (month)', f'{total_incidents:,}')
    m2.metric('Busiest hex incidents', f'{int(top_hex)}')
    m3.metric('Avg focus share', f'{avg_focus:.0%}')

    trend_fig = px.line(
        l3_df.groupby('date', as_index=False)['n_crimes'].sum(),
        x='date', y='n_crimes', markers=True,
        title=f'Daily incidents – {year}-{month:02d}'
    )
    trend_fig.update_layout(yaxis_title='Incidents', xaxis_title='Date', height=360)
    st.plotly_chart(trend_fig, use_container_width=True)

    map_fig = plot_hotspot_map(summary, res, year, month)
    st.plotly_chart(map_fig, use_container_width=True)

    col_story, col_breakdown = st.columns([1, 1])
    with col_story:
        st.subheader('What stands out this month')
        st.markdown(narrative_bullets(summary))
    with col_breakdown:
        st.subheader('Crime mix (top 10)')
        crime_mix = (
            l2_df['primary_type']
                 .value_counts()
                 .head(10)
                 .reset_index()
                 .rename(columns={'index': 'primary_type', 'primary_type': 'count'})
        )
        crime_fig = px.bar(crime_mix, x='count', y='primary_type', orientation='h', title='Top crime categories')
        crime_fig.update_layout(yaxis_title='', xaxis_title='Incidents')
        st.plotly_chart(crime_fig, use_container_width=True)

    st.subheader('Signature streets per hotspot')
    streets = (
        summary[['h3_id', 'n_crimes', 'common_crime', 'signature_street', 'district_id', 'focus_share']]
        .sort_values('n_crimes', ascending=False)
        .fillna({'focus_share': 0, 'common_crime': 'UNKNOWN', 'signature_street': 'UNKNOWN'})
    )
    st.dataframe(streets.rename(columns={
        'h3_id': 'H3 cell',
        'n_crimes': 'Incidents',
        'common_crime': 'Common offense',
        'signature_street': 'Signature street',
        'district_id': 'District',
        'focus_share': 'Focus share'
    }))

    with st.expander('Optional: recompute clustering overlay (UMAP + HDBSCAN)'):
        run_clusters = st.checkbox('Rebuild clusters for this slice')
        if run_clusters:
            try:
                from l3_clustering_prototype import run_clustering
                run_clustering(year, month, res=res)
                st.success('Cluster run complete. Check data/l3/clusters/ for outputs.')
            except Exception as exc:
                st.error(f'Clustering failed: {exc}\nInstall dependencies: pip install umap-learn hdbscan')


if __name__ == '__main__':
    main()
