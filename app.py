import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import h3
import sys

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

L3_BASE = DATA_DIR / 'l3'
AVAILABLE_RES = [7, 8, 9]


def _list_years(res: int):
    res_dir = L3_BASE / f'res={res}'
    return sorted(int(p.name.split('=')[1]) for p in res_dir.glob('year=*') if p.is_dir())


def _list_months(res: int, year: int):
    year_dir = L3_BASE / f'res={res}' / f'year={year}'
    return sorted(int(p.name.split('=')[1]) for p in year_dir.glob('month=*') if p.is_dir())


def load_l3(year: int, month: int, res: int) -> pd.DataFrame:
    path = L3_BASE / f'res={res}' / f'year={year}' / f'month={month:02d}' / f'l3-aggregates-{res}-{year}-{month:02d}.parquet'
    if not path.exists():
        st.error(f"Missing L3 partition: {path}")
        st.stop()
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def summarise_month(df: pd.DataFrame, h3_res: int) -> pd.DataFrame:
    summary_cols = ['n_crimes', 'n_arrests', 'low_conf', 'smoothed_rate', 'pooled_smoothed']
    out = (
        df.assign(low_conf=lambda d: d['low_conf'].astype(bool))
          .groupby(f'h3_r{h3_res}', dropna=True)[summary_cols]
          .agg({'n_crimes': 'sum', 'n_arrests': 'sum', 'low_conf': 'mean', 'smoothed_rate': 'mean', 'pooled_smoothed': 'mean'})
          .reset_index()
    )
    out.rename(columns={f'h3_r{h3_res}': 'h3_id', 'low_conf': 'low_conf_share'}, inplace=True)
    out['low_conf_share'] = out['low_conf_share'].round(3)
    out['lat'] = out['h3_id'].apply(lambda h: h3.cell_to_latlng(h)[0])
    out['lon'] = out['h3_id'].apply(lambda h: h3.cell_to_latlng(h)[1])
    return out


def plot_daily_trend(df: pd.DataFrame, year: int, month: int):
    daily = df.groupby('date', as_index=False)['n_crimes'].sum()
    fig = px.line(daily, x='date', y='n_crimes', markers=True,
                  title=f'Daily incidents – {year}-{month:02d}')
    fig.update_layout(yaxis_title='Incidents', xaxis_title='Date')
    return fig


def plot_hotspot_map(summary: pd.DataFrame, res: int, year: int, month: int):
    fig = px.scatter_geo(
        summary,
        lat='lat',
        lon='lon',
        size='n_crimes',
        color='smoothed_rate',
        hover_data=['h3_id', 'n_crimes', 'n_arrests', 'low_conf_share', 'pooled_smoothed'],
        scope='north america',
        title=f'H3 hotspots (res={res}) – {year}-{month:02d}',
    )
    fig.update_layout(height=600)
    return fig


def main():
    st.set_page_config(page_title='Chicago Crime Hotspots', layout='wide')
    st.title('Chicago Crime Hotspot Storytelling')

    col1, col2, col3 = st.columns(3)
    with col1:
        res = st.selectbox('H3 resolution', AVAILABLE_RES, index=AVAILABLE_RES.index(9))
    years = _list_years(res)
    if not years:
        st.error('No L3 data found for the selected resolution.')
        st.stop()
    with col2:
        year = st.selectbox('Year', years, index=len(years) - 1)
    months = _list_months(res, year)
    if not months:
        st.error('No months available for selected year/resolution.')
        st.stop()
    with col3:
        month = st.selectbox('Month', months, index=len(months) - 1, format_func=lambda m: f'{m:02d}')

    l3_df = load_l3(year, month, res)
    month_summary = summarise_month(l3_df, res)

    trend_fig = plot_daily_trend(l3_df, year, month)
    hotspot_fig = plot_hotspot_map(month_summary, res, year, month)

    st.plotly_chart(trend_fig, use_container_width=True)
    st.plotly_chart(hotspot_fig, use_container_width=True)

    st.subheader('Top hotspots (by incidents)')
    st.dataframe(month_summary.sort_values('n_crimes', ascending=False).head(15))

    st.subheader('Low-confidence cells (share > 0.5)')
    low_conf = month_summary[month_summary['low_conf_share'] > 0.5].sort_values('low_conf_share', ascending=False)
    if low_conf.empty:
        st.write('No cells flagged as low confidence (threshold 0.5).')
    else:
        st.dataframe(low_conf[['h3_id', 'n_crimes', 'low_conf_share']])

    with st.expander('Optional: run clustering prototype (UMAP + HDBSCAN)'):
        run_clusters = st.checkbox('Compute / refresh clusters for this month')
        if run_clusters:
            try:
                from l3_clustering_prototype import run_clustering
                run_clustering(year, month, res=res)
                st.success('Cluster run complete. Parquet written under data/l3/clusters/.')
            except Exception as exc:
                st.error(f'Clustering failed: {exc}\nInstall dependencies: pip install umap-learn hdbscan')


if __name__ == '__main__':
    main()
