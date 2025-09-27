import warnings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import h3
from pathlib import Path
import sys
import calendar
from typing import Iterable, Dict, Tuple

from collections.abc import Iterable as _Iterable

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import expand_categories

L3_BASE = DATA_DIR / 'l3'
L2_BASE = DATA_DIR / 'l2'
AVAILABLE_RES = [7, 8, 9]
CHICAGO_CENTER = {"lat": 41.881832, "lon": -87.623177}

RESOLUTION_GUIDE: Dict[int, Dict[str, str]] = {
    7: {"label": "District-scale", "size": "≈1.2 km across", "story": "Best for citywide comparisons and strategic deployment."},
    8: {"label": "Neighborhood-scale", "size": "≈0.46 km across", "story": "Balances coverage with localized patterns – great for beat discussions."},
    9: {"label": "Street-scale", "size": "≈0.17 km across", "story": "Highlights specific blocks/intersections for tactical response."},
}


warnings.filterwarnings('ignore', message='.*choropleth_mapbox.*', category=DeprecationWarning)


def _normalize_months(months) -> Tuple[int, ...]:
    if isinstance(months, (list, tuple, set)):
        return tuple(sorted(int(m) for m in months))
    if isinstance(months, _Iterable) and not isinstance(months, (str, bytes)):
        return tuple(sorted(int(m) for m in list(months)))
    return (int(months),)


@st.cache_data(show_spinner=False)
def load_l3(year: int, months: Tuple[int, ...] | int, res: int) -> pd.DataFrame:
    months_tuple = _normalize_months(months)
    frames: list[pd.DataFrame] = []
    for month in months_tuple:
        path = L3_BASE / f'res={res}' / f'year={year}' / f'month={month:02d}' / f'l3-aggregates-{res}-{year}-{month:02d}.parquet'
        if not path.exists():
            raise FileNotFoundError(f'Missing L3 partition: {path}')
        df = pd.read_parquet(path)
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = month
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_l2(year: int, months: Tuple[int, ...] | int) -> pd.DataFrame:
    months_tuple = _normalize_months(months)
    keep_cols = [
        'datetime', 'primary_type', 'crime_category', 'street_norm', 'block_address', 'arrest_made', 'community_area_id',
        'district_id', 'ward_id', 'h3_r7', 'h3_r8', 'h3_r9'
    ]
    frames: list[pd.DataFrame] = []
    for month in months_tuple:
        path = L2_BASE / f'year={year}' / f'month={month:02d}' / f'features-{year}-{month:02d}.parquet'
        if not path.exists():
            raise FileNotFoundError(f'Missing L2 partition: {path}')
        df = pd.read_parquet(path)
        df['month'] = month
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])

        if 'arrest_made' not in df.columns and 'arrest' in df.columns:
            df['arrest_made'] = df['arrest']
        if 'primary_type' not in df.columns and 'crime_type' in df.columns:
            df['primary_type'] = df['crime_type']
        if 'street_norm' not in df.columns:
            if 'block_address' in df.columns:
                df['street_norm'] = df['block_address']
            elif 'block' in df.columns:
                df['street_norm'] = df['block']
        if 'block_address' not in df.columns and 'block' in df.columns:
            df['block_address'] = df['block']

        available_cols = [c for c in keep_cols if c in df.columns]
        df = df[available_cols + ['month']].copy()
        if 'arrest_made' in df.columns:
            try:
                df['arrest_made'] = df['arrest_made'].astype('boolean').fillna(False)
            except Exception:
                df['arrest_made'] = df['arrest_made'].apply(lambda x: bool(x) if pd.notna(x) else False).astype('boolean')
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise_period(l3_df: pd.DataFrame, res: int) -> pd.DataFrame:
    h3_col = f'h3_r{res}'
    if l3_df.empty or h3_col not in l3_df.columns:
        return pd.DataFrame(columns=['h3_id'])
    working = l3_df.assign(low_conf=lambda d: d['low_conf'].astype(bool))
    grouped = working.groupby(h3_col, dropna=True)
    aggregates = grouped[['n_crimes', 'n_arrests']].sum().reset_index()
    aggregates.rename(columns={h3_col: 'h3_id'}, inplace=True)
    aggregates['low_conf_share'] = grouped['low_conf'].mean().round(3).to_numpy()

    for rate_col in ['smoothed_rate', 'pooled_smoothed']:
        if rate_col in working.columns:
            try:
                weighted = grouped.apply(
                    lambda g, col=rate_col: np.average(g[col], weights=g['n_crimes'].clip(lower=1)),
                    include_groups=False,
                )
            except TypeError:
                weighted = grouped.apply(lambda g, col=rate_col: np.average(g[col], weights=g['n_crimes'].clip(lower=1)))
            aggregates[rate_col] = weighted.to_numpy()
        else:
            aggregates[rate_col] = np.nan

    aggregates['lat'] = aggregates['h3_id'].apply(lambda h: h3.cell_to_latlng(h)[0])
    aggregates['lon'] = aggregates['h3_id'].apply(lambda h: h3.cell_to_latlng(h)[1])
    return aggregates


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
    top = counts.drop_duplicates(h3_col).copy()
    top.rename(columns={value_col: alias, 'count': f'{alias}_count'}, inplace=True)
    return top[[h3_col, alias, f'{alias}_count']]


def build_context(l2_df: pd.DataFrame, res: int) -> pd.DataFrame:
    h3_col = f'h3_r{res}'
    context_frames = []
    if 'primary_type' in l2_df.columns:
        context_frames.append(top_value(l2_df, h3_col, 'primary_type', 'common_crime'))
    if 'street_norm' in l2_df.columns:
        context_frames.append(top_value(l2_df, h3_col, 'street_norm', 'signature_street'))
    if 'block_address' in l2_df.columns:
        context_frames.append(top_value(l2_df, h3_col, 'block_address', 'signature_block'))
    if 'district_id' in l2_df.columns:
        context_frames.append(top_value(l2_df, h3_col, 'district_id', 'district_id'))
    if 'ward_id' in l2_df.columns:
        context_frames.append(top_value(l2_df, h3_col, 'ward_id', 'ward_id'))
    if 'community_area_id' in l2_df.columns:
        context_frames.append(top_value(l2_df, h3_col, 'community_area_id', 'community_area_id'))

    context = None
    for frame in context_frames:
        context = frame if context is None else context.merge(frame, on=h3_col, how='outer')
    if context is None:
        context = pd.DataFrame(columns=[h3_col])
    context = context.copy()
    if h3_col in context.columns:
        context.rename(columns={h3_col: 'h3_id'}, inplace=True)
    context.fillna({'common_crime': 'UNKNOWN', 'signature_street': 'UNKNOWN', 'signature_block': 'UNKNOWN'}, inplace=True)
    return context


def focus_share(l2_df: pd.DataFrame, res: int, selected_types: Iterable[str]) -> pd.DataFrame:
    h3_col = f'h3_r{res}'
    base = l2_df.dropna(subset=[h3_col]).copy()
    if base.empty:
        return pd.DataFrame(columns=['h3_id', 'incident_total', 'focus_count', 'focus_share'])
    total = base.groupby(h3_col).size().reset_index(name='incident_total')
    if selected_types and 'primary_type' in base.columns:
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
        try:
            boundary = h3.cell_to_boundary(row['h3_id'])
        except Exception:
            continue
        coords = [[lon, lat] for lat, lon in boundary]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append({
            'type': 'Feature',
            'id': row['h3_id'],
            'properties': {'h3_id': row['h3_id']},
            'geometry': {'type': 'Polygon', 'coordinates': [coords]},
        })
    return {'type': 'FeatureCollection', 'features': features}


def plot_hotspot_map(summary: pd.DataFrame, res: int, title: str, color_metric: str, metric_label: str, color_range: Tuple[float, float]):
    geojson = build_geojson(summary)
    if summary.empty or color_metric not in summary.columns:
        return px.choropleth_mapbox()
    display = summary.copy()
    display['display_street'] = (
        display.get('signature_block', pd.Series(dtype=str))
        .fillna('Street unknown')
        .replace({'UNKNOWN': 'Street unknown'})
        .str.title()
    )
    display['display_crime'] = (
        display.get('common_crime', pd.Series(dtype=str))
        .fillna('Mixed offenses')
        .replace({'UNKNOWN': 'Mixed offenses'})
    )
    display['display_focus'] = display.get('focus_share', pd.Series(dtype=float)).fillna(0.0)
    display['display_low_conf'] = display.get('low_conf_share', pd.Series(dtype=float)).fillna(0.0)
    display['display_rate'] = display.get('smoothed_rate', pd.Series(dtype=float)).fillna(0.0)
    display['display_arrests'] = display.get('n_arrests', pd.Series(dtype=float)).fillna(0.0)
    color_min, color_max = color_range
    fig = px.choropleth_mapbox(
        display,
        geojson=geojson,
        locations='h3_id',
        color=color_metric,
        color_continuous_scale=['#9bd174', '#6bbf59', '#f5e663', '#f9a64c', '#ef6248', '#b0202f'],
        range_color=(color_min, color_max),
        featureidkey='properties.h3_id',
        title=title,
        mapbox_style='carto-positron',
        center=CHICAGO_CENTER,
        zoom=11.0,
        opacity=0.45,
    )
    hover_data = display[['display_street', 'display_crime', 'n_crimes', 'display_arrests', 'display_rate', 'display_low_conf', 'display_focus']].to_numpy()
    fig.update_traces(
        marker_line_width=0.6,
        marker_line_color='rgba(255,255,255,0.45)',
        customdata=hover_data,
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'Top offense: %{customdata[1]}<br>'
            'Incidents: %{customdata[2]:,.0f}<br>'
            'Arrests: %{customdata[3]:,.0f}<br>'
            'Smoothed arrest rate: %{customdata[4]:.0%}<br>'
            'Low-confidence share: %{customdata[5]:.0%}<br>'
            'Focus share: %{customdata[6]:.0%}<extra></extra>'
        )
    )
    fig.update_layout(
        margin={'r': 0, 't': 50, 'l': 0, 'b': 0},
        coloraxis_colorbar=dict(title=metric_label, ticksuffix=''),
    )
    return fig


def plot_focus_map(summary: pd.DataFrame, res: int, title: str):
    geojson = build_geojson(summary)
    if summary.empty or 'focus_share' not in summary.columns:
        return px.choropleth_mapbox()
    display = summary.copy()
    display['display_street'] = (
        display.get('signature_block', pd.Series(dtype=str))
        .fillna('Street unknown')
        .replace({'UNKNOWN': 'Street unknown'})
        .str.title()
    )
    display['display_crime'] = (
        display.get('common_crime', pd.Series(dtype=str))
        .fillna('Mixed offenses')
        .replace({'UNKNOWN': 'Mixed offenses'})
    )
    display['display_focus'] = display.get('focus_share', pd.Series(dtype=float)).fillna(0.0)
    display['display_incidents'] = display.get('n_crimes', pd.Series(dtype=float)).fillna(0.0)
    fig = px.choropleth_mapbox(
        display,
        geojson=geojson,
        locations='h3_id',
        color='focus_share',
        color_continuous_scale=['#9bd174', '#f1f18c', '#f7b267', '#ef6248', '#b0202f'],
        range_color=(0, 1),
        featureidkey='properties.h3_id',
        title=title,
        mapbox_style='carto-positron',
        center=CHICAGO_CENTER,
        zoom=11.0,
        opacity=0.45,
    )
    hover_data = display[['display_street', 'display_crime', 'display_incidents', 'display_focus']].to_numpy()
    fig.update_traces(
        marker_line_width=0.6,
        marker_line_color='rgba(255,255,255,0.45)',
        customdata=hover_data,
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'Top offense: %{customdata[1]}<br>'
            'Incidents (period): %{customdata[2]:,.0f}<br>'
            'Selected crime share: %{customdata[3]:.0%}<extra></extra>'
        )
    )
    fig.update_layout(
        margin={'r': 0, 't': 50, 'l': 0, 'b': 0},
        coloraxis_colorbar=dict(title='Selected crime share', ticksuffix=''),
    )
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
        street = row.get('signature_block')
        if pd.isna(street) or street == 'UNKNOWN':
            street = row.get('signature_street')
        if pd.isna(street) or street == 'UNKNOWN':
            street = None

        crime = row.get('common_crime')
        if pd.isna(crime) or crime == 'UNKNOWN':
            crime = 'mixed offenses'

        share = row.get('focus_share')
        if street:
            locale = f"around **{street}**"
        else:
            district = row.get('district_id')
            locale = f"in District {district}" if pd.notna(district) else "in this area"
        focus_txt = f" (selected crimes share {share:.0%})" if share not in (None, 0, np.nan) else ""
        lines.append(
            f"- {locale} we saw **{int(row['n_crimes'])} incidents**; most common offense: **{crime}**{focus_txt}."
        )
    return "\n".join(lines)


def format_month_range(months: Iterable[int]) -> str:
    months = sorted({int(m) for m in months})
    if not months:
        return ''
    if len(months) == 1:
        return calendar.month_name[months[0]]
    return f"{calendar.month_abbr[months[0]]}–{calendar.month_abbr[months[-1]]}"


def main():
    st.set_page_config(page_title='Chicago Crime Hotspots', layout='wide')
    st.title('Chicago Crime Hotspot Storytelling')

    st.sidebar.header('Explore the city')
    res = st.sidebar.radio('H3 resolution', AVAILABLE_RES, index=AVAILABLE_RES.index(9))

    years = sorted(int(p.name.split('=')[1]) for p in (L3_BASE / f'res={res}').glob('year=*') if p.is_dir())
    if not years:
        st.error('No L3 data found for the selected resolution.')
        st.stop()

    year = st.sidebar.selectbox('Year', years, index=len(years) - 1)
    months = sorted(int(p.name.split('=')[1]) for p in (L3_BASE / f'res={res}' / f'year={year}').glob('month=*') if p.is_dir())
    if not months:
        st.error('No months available for selected year/resolution.')
        st.stop()

    month_label = lambda m: calendar.month_abbr[m]
    default_months = [months[-1]]
    selected_months = st.sidebar.multiselect('Months', months, default=default_months, format_func=month_label)
    if not selected_months:
        selected_months = default_months
    months_tuple = tuple(sorted(selected_months))
    timeframe_label = format_month_range(months_tuple)

    try:
        l2_df = load_l2(year, months_tuple)
        l3_df = load_l3(year, months_tuple, res)
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()

    st.sidebar.markdown('---')
    st.sidebar.subheader('Crime filters')

    if 'primary_type' in l2_df.columns and not l2_df.empty:
        category_map = expand_categories(l2_df['primary_type'].dropna().unique())
    else:
        category_map = {}

    available_categories = sorted([c for c in category_map if c != 'Unclassified'])
    selected_categories = st.sidebar.multiselect('Categories', available_categories, default=[])

    base_types = sorted(l2_df['primary_type'].dropna().unique()) if 'primary_type' in l2_df.columns else []
    types_within_selected = sorted({t for cat in selected_categories for t in category_map.get(cat, [])})
    type_options = types_within_selected if selected_categories else base_types
    selected_types = st.sidebar.multiselect('Specific crime types', type_options, default=[])

    category_set = set(selected_categories)
    type_set = set(selected_types)
    if category_set and not type_set and types_within_selected:
        type_set = set(types_within_selected)

    if type_set:
        filtered_l2 = l2_df[l2_df['primary_type'].isin(type_set)].copy()
    elif category_set and 'crime_category' in l2_df.columns:
        filtered_l2 = l2_df[l2_df['crime_category'].isin(category_set)].copy()
        type_set = set(filtered_l2['primary_type'].dropna())
    else:
        filtered_l2 = l2_df.copy()

    focus_types = sorted(type_set)

    summary = summarise_period(l3_df, res)
    context_source = filtered_l2 if focus_types else l2_df
    context = build_context(context_source, res)
    share = focus_share(l2_df, res, focus_types)
    summary = summary.merge(context, on='h3_id', how='left').merge(share, on='h3_id', how='left')
    summary['focus_share'] = summary.get('focus_share', pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan)
    summary['focus_share'] = summary['focus_share'].fillna(0.0 if focus_types else np.nan)

    h3_col = f'h3_r{res}'
    if focus_types and not filtered_l2.empty and h3_col in filtered_l2.columns:
        selection_counts = (
            filtered_l2.dropna(subset=[h3_col])
            .groupby(h3_col)
            .size()
            .reset_index(name='selection_n_crimes')
            .rename(columns={h3_col: 'h3_id'})
        )
        summary = summary.merge(selection_counts, on='h3_id', how='left')
        summary['selection_n_crimes'] = summary['selection_n_crimes'].fillna(0).astype(int)
    else:
        summary['selection_n_crimes'] = summary['n_crimes'].fillna(0).astype(int)

    metric_options: Dict[str, Tuple[str, str]] = {'Incident volume': ('n_crimes', 'Incident volume')}
    if focus_types:
        metric_options['Selected incident volume'] = ('selection_n_crimes', 'Selected incident volume')
        metric_options['Selected share'] = ('focus_share', 'Selected crime share')
    if 'n_arrests' in summary.columns:
        metric_options['Arrests'] = ('n_arrests', 'Arrest count')
    if 'smoothed_rate' in summary.columns:
        metric_options['Smoothed arrest rate'] = ('smoothed_rate', 'Smoothed arrest rate')
    if 'pooled_smoothed' in summary.columns:
        metric_options['Neighbor smoothed rate'] = ('pooled_smoothed', 'Neighbor pooled rate')

    st.sidebar.markdown('---')
    st.sidebar.subheader('Map settings')
    metric_labels = list(metric_options.keys())
    default_metric_label = 'Selected incident volume' if focus_types and 'Selected incident volume' in metric_options else 'Incident volume'
    default_index = metric_labels.index(default_metric_label) if default_metric_label in metric_labels else 0
    chosen_metric_label = st.sidebar.selectbox('Color hexes by', metric_labels, index=default_index)
    color_metric, metric_label = metric_options[chosen_metric_label]
    if summary.empty or color_metric not in summary.columns:
        color_range = (0.0, 1.0)
    else:
        metric_series = summary[color_metric].fillna(0)
        if color_metric in {'focus_share', 'smoothed_rate', 'pooled_smoothed'}:
            upper = float(metric_series.max()) if metric_series.max() > 0 else 0.1
            color_range = (0.0, max(0.1, upper))
        else:
            upper = float(metric_series.max()) if metric_series.max() > 0 else 1.0
            color_range = (0.0, max(1.0, upper))

    st.sidebar.info(h3_story(res))

    summary_title = f'Hotspots (r{res}) – {timeframe_label} {year}'
    st.caption(f'Viewing **{timeframe_label} {year}** at resolution r{res}.')

    focus_counts = share[['h3_id', 'focus_count']] if 'focus_count' in share.columns else pd.DataFrame(columns=['h3_id', 'focus_count'])
    focus_counts = focus_counts.fillna({'focus_count': 0})

    if focus_types:
        total_incidents = int(filtered_l2.shape[0])
        busiest_focus = focus_counts.sort_values('focus_count', ascending=False).head(1)
        top_hex = int(busiest_focus['focus_count'].iloc[0]) if not busiest_focus.empty else 0
        avg_focus = summary['focus_share'].mean()
    else:
        total_incidents = int(summary['n_crimes'].sum()) if 'n_crimes' in summary.columns else 0
        top_value = summary['n_crimes'].max() if not summary.empty else 0
        top_hex = int(top_value) if not pd.isna(top_value) else 0
        avg_focus = np.nan

    working_l2 = filtered_l2 if not filtered_l2.empty else l2_df
    if 'datetime' in working_l2.columns:
        working_l2 = working_l2.assign(date=working_l2['datetime'].dt.floor('D'))
        daily = working_l2.groupby('date').size().reset_index(name='n_crimes')
        if 'arrest_made' in working_l2.columns:
            arrests_daily = working_l2.groupby('date')['arrest_made'].apply(lambda s: s.eq(True).sum()).reset_index(name='arrests')
            daily = daily.merge(arrests_daily, on='date', how='left').fillna({'arrests': 0})
        else:
            daily['arrests'] = 0
    else:
        daily = l3_df.groupby('date', as_index=False)['n_crimes'].sum()
        daily['arrests'] = 0

    early_mean = daily['n_crimes'].head(7).mean() if len(daily) >= 7 else daily['n_crimes'].mean()
    late_mean = daily['n_crimes'].tail(7).mean() if len(daily) >= 7 else daily['n_crimes'].mean()
    trend_delta = late_mean - early_mean

    arrest_early = daily['arrests'].head(7).mean() if len(daily) >= 7 else daily['arrests'].mean()
    arrest_late = daily['arrests'].tail(7).mean() if len(daily) >= 7 else daily['arrests'].mean()
    arrest_delta = arrest_late - arrest_early
    total_arrests = int(daily['arrests'].sum())

    if focus_types and not focus_counts.empty:
        leading_hex = focus_counts.sort_values('focus_count', ascending=False).head(1)
        target_id = leading_hex['h3_id'].iloc[0]
        focus_row = summary.loc[summary['h3_id'] == target_id, 'focus_share']
        leading_focus = float(focus_row.iloc[0]) if not focus_row.empty else np.nan
        leading_delta = leading_focus - summary['focus_share'].mean() if not np.isnan(leading_focus) else np.nan
    else:
        leading_hex = summary.sort_values('n_crimes', ascending=False).head(1)
        leading_focus = np.nan
        leading_delta = np.nan

    total_label = 'Selected incidents' if focus_types else 'Total incidents'
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(total_label, f'{total_incidents:,}', delta=f"{trend_delta:+.1f} vs 1st week avg", delta_color='inverse')
    focus_delta_text = f"Focus {leading_focus:.0%}" if focus_types and not np.isnan(leading_focus) else None
    busiest_label = 'Busiest hex incidents' if not focus_types else 'Busiest hex (selected)'
    if focus_types and focus_delta_text:
        m2.metric(busiest_label, f'{int(top_hex)}', delta=focus_delta_text, delta_color='inverse')
    else:
        m2.metric(busiest_label, f'{int(top_hex)}', delta=focus_delta_text)
    avg_focus_text = '--' if np.isnan(avg_focus) else f'{avg_focus:.0%}'
    delta_text = f"{leading_delta:+.0%}" if focus_types and not np.isnan(leading_delta) else None
    if focus_types and delta_text:
        m3.metric('Avg focus share', avg_focus_text, delta=delta_text, delta_color='inverse')
    else:
        m3.metric('Avg focus share', avg_focus_text, delta=delta_text)
    m4.metric('Total arrests', f'{total_arrests:,}', delta=f"{arrest_delta:+.1f} vs 1st week avg", delta_color='normal')

    trend_title = f"{'Selected ' if focus_types else ''}daily incidents – {timeframe_label} {year}"
    trend_fig = px.area(daily, x='date', y='n_crimes', title=trend_title, color_discrete_sequence=['#d73027'])
    trend_fig.update_traces(line_color='#a50026', fillcolor='rgba(215,48,31,0.35)')
    trend_fig.update_layout(yaxis_title='Incidents', xaxis_title='Date', height=360, template='plotly_white', hovermode='x unified')
    st.plotly_chart(trend_fig, use_container_width=True)

    map_tab, focus_tab = st.tabs(["Hotspot intensity", "Focus selection share"])
    with map_tab:
        map_fig = plot_hotspot_map(summary, res, summary_title, color_metric, metric_label, color_range)
        st.plotly_chart(map_fig, use_container_width=True)
        st.caption(f"Map key: greens stay calm, yellow = caution, deep reds = high incident volume. Resolution r{res} ≈ {RESOLUTION_GUIDE[res]['size']}.")
    with focus_tab:
        if focus_types and 'focus_share' in summary.columns:
            st.caption('Each hex shows the proportion of incidents that belong to the selected crime categories — useful to see concentration versus overall volume.')
            focus_fig = plot_focus_map(summary, res, f'Focus share – {timeframe_label} {year}')
            st.plotly_chart(focus_fig, use_container_width=True)
        else:
            st.info('Pick one or more categories or crime types to reveal the focus-share heatmap.')

    if not summary.empty:
        st.markdown('### Quick hits')
        st.markdown(narrative_bullets(summary))

    if 'datetime' in working_l2.columns:
        working_l2 = working_l2.copy()
        has_arrest = 'arrest_made' in working_l2.columns
        if has_arrest:
            working_l2['arrest_made'] = working_l2['arrest_made'].astype(bool)

        st.subheader('When do incidents spike?')
        hourly = (working_l2.assign(hour=working_l2['datetime'].dt.hour)
                              .groupby('hour')
                              .agg(incidents=('datetime', 'size'))
                              .reset_index())
        if has_arrest:
            arrests_by_hour = (working_l2.assign(hour=working_l2['datetime'].dt.hour)
                                           .groupby('hour')['arrest_made']
                                           .apply(lambda s: s.eq(True).sum())
                                           .reset_index(name='arrests'))
            hourly = hourly.merge(arrests_by_hour, on='hour', how='left').fillna({'arrests': 0})
        hourly_fig = px.line(hourly, x='hour', y='incidents', markers=True,
                             title='Incidents by hour of day', color_discrete_sequence=['#ff7f00'])
        hourly_fig.update_layout(template='plotly_white', xaxis_title='Hour of day', yaxis_title='Incidents')
        st.plotly_chart(hourly_fig, use_container_width=True)

        dow = (working_l2.assign(dow=working_l2['datetime'].dt.day_name())
                           .groupby('dow')
                           .agg(incidents=('datetime', 'size'))
                           .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
                           .reset_index())
        if has_arrest:
            dow_arrests = (working_l2.assign(dow=working_l2['datetime'].dt.day_name())
                                      .groupby('dow')['arrest_made']
                                      .apply(lambda s: s.eq(True).sum())
                                      .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
                                      .reset_index(name='arrests'))
            dow = dow.merge(dow_arrests, on='dow', how='left').fillna({'arrests': 0})
        dow_fig = px.bar(dow, x='dow', y='incidents', title='Incidents by day of week',
                         color='incidents', color_continuous_scale=['#fee0d2', '#de2d26'])
        dow_fig.update_layout(template='plotly_white', xaxis_title='Day', yaxis_title='Incidents', showlegend=False)
        st.plotly_chart(dow_fig, use_container_width=True)

    st.subheader('Signature streets per hotspot')
    street_cols = [col for col in ['signature_block', 'n_crimes', 'selection_n_crimes', 'common_crime', 'district_id', 'ward_id', 'focus_share'] if col in summary.columns]
    if street_cols:
        streets = (
            summary[street_cols]
            .sort_values('selection_n_crimes' if focus_types else 'n_crimes', ascending=False)
            .head(20)
            .fillna({'focus_share': 0, 'common_crime': 'UNKNOWN', 'signature_block': 'UNKNOWN'})
        )
        if 'focus_share' in streets.columns:
            streets['focus_share'] = streets['focus_share'].apply(lambda x: f"{x:.0%}" if isinstance(x, (int, float)) and not np.isnan(x) else '—')
        rename_map = {
            'signature_block': 'Block / anchor',
            'n_crimes': 'Incidents',
            'selection_n_crimes': 'Selected incidents',
            'common_crime': 'Common offense',
            'district_id': 'District',
            'ward_id': 'Ward',
            'focus_share': 'Focus share'
        }
        st.dataframe(streets.rename(columns=rename_map))

    with st.expander('Optional: recompute clustering overlay (UMAP + HDBSCAN)'):
        run_clusters = st.checkbox('Rebuild clusters for this slice')
        if run_clusters:
            try:
                from l3_clustering_prototype import run_clustering
                for month in months_tuple:
                    run_clustering(year, month, res=res)
                st.success('Cluster run complete. Check data/l3/clusters/ for outputs.')
            except Exception as exc:
                st.error(f'Clustering failed: {exc}\nInstall dependencies: pip install umap-learn hdbscan')


if __name__ == '__main__':
    main()
