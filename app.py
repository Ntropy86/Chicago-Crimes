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
MIN_RES = 6
MAX_RES = 10
AVAILABLE_RES = list(range(MIN_RES, MAX_RES + 1))
CHICAGO_CENTER = {"lat": 41.881832, "lon": -87.623177}

RESOLUTION_ZOOM = {
    6: 9.0,
    7: 10.2,
    8: 11.4,
    9: 12.8,
    10: 14.0,
}

RESOLUTION_GUIDE: Dict[int, Dict[str, str]] = {
    6: {"label": "Citywide", "size": "≈4.7 km across", "story": "Best for executive briefings and spotting macro trends."},
    7: {"label": "District-scale", "size": "≈1.2 km across", "story": "Best for citywide comparisons and strategic deployment."},
    8: {"label": "Neighborhood-scale", "size": "≈0.46 km across", "story": "Balances coverage with localized patterns – great for beat discussions."},
    9: {"label": "Street-scale", "size": "≈0.17 km across", "story": "Highlights specific blocks/intersections for tactical response."},
    10: {"label": "Micro-block", "size": "≈0.06 km across", "story": "Extremely granular – aggregate multiple months for stability."},
}


warnings.filterwarnings('ignore', message='.*choropleth_mapbox.*', category=DeprecationWarning)

CRIME_SCALE = ['#9bd174', '#6bbf59', '#f5e663', '#f9a64c', '#ef6248', '#b0202f']
ARREST_SCALE = list(reversed(CRIME_SCALE))

PLOTLY_CONFIG = {
    'scrollZoom': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}


def load_parent_child_mapping(parent_res: int, child_res: int) -> pd.DataFrame:
    path = DATA_DIR / 'h3_mappings' / f'parents_res_{parent_res}_to_{child_res}.parquet'
    if not path.exists():
        return pd.DataFrame(columns=['parent', 'child'])
    return pd.read_parquet(path)


def describe_hex_area(l2_df: pd.DataFrame, hex_id: str, res: int) -> str:
    col = f'h3_r{res}'
    guide_label = RESOLUTION_GUIDE.get(res, {'label': f'r{res}'}).get('label', f'r{res}')
    fallback = f"{guide_label} hotspot"
    if col not in l2_df.columns:
        return fallback
    subset = l2_df[l2_df[col] == hex_id]
    if subset.empty:
        return fallback

    def format_block(raw_block: str | None) -> str | None:
        if not raw_block or raw_block == 'UNKNOWN':
            return None
        clean = str(raw_block).strip().title()
        if clean.upper().startswith('0X0X0X'):  # guard random placeholders
            return None
        return clean

    def mode(series, replacements=None):
        s = series.dropna()
        if replacements:
            s = s.replace(replacements)
        s = s[s != 'UNKNOWN']
        if s.empty:
            return None
        return s.mode().iat[0]

    if res >= 9:
        block_val = None
        if 'signature_block' in subset.columns:
            block_val = mode(subset['signature_block'])
        if not block_val and 'block_address' in subset.columns:
            block_val = mode(subset['block_address'])
        pretty_block = format_block(block_val)
        if pretty_block:
            return pretty_block

    if res >= 9 and 'street_norm' in subset.columns:
        street = mode(subset['street_norm'], replacements={'': None})
        if street:
            return street.title()

    if 'district_id' in subset.columns:
        district = mode(subset['district_id'])
        if pd.notna(district):
            return f"District {int(district):02d}"

    if 'community_area_id' in subset.columns:
        community = mode(subset['community_area_id'])
        if pd.notna(community):
            return f"Community Area {int(community):02d}"

    if 'ward_id' in subset.columns:
        ward = mode(subset['ward_id'])
        if pd.notna(ward):
            return f"Ward {int(ward):02d}"

    return fallback


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
        'district_id', 'ward_id', 'h3_r6', 'h3_r7', 'h3_r8', 'h3_r9', 'h3_r10'
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


def plot_hotspot_map(
    summary: pd.DataFrame,
    res: int,
    title: str,
    color_metric: str,
    metric_label: str,
    color_range: Tuple[float, float],
    color_scale,
    map_center: Dict[str, float] | None = None,
    map_zoom: float | None = None,
):
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
        color_continuous_scale=color_scale,
        range_color=(color_min, color_max),
        featureidkey='properties.h3_id',
        title=title,
        mapbox_style='carto-positron',
        center=map_center or CHICAGO_CENTER,
        zoom=map_zoom or RESOLUTION_ZOOM.get(res, 11.0),
        opacity=0.45,
    )
    hover_data = display[['h3_id', 'display_street', 'display_crime', 'n_crimes', 'display_arrests', 'display_rate', 'display_low_conf', 'display_focus']].to_numpy()
    fig.update_traces(
        marker_line_width=0.6,
        marker_line_color='rgba(255,255,255,0.45)',
        customdata=hover_data,
        hovertemplate=(
            '<b>%{customdata[1]}</b><br>'
            'Hex: %{customdata[0]}<br>'
            'Top offense: %{customdata[2]}<br>'
            'Incidents: %{customdata[3]:,.0f}<br>'
            'Arrests: %{customdata[4]:,.0f}<br>'
            'Smoothed arrest rate: %{customdata[5]:.0%}<br>'
            'Low-confidence share: %{customdata[6]:.0%}<br>'
            'Focus share: %{customdata[7]:.0%}<extra></extra>'
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
    return f"{guide['label']} (r{res}) – {guide['size']}. {guide['story']}"


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

def metric_palette(metric: str) -> Tuple[list, str, str]:
    arrest_metrics = {'n_arrests', 'smoothed_rate', 'pooled_smoothed'}
    if metric in arrest_metrics:
        return ARREST_SCALE, 'fewer arrests / lower clearance', 'more arrests / higher clearance'
    return CRIME_SCALE, 'lower incident intensity', 'higher incident intensity'


def main():
    st.set_page_config(page_title='Chicago Crime Hotspots', layout='wide')
    st.markdown('## Chicago Crime Hotspot Console')
    st.caption('Interactive situational awareness for Chicago incidents, arrests, and hotspot context.')

    if 'drill_stack' not in st.session_state:
        st.session_state['drill_stack'] = []
    if 'base_signature' not in st.session_state:
        st.session_state['base_signature'] = None
    if 'base_res_choice' not in st.session_state:
        st.session_state['base_res_choice'] = 7
    if 'selected_dow' not in st.session_state:
        st.session_state['selected_dow'] = None
    if 'selected_dow' not in st.session_state:
        st.session_state['selected_dow'] = None
    pending_res = st.session_state.pop('base_res_pending', None)
    if pending_res is not None:
        pending_res = int(np.clip(pending_res, MIN_RES, MAX_RES))
        st.session_state['base_res_choice'] = pending_res

    st.sidebar.header('Explore the city')
    st.sidebar.subheader('Map settings')
    st.sidebar.selectbox(
        'Hex resolution',
        AVAILABLE_RES,
        index=AVAILABLE_RES.index(st.session_state['base_res_choice']),
        key='base_res_choice',
    )
    base_res = st.session_state['base_res_choice']

    years = sorted(int(p.name.split('=')[1]) for p in (L3_BASE / f'res={base_res}').glob('year=*') if p.is_dir())
    if not years:
        st.error('No L3 data found for the selected resolution.')
        st.stop()

    year = st.sidebar.selectbox('Year', years, index=len(years) - 1)
    months = sorted(int(p.name.split('=')[1]) for p in (L3_BASE / f'res={base_res}' / f'year={year}').glob('month=*') if p.is_dir())
    if not months:
        st.error('No months available for selected year/resolution.')
        st.stop()

    month_label = lambda m: calendar.month_abbr[m]
    default_months = [months[-1]]
    months_key = f"months-r{base_res}-{year}"
    selected_months = st.sidebar.multiselect('Months', months, default=default_months, format_func=month_label, key=months_key)
    if not selected_months:
        selected_months = default_months
    months_tuple = tuple(sorted(selected_months))
    timeframe_label = format_month_range(months_tuple)

    signature = (base_res, year, months_tuple)
    if st.session_state.get('base_signature') != signature:
        st.session_state['base_signature'] = signature
        st.session_state['drill_stack'] = []
        st.session_state['selected_dow'] = None

    try:
        l2_df = load_l2(year, months_tuple)
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
    categories_key = f"categories-r{base_res}-{year}-{','.join(map(str, months_tuple))}"
    selected_categories = st.sidebar.multiselect('Categories', available_categories, default=[], key=categories_key)

    base_types = sorted(l2_df['primary_type'].dropna().unique()) if 'primary_type' in l2_df.columns else []
    types_within_selected = sorted({t for cat in selected_categories for t in category_map.get(cat, [])})
    type_options = types_within_selected if selected_categories else base_types
    type_key = f"types-r{base_res}-{year}-{','.join(sorted(selected_categories)) or 'all'}"
    default_type_selection = type_options if selected_categories else []
    selected_types = st.sidebar.multiselect('Specific crime types', type_options, default=default_type_selection, key=type_key)
    if selected_categories and not type_options:
        st.sidebar.info('No mapped crime types for the selected categories yet.')

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

    drill_stack = st.session_state['drill_stack']
    if drill_stack:
        current_view = drill_stack[-1]
        effective_res = current_view['res']
        focus_hexes = set(current_view['child_ids'])
        st.sidebar.markdown('---')
        st.sidebar.subheader('Drill-down focus')
        parent_res_label = RESOLUTION_GUIDE.get(current_view['parent_res'], {'label': f"r{current_view['parent_res']}"})['label']
        child_res_label = RESOLUTION_GUIDE.get(current_view['res'], {'label': f"r{current_view['res']}"})['label']
        st.sidebar.write(f"Parent: **{current_view['label']}** ({parent_res_label}) → {child_res_label}")
        if st.sidebar.button('◀ Step back', key='drill-back'):
            drill_stack.pop()
            st.rerun()
        if st.sidebar.button('⟲ Reset view', key='drill-reset'):
            st.session_state['drill_stack'] = []
            st.rerun()
    else:
        effective_res = base_res
        focus_hexes = None

    try:
        l3_df = load_l3(year, months_tuple, effective_res)
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()

    summary = summarise_period(l3_df, effective_res)
    if focus_hexes is not None:
        summary = summary[summary['h3_id'].isin(focus_hexes)].copy()

    context_source = filtered_l2 if focus_types else l2_df
    context = build_context(context_source, effective_res)
    share = focus_share(l2_df, effective_res, focus_types)
    if focus_hexes is not None:
        share = share[share['h3_id'].isin(focus_hexes)]
    summary = summary.merge(context, on='h3_id', how='left').merge(share, on='h3_id', how='left')
    summary['focus_share'] = summary.get('focus_share', pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan)
    summary['focus_share'] = summary['focus_share'].fillna(0.0 if focus_types else np.nan)

    res_info = RESOLUTION_GUIDE.get(effective_res, {'label': f'r{effective_res}', 'size': ''})
    res_label = res_info['label']
    resolution_display = f"{res_label} (r{effective_res})"

    if 'n_crimes' in summary.columns and not summary.empty:
        density_ratio = float((summary['n_crimes'] >= 5).mean())
    else:
        density_ratio = 0.0
    if effective_res >= 10 and density_ratio < 0.2:
        st.warning(f'Only {density_ratio:.0%} of {resolution_display} hexes contain 5+ incidents. Aggregate multiple months or step back to a coarser resolution for clearer patterns.')

    h3_col_effective = f'h3_r{effective_res}'
    if focus_types and not filtered_l2.empty and h3_col_effective in filtered_l2.columns:
        selection_counts = (
            filtered_l2.dropna(subset=[h3_col_effective])
            .groupby(h3_col_effective)
            .size()
            .reset_index(name='selection_n_crimes')
            .rename(columns={h3_col_effective: 'h3_id'})
        )
        summary = summary.merge(selection_counts, on='h3_id', how='left')
        summary['selection_n_crimes'] = summary['selection_n_crimes'].fillna(0).astype(int)
    else:
        summary['selection_n_crimes'] = summary.get('n_crimes', pd.Series(dtype=int)).fillna(0).astype(int)

    current_l2 = filtered_l2 if not filtered_l2.empty else l2_df
    if focus_hexes is not None and h3_col_effective in current_l2.columns:
        current_l2 = current_l2[current_l2[h3_col_effective].isin(focus_hexes)]
        if current_l2.empty and not summary.empty:
            # Fallback to original slice so KPIs don't break when child hexes lack raw L2 rows
            parent_col = f'h3_r{drill_stack[-1]["parent_res"]}' if drill_stack else None
            if parent_col and parent_col in l2_df.columns:
                parent_hex = drill_stack[-1]['parent_hex']
                current_l2 = l2_df[l2_df[parent_col] == parent_hex].copy()
            if current_l2.empty:
                current_l2 = l2_df.copy()
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

    metric_labels = list(metric_options.keys())
    default_metric_label = 'Selected incident volume' if focus_types and 'Selected incident volume' in metric_options else 'Incident volume'
    default_index = metric_labels.index(default_metric_label) if default_metric_label in metric_labels else 0
    chosen_metric_label = st.sidebar.selectbox('Color hexes by', metric_labels, index=default_index)
    color_metric, metric_label = metric_options[chosen_metric_label]

    if summary.empty or color_metric not in summary.columns:
        color_range = (0.0, 1.0)
    else:
        metric_series = summary[color_metric].fillna(0)
        upper = float(metric_series.max())
        if color_metric in {'focus_share', 'smoothed_rate', 'pooled_smoothed', 'n_arrests'}:
            upper = max(upper, 0.1)
            color_range = (0.0, upper)
        else:
            upper = max(upper, 1.0)
            color_range = (0.0, upper)

    color_scale, legend_low, legend_high = metric_palette(color_metric)

    st.sidebar.caption(h3_story(effective_res))

    if drill_stack:
        focus_name = drill_stack[-1]['label']
    else:
        focus_name = 'Citywide'

    scope_label = f"{res_label} · {focus_name}"
    view_title = scope_label
    timeframe_caption = f"{timeframe_label} {year}".strip()
    summary_title = f"{scope_label} – {timeframe_caption}"

    focus_counts = share[['h3_id', 'focus_count']] if 'focus_count' in share.columns else pd.DataFrame(columns=['h3_id', 'focus_count'])
    focus_counts = focus_counts.fillna({'focus_count': 0})

    map_summary = summary
    if effective_res > 7 and focus_hexes is None and not summary.empty:
        map_summary = summary.sort_values('n_crimes', ascending=False).head(600).copy()
    summary_view = map_summary
    scope_hex_ids: set[str] = set(summary_view['h3_id']) if not summary_view.empty else set()
    if not focus_counts.empty and scope_hex_ids:
        focus_counts = focus_counts[focus_counts['h3_id'].isin(scope_hex_ids)]

    if focus_types:
        total_incidents = int(current_l2.shape[0])
        busiest_focus = focus_counts.sort_values('focus_count', ascending=False).head(1)
        top_hex = int(busiest_focus['focus_count'].iloc[0]) if not busiest_focus.empty else 0
        avg_focus = summary_view['focus_share'].mean()
    else:
        total_incidents = int(current_l2.shape[0]) if not current_l2.empty else 0
        top_value = summary_view['n_crimes'].max() if not summary_view.empty else 0
        top_hex = int(top_value) if not pd.isna(top_value) else 0
        avg_focus = np.nan

    if 'datetime' in current_l2.columns:
        current_l2 = current_l2.assign(date=current_l2['datetime'].dt.floor('D'))
        daily = current_l2.groupby('date').size().reset_index(name='n_crimes')
        if 'arrest_made' in current_l2.columns:
            arrests_daily = current_l2.groupby('date')['arrest_made'].apply(lambda s: s.eq(True).sum()).reset_index(name='arrests')
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
        focus_row = summary_view.loc[summary_view['h3_id'] == target_id, 'focus_share']
        leading_focus = float(focus_row.iloc[0]) if not focus_row.empty else np.nan
        leading_delta = leading_focus - summary_view['focus_share'].mean() if not np.isnan(leading_focus) else np.nan
    else:
        leading_hex = summary_view.sort_values('n_crimes', ascending=False).head(1)
        leading_focus = np.nan
        leading_delta = np.nan

    total_label = 'Selected incidents' if focus_types else 'Total incidents'
    focus_delta_text = f"Focus {leading_focus:.0%}" if focus_types and not np.isnan(leading_focus) else None
    busiest_label = 'Busiest hex incidents' if not focus_types else 'Busiest hex (selected)'
    avg_focus_text = '--' if np.isnan(avg_focus) else f'{avg_focus:.0%}'
    delta_text = f"{leading_delta:+.0%}" if focus_types and not np.isnan(leading_delta) else None
    trend_delta_text = None if pd.isna(trend_delta) else f"{trend_delta:+.1f} vs 1st week avg"
    arrest_delta_text = None if pd.isna(arrest_delta) else f"{arrest_delta:+.1f} vs 1st week avg"

    kpis = [
        {'label': total_label, 'value': f'{total_incidents:,}', 'delta': trend_delta_text, 'delta_color': 'inverse'},
        {'label': busiest_label, 'value': f'{int(top_hex)}', 'delta': focus_delta_text, 'delta_color': 'inverse' if focus_delta_text else None},
        {'label': 'Avg focus share', 'value': avg_focus_text, 'delta': delta_text, 'delta_color': 'inverse' if delta_text else None},
        {'label': 'Total arrests', 'value': f'{total_arrests:,}', 'delta': arrest_delta_text, 'delta_color': 'normal'},
    ]

    map_zoom = RESOLUTION_ZOOM.get(effective_res, 11.0)
    map_center = dict(CHICAGO_CENTER)
    if drill_stack:
        parent_lat, parent_lon = h3.cell_to_latlng(drill_stack[-1]['parent_hex'])
        map_center = {'lat': parent_lat, 'lon': parent_lon}
    if focus_hexes is not None and not map_summary.empty:
        latitudes = map_summary['lat'].dropna()
        longitudes = map_summary['lon'].dropna()
        if not latitudes.empty and not longitudes.empty:
            map_center = {'lat': float(latitudes.mean()), 'lon': float(longitudes.mean())}
            map_zoom = min(RESOLUTION_ZOOM.get(effective_res, map_zoom) + 0.6, 14.5)

    st.markdown('## Spatial Hotspot Explorer')
    map_col, metrics_col = st.columns([2.8, 1.2], gap='large')

    with map_col:
        map_col.markdown('### Map & Drill Controls')
        map_col.caption(f"Viewing {timeframe_caption} at {resolution_display}.")
        ctrl_prev, ctrl_mid, ctrl_next = st.columns([1.1, 1.8, 1.1])
        prev_disabled = base_res <= MIN_RES
        prev_label = RESOLUTION_GUIDE.get(base_res - 1, {'label': f'r{base_res - 1}'})['label']
        if ctrl_prev.button(f"◀ {prev_label}", disabled=prev_disabled, use_container_width=True):
            st.session_state['base_res_pending'] = base_res - 1
            st.session_state['drill_stack'] = []
            st.session_state['selected_dow'] = None
            st.rerun()

        ctrl_mid.markdown(f"**{view_title}**")
        ctrl_mid.caption(f"{timeframe_caption} · {resolution_display}")

        next_disabled = base_res >= MAX_RES
        next_label = RESOLUTION_GUIDE.get(base_res + 1, {'label': f'r{base_res + 1}'})['label']
        if ctrl_next.button(f"{next_label} ▶", disabled=next_disabled, use_container_width=True):
            st.session_state['base_res_pending'] = base_res + 1
            st.session_state['drill_stack'] = []
            st.session_state['selected_dow'] = None
            st.rerun()

        if drill_stack:
            parent_view = drill_stack[-1]
            parent_res_label = RESOLUTION_GUIDE.get(parent_view['parent_res'], {'label': f"r{parent_view['parent_res']}"})['label']
            parent_label = parent_view.get('label', parent_res_label)
            if st.button(
                f"⬆ Back to {parent_res_label} – {parent_label}",
                key='map-drill-back',
                use_container_width=True,
            ):
                drill_stack.pop()
                st.session_state['drill_stack'] = drill_stack
                st.session_state['selected_dow'] = None
                st.rerun()

        tab_labels = ["Hotspot intensity", "Focus selection share"]
        if drill_stack:
            tab_labels.append("Breadcrumb")
        map_tab, focus_tab, *extra_tabs = st.tabs(tab_labels)
        with map_tab:
            map_fig = plot_hotspot_map(
                map_summary,
                effective_res,
                summary_title,
                color_metric,
                metric_label,
                color_range,
                color_scale,
                map_center=map_center,
                map_zoom=map_zoom,
            )
            event_key = f"hex-map-r{effective_res}-{year}-{','.join(map(str, months_tuple))}"
            selection_state = st.plotly_chart(
                map_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key=event_key,
                on_select='rerun',
                selection_mode=('points',),
            )
            st.caption(
                f"Legend: green → {legend_low}; red → {legend_high}. Resolution {resolution_display} ≈ {RESOLUTION_GUIDE[effective_res]['size']}."
            )
            selected_hex = None
            if isinstance(selection_state, dict):
                points = selection_state.get('selection', {}).get('points', [])
                if points:
                    point_payload = points[0]
                    location = point_payload.get('location')
                    if location:
                        selected_hex = location
                    else:
                        customdata = point_payload.get('customdata')
                        if isinstance(customdata, (list, tuple)) and customdata:
                            selected_hex = customdata[0]
            parent_res = effective_res
            if selected_hex:
                if parent_res >= MAX_RES:
                    st.info('Already viewing the finest resolution available.')
                else:
                    mapping_df = load_parent_child_mapping(parent_res, parent_res + 1)
                    child_ids = mapping_df[mapping_df['parent'] == selected_hex]['child'].tolist()
                    if not child_ids:
                        next_label = RESOLUTION_GUIDE.get(parent_res + 1, {'label': f'r{parent_res + 1}'})['label']
                        st.warning(f'No child hexes available for {selected_hex} at {next_label}.')
                    else:
                        label = describe_hex_area(l2_df, selected_hex, parent_res)
                        drill_stack.append({
                            'parent_res': parent_res,
                            'parent_hex': selected_hex,
                            'res': parent_res + 1,
                            'child_ids': child_ids,
                            'label': label,
                        })
                        st.session_state['drill_stack'] = drill_stack
                        st.session_state['selected_dow'] = None
                        st.rerun()
        with focus_tab:
            if focus_types and 'focus_share' in summary.columns:
                st.caption('Each hex shows the proportion of incidents that belong to the selected crime categories — useful to see concentration versus overall volume.')
                focus_fig = plot_focus_map(map_summary, effective_res, f'Focus share – {timeframe_label} {year}')
                st.plotly_chart(focus_fig, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.info('Pick one or more categories or crime types to reveal the focus-share heatmap.')

    with metrics_col:
        metrics_col.markdown(f"### Incident Snapshot — {scope_label}")
        metrics_col.caption(f'{timeframe_caption} focus across the selected filters.')
        
        for item in kpis:
            kwargs = {}
            if item.get('delta') is not None:
                kwargs['delta'] = item['delta']
                if item.get('delta_color'):
                    kwargs['delta_color'] = item['delta_color']
            # each KPI gets the full width of metrics_col
            metrics_col.metric(item['label'], item['value'], **kwargs)


    working_l2 = current_l2.copy() if isinstance(current_l2, pd.DataFrame) else pd.DataFrame()
    has_datetime = 'datetime' in working_l2.columns
    has_arrest_flag = 'arrest_made' in working_l2.columns
    if has_arrest_flag:
        working_l2['arrest_made'] = working_l2['arrest_made'].astype(bool)

    if drill_stack and extra_tabs:
        breadcrumb_tab = extra_tabs[0]
        with breadcrumb_tab:
            st.markdown('### Drill history')
            for level, entry in enumerate(drill_stack, start=1):
                parent_label = RESOLUTION_GUIDE.get(entry['parent_res'], {'label': f"r{entry['parent_res']}"})['label']
                child_label = RESOLUTION_GUIDE.get(entry['res'], {'label': f"r{entry['res']}"})['label']
                parent_area = describe_hex_area(l2_df, entry['parent_hex'], entry['parent_res'])
                st.write(f"{level}. {parent_label} → {child_label} | {parent_area}")

    if not summary.empty:
        st.markdown('## Narrative & Composition')
        story_col, mix_col = st.columns([1.4, 1])
        with story_col:
            st.markdown('### Quick hits')
            st.caption('Headline talking points for the selected hotspots and focus area.')
            st.markdown(narrative_bullets(summary_view))
        with mix_col:
            st.markdown('### Crime mix (top 10)')
            st.caption('Dominant offense types for the filtered incidents.')
            if not working_l2.empty and 'primary_type' in working_l2.columns:
                crime_mix = (
                    working_l2['primary_type']
                    .dropna()
                    .value_counts()
                    .head(10)
                    .rename_axis('primary_type')
                    .reset_index(name='incident_count')
                )
                if not crime_mix.empty:
                    palette = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666', '#8dd3c7', '#fb8072']
                    color_map = {ptype: palette[i % len(palette)] for i, ptype in enumerate(crime_mix['primary_type'])}
                    crime_fig = px.bar(
                        crime_mix,
                        x='incident_count',
                        y='primary_type',
                        orientation='h',
                        title=f'Top crime categories — {scope_label}',
                        color='primary_type',
                        color_discrete_map=color_map,
                    )
                    crime_fig.update_layout(
                        yaxis_title='',
                        xaxis_title='Incidents',
                        template='plotly_white',
                        legend_title='Crime type',
                    )
                    st.plotly_chart(crime_fig, use_container_width=True, config=PLOTLY_CONFIG)
                    st.caption('Bar colours mirror the crime chips for this selection.')
                else:
                    st.info('No crime mix available for this selection.')
            else:
                st.info('Crime type metadata not available for this selection.')

    if has_datetime:
        st.markdown('## Temporal Patterns')
        st.caption('How incident and arrest activity evolves over the selected period.')
        st.subheader('Daily trend & arrest compare')
        trend_title = f"Daily incidents — {scope_label}"
        trend_fig = px.area(daily, x='date', y='n_crimes', title=trend_title, color_discrete_sequence=['#d73027'])
        trend_fig.update_traces(line_color='#a50026', fillcolor='rgba(215,48,31,0.35)')
        trend_fig.update_layout(yaxis_title='Incidents', xaxis_title='Date', height=360, template='plotly_white', hovermode='x unified')

        compare_cols = st.columns(2, gap='large')
        with compare_cols[0]:
            st.plotly_chart(trend_fig, use_container_width=True, config=PLOTLY_CONFIG)

        with compare_cols[1]:
            if has_arrest_flag:
                trend_compare = (
                    working_l2.assign(date=working_l2['datetime'].dt.floor('D'))
                    .groupby('date')
                    .agg(
                        incidents=('datetime', 'size'),
                        arrests=('arrest_made', lambda s: s.eq(True).sum()),
                    )
                    .reset_index()
                )
                line_colors = {'incidents': '#d73027', 'arrests': '#238b45'}
                melted = trend_compare.melt(id_vars='date', value_vars=['incidents', 'arrests'], var_name='metric', value_name='count')
                compare_fig = px.line(
                    melted,
                    x='date',
                    y='count',
                    color='metric',
                    color_discrete_map=line_colors,
                    title=f'Incidents vs arrests — {scope_label}',
                )
                compare_fig.update_layout(
                    template='plotly_white',
                    hovermode='x unified',
                    yaxis_title='Count',
                    xaxis_title='Date',
                    legend_title='Metric',
                )
                st.plotly_chart(compare_fig, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.info('Arrest data not available for this selection.')

        st.subheader('When do incidents spike?')
        hourly = (
            working_l2.assign(hour=working_l2['datetime'].dt.hour)
            .groupby('hour')
            .agg(incidents=('datetime', 'size'))
            .reset_index()
        )
        if has_arrest_flag:
            arrests_by_hour = (
                working_l2.assign(hour=working_l2['datetime'].dt.hour)
                .groupby('hour')['arrest_made']
                .apply(lambda s: s.eq(True).sum())
                .reset_index(name='arrests')
            )
            hourly = hourly.merge(arrests_by_hour, on='hour', how='left').fillna({'arrests': 0})

        dow = (
            working_l2.assign(dow=working_l2['datetime'].dt.day_name())
            .groupby('dow')
            .agg(incidents=('datetime', 'size'))
            .reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            .reset_index()
        )
        if has_arrest_flag:
            dow_arrests = (
                working_l2.assign(dow=working_l2['datetime'].dt.day_name())
                .groupby('dow')['arrest_made']
                .apply(lambda s: s.eq(True).sum())
                .reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                .reset_index(name='arrests')
            )
            dow = dow.merge(dow_arrests, on='dow', how='left').fillna({'arrests': 0})

        highlight_dow = st.session_state.get('selected_dow')
        hourly_focus = hourly.copy()
        if highlight_dow and highlight_dow in dow['dow'].values:
            filtered = working_l2[working_l2['datetime'].dt.day_name() == highlight_dow]
            if not filtered.empty:
                hourly_focus = (
                    filtered.assign(hour=filtered['datetime'].dt.hour)
                    .groupby('hour')
                    .agg(incidents=('datetime', 'size'))
                    .reset_index()
                )
                if has_arrest_flag:
                    hourly_focus = hourly_focus.merge(
                        filtered.assign(hour=filtered['datetime'].dt.hour)
                        .groupby('hour')['arrest_made']
                        .apply(lambda s: s.eq(True).sum())
                        .reset_index(name='arrests'),
                        on='hour',
                        how='left'
                    ).fillna({'arrests': 0})
            else:
                highlight_dow = None

        hourly_title = 'Hourly incidents — all days'
        if highlight_dow:
            hourly_title = f'Hourly incidents — {highlight_dow} focus'

        hourly_fig = px.line(
            hourly_focus,
            x='hour',
            y='incidents',
            markers=True,
            title=hourly_title,
            color_discrete_sequence=['#ff7f00'],
        )
        hourly_fig.update_layout(template='plotly_white', xaxis_title='Hour of day', yaxis_title='Incidents')
        if has_arrest_flag and 'arrests' in hourly_focus.columns:
            hourly_fig.add_bar(
                x=hourly_focus['hour'],
                y=hourly_focus['arrests'],
                name='Arrests',
                marker_color='#74a9cf',
                opacity=0.45,
            )

        charts_row = st.columns(2, gap='large')

        with charts_row[0]:
            dow_title = f'Day-of-week intensity — {scope_label}'
            dow_fig = px.bar(
                dow,
                x='dow',
                y='incidents',
                title=dow_title,
                color='incidents',
                color_continuous_scale=['#fee0d2', '#de2d26'],
            )
            dow_fig.update_layout(template='plotly_white', xaxis_title='Day', yaxis_title='Incidents', showlegend=False)
            if highlight_dow:
                highlight_lines = ['#de2d26' if day == highlight_dow else '#bbbbbb' for day in dow['dow']]
                dow_fig.update_traces(marker_line_color=highlight_lines, marker_line_width=2)
            dow_selection = st.plotly_chart(
                dow_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key='dow-chart',
                on_select='rerun',
                selection_mode=('points',),
            )
            if isinstance(dow_selection, dict):
                points = dow_selection.get('selection', {}).get('points', [])
                if points:
                    point_index = points[0].get('point_index')
                    if point_index is not None and 0 <= point_index < len(dow):
                        st.session_state['selected_dow'] = dow.loc[point_index, 'dow']
                else:
                    st.session_state['selected_dow'] = None
            if st.session_state.get('selected_dow'):
                if st.button('Reset day-of-week filter', key='reset-dow', use_container_width=True):
                    st.session_state['selected_dow'] = None
                    st.rerun()

        with charts_row[1]:
            st.plotly_chart(hourly_fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader('Crimes vs arrests by category')
        if 'primary_type' in working_l2.columns:
            crime_incidents = (
                working_l2.groupby('primary_type')
                .size()
                .reset_index(name='incidents')
                .sort_values('incidents', ascending=False)
                .head(12)
            )
            if not crime_incidents.empty and has_arrest_flag:
                arrests = (
                    working_l2[working_l2['arrest_made']]
                    .groupby('primary_type')
                    .size()
                    .reset_index(name='arrests')
                )
                crime_arrests = crime_incidents.merge(arrests, on='primary_type', how='left').fillna({'arrests': 0})
                melted_ca = crime_arrests.melt(
                    id_vars='primary_type',
                    value_vars=['incidents', 'arrests'],
                    var_name='metric',
                    value_name='count',
                )
                ca_fig = px.bar(
                    melted_ca,
                    x='primary_type',
                    y='count',
                    color='metric',
                    barmode='group',
                    color_discrete_map={'incidents': '#d73027', 'arrests': '#1b7837'},
                    title=f'Incident vs arrest counts — {scope_label}',
                )
                ca_fig.update_layout(
                    template='plotly_white',
                    xaxis_title='Crime type',
                    yaxis_title='Count',
                    legend_title='Metric',
                    xaxis_tickangle=-35,
                )
                st.plotly_chart(ca_fig, use_container_width=True, config=PLOTLY_CONFIG)
            elif not crime_incidents.empty:
                ca_fig = px.bar(
                    crime_incidents,
                    x='primary_type',
                    y='incidents',
                    color='incidents',
                    title=f'Incident counts by category — {scope_label}',
                    color_continuous_scale=['#fee5d9', '#fcae91', '#fb6a4a', '#cb181d'],
                )
                ca_fig.update_layout(template='plotly_white', xaxis_tickangle=-35, yaxis_title='Incidents')
                st.plotly_chart(ca_fig, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.info('Not enough incident volume to summarise by category.')
        else:
            st.info('Crime categories unavailable for this selection.')

        if 'district_id' in working_l2.columns:
            district_counts = (
                working_l2.groupby('district_id')
                .size()
                .reset_index(name='incidents')
                .sort_values('incidents', ascending=False)
                .head(10)
            )
            if not district_counts.empty:
                district_counts['district_id'] = district_counts['district_id'].fillna(-1).astype(int).astype(str).replace({'-1': 'Unknown'})
                district_fig = px.bar(
                    district_counts,
                    x='incidents',
                    y='district_id',
                    orientation='h',
                    title=f'Districts with highest incident volume — {scope_label}',
                    color='incidents',
                    color_continuous_scale=['#edf8fb', '#66c2a4', '#006d2c'],
                )
                district_fig.update_layout(template='plotly_white', yaxis_title='District', xaxis_title='Incidents', showlegend=False)
                st.plotly_chart(district_fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.subheader('Signature streets per hotspot')
    street_cols = [col for col in ['signature_block', 'n_crimes', 'selection_n_crimes', 'common_crime', 'district_id', 'ward_id', 'focus_share'] if col in summary_view.columns]
    if street_cols:
        streets = (
            summary_view[street_cols]
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
                    run_clustering(year, month, res=base_res)
                st.success('Cluster run complete. Check data/l3/clusters/ for outputs.')
            except Exception as exc:
                st.error(f'Clustering failed: {exc}\nInstall dependencies: pip install umap-learn hdbscan')


if __name__ == '__main__':
    main()
