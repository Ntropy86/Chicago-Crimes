import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import h3
from pathlib import Path
import sys
import calendar
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
        'datetime', 'primary_type', 'street_norm', 'block_address', 'arrest_made', 'community_area_id',
        'district_id', 'ward_id', 'h3_r7', 'h3_r8', 'h3_r9'
    ]
    df = pd.read_parquet(path)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # Normalize column names when older exports use alternate labels
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
    df = df[available_cols].copy()
    if 'arrest_made' in df.columns:
        try:
            df['arrest_made'] = df['arrest_made'].astype('boolean').fillna(False)
        except Exception:
            df['arrest_made'] = df['arrest_made'].apply(lambda x: bool(x) if pd.notna(x) else False).astype('boolean')
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
        if context is None:
            context = frame
        else:
            context = context.merge(frame, on=h3_col, how='outer')
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
        # h3 returns (lat, lon); Mapbox expects (lon, lat)
        coords = [[lon, lat] for lat, lon in boundary]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append({
            'type': 'Feature',
            'id': row['h3_id'],
            'properties': {
                'h3_id': row['h3_id']
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [coords]
            }
        })
    return {'type': 'FeatureCollection', 'features': features}


def plot_hotspot_map(summary: pd.DataFrame, res: int, year: int, month: int):
    geojson = build_geojson(summary)
    display = summary.copy() if not summary.empty else pd.DataFrame({'h3_id': [], 'n_crimes': []})
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
    color_max = max(display['n_crimes'].max(), 1) if not display.empty else 1
    if summary.empty:
        return px.choropleth_mapbox()
    fig = px.choropleth_mapbox(
        display,
        geojson=geojson,
        locations='h3_id',
        color='n_crimes',
        color_continuous_scale=['#9bd174', '#6bbf59', '#f5e663', '#f9a64c', '#ef6248', '#b0202f'],
        range_color=(0, color_max),
        featureidkey='properties.h3_id',
        title=f'Hotspots (r{res}) – {year}-{month:02d}',
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
        coloraxis_colorbar=dict(title='Incident volume', ticksuffix='')
    )
    return fig


def plot_focus_map(summary: pd.DataFrame, res: int, year: int, month: int):
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
        title='Focus crime share across hexes',
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
            'Incidents (month): %{customdata[2]:,.0f}<br>'
            'Selected crime share: %{customdata[3]:.0%}<extra></extra>'
        )
    )
    fig.update_layout(
        margin={'r': 0, 't': 50, 'l': 0, 'b': 0},
        coloraxis_colorbar=dict(title='Selected crime share', ticksuffix='')
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
        street = row.get('signature_block') or row.get('signature_street') or 'unknown streets'
        crime = row.get('common_crime') or 'mixed offenses'
        share = row.get('focus_share')
        if street and street != 'UNKNOWN':
            locale = f"around **{street}**"
        else:
            district = row.get('district_id')
            locale = f"in District {district}" if pd.notna(district) else "in this area"
        focus_txt = f" (selected crimes share {share:.0%})" if share not in (None, 0, np.nan) else ""
        lines.append(
            f"- {locale} we saw **{int(row['n_crimes'])} incidents**; most common offense: **{crime}**{focus_txt}."
        )
    return "\n".join(lines)


def main():
    st.set_page_config(page_title='Chicago Crime Hotspots', layout='wide')
    st.markdown(
        """
        <style>
        .sticky-controls {
            position: sticky;
            top: 0;
            z-index: 100;
            background-color: #0e1117;
            padding-top: 0.75rem;
            padding-bottom: 0.85rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title('Chicago Crime Hotspot Storytelling')

    control_container = st.container()
    with control_container:
        st.markdown('<div class="sticky-controls">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1.1, 1.1, 1.4])
        with col1:
            res = st.radio('H3 resolution', AVAILABLE_RES, index=AVAILABLE_RES.index(9), horizontal=True)
        years = sorted(int(p.name.split('=')[1]) for p in (L3_BASE / f'res={res}').glob('year=*') if p.is_dir())
        if not years:
            st.error('No L3 data found for the selected resolution.')
            st.stop()
        with col2:
            year = st.select_slider('Year', options=years, value=years[-1])
        months = sorted(int(p.name.split('=')[1]) for p in (L3_BASE / f'res={res}' / f'year={year}').glob('month=*') if p.is_dir())
        if not months:
            st.error('No months available for selected year/resolution.')
            st.stop()
        month_label = lambda m: f"{calendar.month_abbr[m]}"
        with col3:
            month = st.select_slider('Month', options=months, value=months[-1], format_func=month_label)
        month_name = calendar.month_name[month]

        l2_df = load_l2(year, month)
        if 'primary_type' in l2_df.columns:
            primary_types = sorted([pt for pt in l2_df['primary_type'].dropna().unique()])
        else:
            primary_types = []

        focus_types = []
        if primary_types:
            try:
                selection = st.pills('Focus on crime categories', options=primary_types, selection_mode='multi', default=[], key='crime-pills')
                if selection is None:
                    focus_types = []
                elif isinstance(selection, (list, tuple, set)):
                    focus_types = list(selection)
                else:
                    focus_types = [selection]
            except TypeError:
                focus_types = st.multiselect('Focus on specific crime types (optional)', primary_types, default=[])
            except AttributeError:
                focus_types = st.multiselect('Focus on specific crime types (optional)', primary_types, default=[])
        else:
            st.info('Crime type metadata not available for this slice; focus map disabled.')
        st.markdown('</div>', unsafe_allow_html=True)

    st.info(h3_story(res))

    l3_df = load_l3(year, month, res)

    summary = summarise_month(l3_df, res)
    context = build_context(l2_df, res)
    share = focus_share(l2_df, res, focus_types)
    summary = summary.merge(context, on='h3_id', how='left').merge(share, on='h3_id', how='left')
    summary['focus_share'] = summary.get('focus_share', pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan)
    summary['focus_share'] = summary['focus_share'].fillna(0.0 if focus_types else np.nan)

    focus_counts = share[['h3_id', 'focus_count']] if 'focus_count' in share.columns else pd.DataFrame(columns=['h3_id', 'focus_count'])
    focus_counts = focus_counts.fillna({'focus_count': 0})

    if focus_types and 'primary_type' in l2_df.columns:
        filtered_l2 = l2_df[l2_df['primary_type'].isin(focus_types)].copy()
        total_incidents = int(filtered_l2.shape[0])
        busiest_focus = focus_counts.sort_values('focus_count', ascending=False).head(1)
        top_hex = int(busiest_focus['focus_count'].iloc[0]) if not busiest_focus.empty else 0
        avg_focus = summary['focus_share'].mean()
    else:
        filtered_l2 = l2_df
        total_incidents = int(summary['n_crimes'].sum())
        top_value = summary['n_crimes'].max() if not summary.empty else 0
        top_hex = int(top_value) if not pd.isna(top_value) else 0
        avg_focus = np.nan

    has_datetime = 'datetime' in filtered_l2.columns
    if has_datetime:
        filtered_l2 = filtered_l2.assign(date=filtered_l2['datetime'].dt.floor('D'))
        daily = filtered_l2.groupby('date').size().reset_index(name='n_crimes')
        if 'arrest_made' in filtered_l2.columns:
            filtered_l2['arrest_made'] = filtered_l2['arrest_made'].astype(bool)
            arrest_daily = filtered_l2.groupby('date')['arrest_made'].sum().reset_index(name='arrests')
            daily = daily.merge(arrest_daily, on='date', how='left').fillna({'arrests': 0})
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

    total_label = 'Selected incidents (month)' if focus_types else 'Total incidents (month)'
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
    m4.metric('Total arrests (month)', f'{total_arrests:,}', delta=f"{arrest_delta:+.1f} vs 1st week avg", delta_color='normal')

    trend_title = f"{'Selected ' if focus_types else ''}daily incidents – {month_name} {year}"
    trend_fig = px.area(
        daily,
        x='date', y='n_crimes',
        title=trend_title,
        color_discrete_sequence=['#d73027']
    )
    trend_fig.update_traces(line_color='#a50026', fillcolor='rgba(215,48,31,0.35)')
    trend_fig.update_layout(yaxis_title='Incidents', xaxis_title='Date', height=360,
                            template='plotly_white', hovermode='x unified')
    st.plotly_chart(trend_fig, use_container_width=True)

    map_tab, focus_tab = st.tabs(["Hotspot intensity", "Focus selection share"])
    with map_tab:
        map_fig = plot_hotspot_map(summary, res, year, month)
        st.plotly_chart(map_fig, use_container_width=True)
        st.caption(f"Map key: greens stay calm, yellow = caution, deep reds = high incident volume. Resolution r{res} ≈ {RESOLUTION_GUIDE[res]['size']}.")
    with focus_tab:
        if focus_types and 'focus_share' in summary.columns:
            st.caption('Each hex shows the proportion of incidents that belong to the selected crime categories — useful to see concentration versus overall volume.')
            focus_fig = plot_focus_map(summary, res, year, month)
            st.plotly_chart(focus_fig, use_container_width=True)
        else:
            st.info('Pick one or more crime categories above to reveal the focus-share heatmap.')

    if 'datetime' in filtered_l2.columns:
        has_arrest = 'arrest_made' in filtered_l2.columns
        if not has_arrest:
            filtered_l2 = filtered_l2.assign(arrest_made=False)
            has_arrest = True
        trend_compare = (filtered_l2.assign(date=filtered_l2['datetime'].dt.floor('D'))
                                      .groupby('date')
                                      .agg(incidents=('datetime', 'size'),
                                           arrests=('arrest_made', lambda s: s.eq(True).sum() if has_arrest else 0))
                                      .reset_index())
        line_colors = {'incidents': '#d73027', 'arrests': '#238b45'}
        melted = trend_compare.melt(id_vars='date', value_vars=['incidents', 'arrests'], var_name='metric', value_name='count')
        compare_fig = px.line(melted, x='date', y='count', color='metric', color_discrete_map=line_colors,
                              title='Incidents vs arrests by day')
        compare_fig.update_layout(template='plotly_white', hovermode='x unified', yaxis_title='Count', xaxis_title='Date', legend_title='Metric')
        st.plotly_chart(compare_fig, use_container_width=True)

    if 'district_id' in filtered_l2.columns:
        district_counts = (filtered_l2.groupby('district_id').size().reset_index(name='incidents')
                                              .sort_values('incidents', ascending=False).head(10))
        district_counts['district_id'] = district_counts['district_id'].fillna(-1).astype(int).astype(str).replace({'-1': 'Unknown'})
        district_fig = px.bar(district_counts, x='incidents', y='district_id', orientation='h',
                              title='Districts with highest incident volume',
                              color='incidents', color_continuous_scale=['#edf8fb', '#66c2a4', '#006d2c'])
        district_fig.update_layout(template='plotly_white', yaxis_title='District', xaxis_title='Incidents', showlegend=False)
        st.plotly_chart(district_fig, use_container_width=True)

    col_story, col_breakdown = st.columns([1, 1])
    with col_story:
        st.subheader(f'Key reports for {month_name} {year}')
        st.markdown(narrative_bullets(summary))
    with col_breakdown:
        st.subheader('Crime mix (top 10)')
        if 'primary_type' in l2_df.columns:
            crime_mix = (
                l2_df['primary_type']
                     .value_counts()
                     .head(10)
                     .rename_axis('primary_type')
                     .reset_index(name='incident_count')
            )
            palette = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666', '#8dd3c7', '#fb8072']
            color_map = {ptype: palette[i % len(palette)] for i, ptype in enumerate(crime_mix['primary_type'])}
            crime_fig = px.bar(crime_mix, x='incident_count', y='primary_type', orientation='h',
                               title='Top crime categories', color='primary_type',
                               color_discrete_map=color_map)
            crime_fig.update_layout(yaxis_title='', xaxis_title='Incidents', template='plotly_white', legend_title='Crime type')
            st.plotly_chart(crime_fig, use_container_width=True)
            st.caption('Each bar colour corresponds to a specific crime category; match colours with the chips above when reviewing focus-share.')
        else:
            st.write('Primary crime types unavailable for this month.')

    if 'primary_type' in filtered_l2.columns:
        st.subheader('Crimes vs arrests by category')
        if 'arrest_made' in filtered_l2.columns:
            crime_arrests = (filtered_l2.groupby('primary_type')
                                           .agg(incidents=('primary_type', 'size'),
                                                arrests=('arrest_made', lambda s: s.eq(True).sum()))
                                           .reset_index()
                                           .sort_values('incidents', ascending=False)
                                           .head(12))
            melted_ca = crime_arrests.melt(id_vars='primary_type', value_vars=['incidents', 'arrests'],
                                           var_name='metric', value_name='count')
            ca_fig = px.bar(melted_ca, x='primary_type', y='count', color='metric', barmode='group',
                            color_discrete_map={'incidents': '#d73027', 'arrests': '#1b7837'},
                            title='Incident vs arrest counts (top categories)')
            ca_fig.update_layout(template='plotly_white', xaxis_title='Crime type', yaxis_title='Count',
                                 legend_title='Metric', xaxis_tickangle=-35)
            st.plotly_chart(ca_fig, use_container_width=True)
        else:
            st.info('Arrest data not available for this month.')

    if 'datetime' in filtered_l2.columns:
        st.subheader('When do incidents spike?')
        hourly = (filtered_l2.assign(hour=filtered_l2['datetime'].dt.hour)
                              .groupby('hour')
                              .agg(incidents=('datetime', 'size'))
                              .reset_index())
        if 'arrest_made' in filtered_l2.columns:
            arrests_by_hour = (filtered_l2.assign(hour=filtered_l2['datetime'].dt.hour)
                                           .groupby('hour')['arrest_made']
                                           .apply(lambda s: s.eq(True).sum())
                                           .reset_index(name='arrests'))
            hourly = hourly.merge(arrests_by_hour, on='hour', how='left').fillna({'arrests': 0})
        hourly_fig = px.line(hourly, x='hour', y='incidents', markers=True,
                             title='Incidents by hour of day', color_discrete_sequence=['#ff7f00'])
        hourly_fig.update_layout(template='plotly_white', xaxis_title='Hour of day', yaxis_title='Incidents')
        st.plotly_chart(hourly_fig, use_container_width=True)

        dow = (filtered_l2.assign(dow=filtered_l2['datetime'].dt.day_name())
                           .groupby('dow')
                           .agg(incidents=('datetime', 'size'))
                           .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
                           .reset_index())
        if 'arrest_made' in filtered_l2.columns:
            dow_arrests = (filtered_l2.assign(dow=filtered_l2['datetime'].dt.day_name())
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
    street_cols = [col for col in ['signature_block', 'n_crimes', 'common_crime', 'district_id', 'ward_id', 'focus_share'] if col in summary.columns]
    streets = (
        summary[street_cols]
        .sort_values('n_crimes', ascending=False)
        .head(20)
        .fillna({'focus_share': 0, 'common_crime': 'UNKNOWN', 'signature_block': 'UNKNOWN'})
    )
    if 'focus_share' in streets.columns:
        streets['focus_share'] = streets['focus_share'].apply(lambda x: f"{x:.0%}" if isinstance(x, (int, float)) and not np.isnan(x) else '—')
    st.dataframe(streets.rename(columns={
        'signature_block': 'Block / anchor',
        'n_crimes': 'Incidents',
        'common_crime': 'Common offense',
        'district_id': 'District',
        'ward_id': 'Ward',
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
