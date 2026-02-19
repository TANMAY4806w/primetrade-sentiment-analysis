# dashboard.py — Interactive Streamlit dashboard for the sentiment analysis
# Run with: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Trader × Sentiment Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium Dark Theme CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0ff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stDateInput label {
        color: #b0b0ff !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #888;
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* KPI cards */
    .kpi-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0 1.5rem;
    }
    .kpi-card {
        flex: 1;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        opacity: 0.08;
        border-radius: 14px;
    }
    .kpi-card.blue   { background: linear-gradient(135deg, #1e3a5f, #0d1b2a); border: 1px solid #2a5a8f; }
    .kpi-card.green  { background: linear-gradient(135deg, #1a3d2e, #0d1b15); border: 1px solid #2a8f5a; }
    .kpi-card.purple { background: linear-gradient(135deg, #2d1f5e, #1a0d3a); border: 1px solid #5a2a8f; }
    .kpi-card.orange { background: linear-gradient(135deg, #3d2e1a, #1b150d); border: 1px solid #8f5a2a; }
    .kpi-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9aa5b4;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
    }
    .kpi-card.blue .kpi-value   { color: #6db3f8; }
    .kpi-card.green .kpi-value  { color: #6df8a8; }
    .kpi-card.purple .kpi-value { color: #b86df8; }
    .kpi-card.orange .kpi-value { color: #f8a86d; }

    /* Section headers */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #333;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #999;
        font-size: 0.8rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    .footer a { color: #667eea; text-decoration: none; }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
</style>
""", unsafe_allow_html=True)

# ─── Plotly Theme ────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=13),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(248,249,250,0.5)',
    margin=dict(l=40, r=20, t=50, b=40),
    title_font=dict(size=16, color='#333'),
    xaxis=dict(gridcolor='#eee', zerolinecolor='#ddd'),
    yaxis=dict(gridcolor='#eee', zerolinecolor='#ddd'),
    hoverlabel=dict(bgcolor='#333', font_color='white', font_size=12),
)

SENTIMENT_COLORS = {'Fear': '#ef4444', 'Greed': '#22c55e', 'Neutral': '#94a3b8'}
SEGMENT_COLORS = px.colors.qualitative.Set2


# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    sentiment_df = pd.read_csv('data/sentiment.csv')
    trader_df = pd.read_csv('data/trader_data.csv')

    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    def simplify_sentiment(cls):
        if cls in ['Fear', 'Extreme Fear']:
            return 'Fear'
        elif cls in ['Greed', 'Extreme Greed']:
            return 'Greed'
        return 'Neutral'

    sentiment_df['sentiment'] = sentiment_df['classification'].apply(simplify_sentiment)
    trader_df['datetime'] = pd.to_datetime(trader_df['Timestamp'], unit='ms')
    trader_df['date'] = trader_df['datetime'].dt.normalize()

    merged = trader_df.merge(
        sentiment_df[['date', 'sentiment', 'value', 'classification']].rename(
            columns={'value': 'fear_greed_value'}),
        on='date', how='left'
    )
    merged = merged.dropna(subset=['sentiment']).copy()
    merged['is_long'] = merged['Side'].str.upper() == 'BUY'
    merged['is_short'] = merged['Side'].str.upper() == 'SELL'
    merged['is_win'] = merged['Closed PnL'] > 0
    merged['has_pnl'] = merged['Closed PnL'] != 0
    merged['abs_size_usd'] = merged['Size USD'].abs()

    daily_trader = merged.groupby(['Account', 'date', 'sentiment']).agg(
        daily_pnl=('Closed PnL', 'sum'),
        trade_count=('Trade ID', 'count'),
        avg_trade_size=('abs_size_usd', 'mean'),
        total_volume=('abs_size_usd', 'sum'),
        long_trades=('is_long', 'sum'),
        short_trades=('is_short', 'sum'),
        wins=('is_win', 'sum'),
        trades_with_pnl=('has_pnl', 'sum'),
        total_fees=('Fee', 'sum'),
    ).reset_index()
    daily_trader['win_rate'] = daily_trader['wins'] / daily_trader['trades_with_pnl'].replace(0, np.nan)
    daily_trader['long_short_ratio'] = daily_trader['long_trades'] / daily_trader['short_trades'].replace(0, np.nan)

    daily_agg = merged.groupby(['date', 'sentiment']).agg(
        total_pnl=('Closed PnL', 'sum'),
        trade_count=('Trade ID', 'count'),
        avg_trade_size=('abs_size_usd', 'mean'),
        total_volume=('abs_size_usd', 'sum'),
        long_trades=('is_long', 'sum'),
        short_trades=('is_short', 'sum'),
        wins=('is_win', 'sum'),
        losses=('Closed PnL', lambda x: (x < 0).sum()),
        unique_traders=('Account', 'nunique'),
    ).reset_index()
    daily_agg['win_rate'] = daily_agg['wins'] / (daily_agg['wins'] + daily_agg['losses']).replace(0, np.nan)
    daily_agg['long_short_ratio'] = daily_agg['long_trades'] / daily_agg['short_trades'].replace(0, np.nan)

    trader_stats = merged.groupby('Account').agg(
        total_pnl=('Closed PnL', 'sum'),
        total_trades=('Trade ID', 'count'),
        avg_trade_size=('abs_size_usd', 'mean'),
        total_volume=('abs_size_usd', 'sum'),
        wins=('is_win', 'sum'),
        trades_with_pnl=('has_pnl', 'sum'),
        active_days=('date', 'nunique'),
        long_trades=('is_long', 'sum'),
        short_trades=('is_short', 'sum'),
    ).reset_index()
    trader_stats['win_rate'] = trader_stats['wins'] / trader_stats['trades_with_pnl'].replace(0, np.nan)
    trader_stats['avg_daily_trades'] = trader_stats['total_trades'] / trader_stats['active_days']
    trader_stats['pnl_per_trade'] = trader_stats['total_pnl'] / trader_stats['total_trades']
    trader_stats['long_short_ratio'] = trader_stats['long_trades'] / trader_stats['short_trades'].replace(0, np.nan)

    return merged, daily_trader, daily_agg, trader_stats, sentiment_df

merged, daily_trader, daily_agg, trader_stats, sentiment_df = load_data()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Dashboard Controls")
    st.markdown("---")

    st.markdown("#### 📅 Date Range")
    date_range = st.date_input(
        "Select dates",
        value=(merged['date'].min(), merged['date'].max()),
        min_value=merged['date'].min(),
        max_value=merged['date'].max(),
        label_visibility="collapsed",
    )

    st.markdown("#### 🎭 Sentiment Filter")
    sentiment_filter = st.multiselect(
        "Filter by sentiment",
        options=['Fear', 'Neutral', 'Greed'],
        default=['Fear', 'Neutral', 'Greed'],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size: 0.75rem; color: #888;">
            Primetrade.ai Intern Assignment<br/>
            Hyperliquid × Fear/Greed Index
        </span>
    </div>
    """, unsafe_allow_html=True)


# ─── Apply Filters ───────────────────────────────────────────────────────────
if len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    mask_a = (daily_agg['date'] >= d0) & (daily_agg['date'] <= d1) & (daily_agg['sentiment'].isin(sentiment_filter))
    mask_t = (daily_trader['date'] >= d0) & (daily_trader['date'] <= d1) & (daily_trader['sentiment'].isin(sentiment_filter))
else:
    mask_a = daily_agg['sentiment'].isin(sentiment_filter)
    mask_t = daily_trader['sentiment'].isin(sentiment_filter)

filtered_agg = daily_agg[mask_a]
filtered_trader = daily_trader[mask_t]


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Trader Performance × Market Sentiment</h1>
    <p>How does the Bitcoin Fear/Greed Index shape Hyperliquid trader behavior?</p>
</div>
""", unsafe_allow_html=True)


# ─── KPI Row ─────────────────────────────────────────────────────────────────
total_trades = filtered_agg['trade_count'].sum()
total_pnl = filtered_agg['total_pnl'].sum()
avg_wr = filtered_agg['win_rate'].mean()
n_days = filtered_agg['date'].nunique()

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card blue">
        <div class="kpi-label">Total Trades</div>
        <div class="kpi-value">{total_trades:,}</div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-label">Total PnL</div>
        <div class="kpi-value">${total_pnl:,.0f}</div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-label">Avg Win Rate</div>
        <div class="kpi-value">{avg_wr:.1%}</div>
    </div>
    <div class="kpi-card orange">
        <div class="kpi-label">Trading Days</div>
        <div class="kpi-value">{n_days}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Performance",
    "🔀  Behavior",
    "👤  Trader Profiles",
    "🔮  Clusters",
])


# ━━━━━━━━━━━━━━━━━━━━━━ TAB 1: PERFORMANCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown('<div class="section-title">Performance by Sentiment Regime</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        pnl_data = filtered_agg.groupby('sentiment')['total_pnl'].mean().reindex(
            ['Fear', 'Neutral', 'Greed']).reset_index()
        pnl_data.columns = ['Sentiment', 'Avg Daily PnL']

        fig = go.Figure(go.Bar(
            x=pnl_data['Sentiment'],
            y=pnl_data['Avg Daily PnL'],
            marker=dict(
                color=[SENTIMENT_COLORS.get(s, '#999') for s in pnl_data['Sentiment']],
                line=dict(width=0),
                cornerradius=6,
            ),
            text=[f"${v:,.0f}" for v in pnl_data['Avg Daily PnL']],
            textposition='outside',
            textfont=dict(size=13, color='#333', family='Inter'),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title='Average Daily PnL',
            yaxis_title='PnL ($)',
            showlegend=False,
        )
        fig.add_hline(y=0, line_dash='dot', line_color='#ccc')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        wr_data = filtered_agg.groupby('sentiment')['win_rate'].mean().reindex(
            ['Fear', 'Neutral', 'Greed']).reset_index()
        wr_data.columns = ['Sentiment', 'Win Rate']
        wr_data['Win Rate %'] = wr_data['Win Rate'] * 100

        fig = go.Figure(go.Bar(
            x=wr_data['Sentiment'],
            y=wr_data['Win Rate %'],
            marker=dict(
                color=[SENTIMENT_COLORS.get(s, '#999') for s in wr_data['Sentiment']],
                line=dict(width=0),
                cornerradius=6,
            ),
            text=[f"{v:.1f}%" for v in wr_data['Win Rate %']],
            textposition='outside',
            textfont=dict(size=13, color='#333', family='Inter'),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title='Average Win Rate',
            yaxis_title='Win Rate (%)',
            yaxis_range=[0, 100],
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cumulative PnL timeline
    st.markdown('<div class="section-title">Cumulative PnL Timeline</div>', unsafe_allow_html=True)

    ts_plot = filtered_agg.sort_values('date').copy()
    ts_plot['cum_pnl'] = ts_plot['total_pnl'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_plot['date'], y=ts_plot['cum_pnl'],
        mode='lines+markers',
        line=dict(color='#667eea', width=2.5),
        marker=dict(
            size=10,
            color=[SENTIMENT_COLORS.get(s, '#999') for s in ts_plot['sentiment']],
            line=dict(color='white', width=2),
        ),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Cumulative PnL: $%{y:,.0f}<extra></extra>',
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title='Cumulative PnL — Markers Colored by Sentiment',
        yaxis_title='Cumulative PnL ($)',
        xaxis_title='Date',
        height=400,
    )
    fig.add_hline(y=0, line_dash='dot', line_color='#ccc')
    st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━ TAB 2: BEHAVIOR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown('<div class="section-title">Behavioral Patterns by Sentiment</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    beh = filtered_agg.groupby('sentiment').agg(
        avg_trades=('trade_count', 'mean'),
        avg_volume=('total_volume', 'mean'),
        avg_ls=('long_short_ratio', 'mean'),
        avg_size=('avg_trade_size', 'mean'),
        avg_traders=('unique_traders', 'mean'),
    ).reindex(['Fear', 'Neutral', 'Greed']).reset_index()

    with c1:
        fig = go.Figure(go.Bar(
            x=beh['sentiment'], y=beh['avg_trades'],
            marker=dict(color=[SENTIMENT_COLORS.get(s) for s in beh['sentiment']], cornerradius=6),
            text=[f"{v:,.0f}" for v in beh['avg_trades']],
            textposition='outside',
            textfont=dict(size=12, family='Inter'),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Avg Trades / Day', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure(go.Bar(
            x=beh['sentiment'], y=beh['avg_volume'],
            marker=dict(color=[SENTIMENT_COLORS.get(s) for s in beh['sentiment']], cornerradius=6),
            text=[f"${v:,.0f}" for v in beh['avg_volume']],
            textposition='outside',
            textfont=dict(size=11, family='Inter'),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Avg Daily Volume ($)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = go.Figure(go.Bar(
            x=beh['sentiment'], y=beh['avg_ls'],
            marker=dict(color=[SENTIMENT_COLORS.get(s) for s in beh['sentiment']], cornerradius=6),
            text=[f"{v:.2f}" for v in beh['avg_ls']],
            textposition='outside',
            textfont=dict(size=12, family='Inter'),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Long/Short Ratio', showlegend=False)
        fig.add_hline(y=1.0, line_dash='dash', line_color='#999',
                      annotation_text='Balanced', annotation_position='top right')
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = go.Figure(go.Bar(
            x=beh['sentiment'], y=beh['avg_size'],
            marker=dict(color=[SENTIMENT_COLORS.get(s) for s in beh['sentiment']], cornerradius=6),
            text=[f"${v:,.0f}" for v in beh['avg_size']],
            textposition='outside',
            textfont=dict(size=11, family='Inter'),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Avg Position Size ($)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # PnL distribution
    st.markdown('<div class="section-title">PnL Distribution</div>', unsafe_allow_html=True)

    pnl_fg = filtered_trader[filtered_trader['sentiment'].isin(['Fear', 'Greed'])].copy()
    if len(pnl_fg) > 0:
        clip_hi = pnl_fg['daily_pnl'].quantile(0.98)
        clip_lo = pnl_fg['daily_pnl'].quantile(0.02)
        pnl_fg['pnl_clipped'] = pnl_fg['daily_pnl'].clip(clip_lo, clip_hi)

        fig = px.violin(pnl_fg, x='sentiment', y='pnl_clipped',
                        color='sentiment', color_discrete_map=SENTIMENT_COLORS,
                        box=True, points='all',
                        labels={'pnl_clipped': 'Daily PnL ($) [clipped]', 'sentiment': 'Sentiment'},
                        title='Trader-Level PnL Distribution (Fear vs Greed)')
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=450)
        fig.add_hline(y=0, line_dash='dot', line_color='#ccc')
        st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━ TAB 3: TRADER PROFILES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown('<div class="section-title">All Trader Profiles</div>', unsafe_allow_html=True)

    display = trader_stats.sort_values('total_pnl', ascending=False).copy()
    display['account_short'] = display['Account'].str[:10] + '…'

    fig = px.scatter(
        display, x='avg_daily_trades', y='total_pnl',
        size='total_trades', color='win_rate',
        color_continuous_scale='RdYlGn',
        range_color=[0.4, 1.0],
        hover_data={'Account': True, 'total_pnl': ':.0f', 'win_rate': ':.2%',
                    'avg_daily_trades': ':.0f', 'total_trades': True},
        title='Trader Map: Trade Frequency vs Total PnL',
        labels={'avg_daily_trades': 'Avg Trades / Day', 'total_pnl': 'Total PnL ($)',
                'win_rate': 'Win Rate', 'total_trades': 'Total Trades'},
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=500, coloraxis_colorbar_title='Win Rate')
    fig.add_hline(y=0, line_dash='dot', line_color='#ccc')
    st.plotly_chart(fig, use_container_width=True)

    # Styled data table
    st.markdown('<div class="section-title">Detailed Trader Table</div>', unsafe_allow_html=True)

    table_df = display[['account_short', 'total_pnl', 'total_trades', 'win_rate',
                         'avg_trade_size', 'avg_daily_trades', 'pnl_per_trade',
                         'long_short_ratio', 'active_days']].copy()
    table_df.columns = ['Account', 'Total PnL ($)', 'Trades', 'Win Rate',
                        'Avg Size ($)', 'Trades/Day', 'PnL/Trade ($)',
                        'L/S Ratio', 'Days']
    table_df['Win Rate'] = table_df['Win Rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    for col in ['Total PnL ($)', 'Avg Size ($)', 'PnL/Trade ($)']:
        table_df[col] = table_df[col].apply(lambda x: f"${x:,.2f}")
    table_df['L/S Ratio'] = table_df['L/S Ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    table_df['Trades/Day'] = table_df['Trades/Day'].apply(lambda x: f"{x:,.0f}")

    st.dataframe(table_df, use_container_width=True, hide_index=True, height=500)


# ━━━━━━━━━━━━━━━━━━━━━━ TAB 4: CLUSTERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown('<div class="section-title">Behavioral Archetypes (K-Means Clustering)</div>', unsafe_allow_html=True)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    cluster_features = ['win_rate', 'avg_daily_trades', 'avg_trade_size', 'total_pnl',
                        'pnl_per_trade', 'active_days']
    ts_clean = trader_stats.dropna(subset=cluster_features).copy()
    ts_clean = ts_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=cluster_features)

    c1, c2 = st.columns([1, 3])
    with c1:
        n_clusters = st.slider("Clusters (k)", 2, min(6, len(ts_clean) - 1), 4)

    if len(ts_clean) > n_clusters:
        X_c = ts_clean[cluster_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_c)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        ts_clean = ts_clean.copy()
        ts_clean['Cluster'] = km.fit_predict(X_scaled).astype(str)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        ts_clean['PC1'] = coords[:, 0]
        ts_clean['PC2'] = coords[:, 1]

        fig = px.scatter(
            ts_clean, x='PC1', y='PC2', color='Cluster',
            hover_data={'Account': True, 'total_pnl': ':.0f', 'win_rate': ':.2%'},
            title=f'Trader Clusters (k={n_clusters}) — PCA Projection',
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig.update_traces(marker=dict(size=14, line=dict(width=2, color='white')))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.0%} var)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.0%} var)',
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cluster profiles table
        st.markdown('<div class="section-title">Cluster Profiles</div>', unsafe_allow_html=True)

        profile = ts_clean.groupby('Cluster')[cluster_features].agg(['mean', 'count']).reset_index()
        profile.columns = ['_'.join(col).strip('_') for col in profile.columns]

        summary = ts_clean.groupby('Cluster').agg(
            count=('Account', 'count'),
            avg_pnl=('total_pnl', 'mean'),
            avg_wr=('win_rate', 'mean'),
            avg_trades=('avg_daily_trades', 'mean'),
            avg_size=('avg_trade_size', 'mean'),
        ).reset_index()

        summary['avg_pnl'] = summary['avg_pnl'].apply(lambda x: f"${x:,.0f}")
        summary['avg_wr'] = summary['avg_wr'].apply(lambda x: f"{x:.1%}")
        summary['avg_trades'] = summary['avg_trades'].apply(lambda x: f"{x:,.0f}")
        summary['avg_size'] = summary['avg_size'].apply(lambda x: f"${x:,.0f}")
        summary.columns = ['Cluster', 'Traders', 'Avg PnL', 'Avg Win Rate',
                           'Avg Trades/Day', 'Avg Position Size']

        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.warning("Not enough traders for the selected cluster count.")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built for <a href="#">Primetrade.ai</a> · Data Science Intern Assignment ·
    Hyperliquid Trader Data × Bitcoin Fear/Greed Index
</div>
""", unsafe_allow_html=True)
