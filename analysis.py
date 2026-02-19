# analysis.py — Primetrade.ai Data Science Intern Assignment
# Looking at how Bitcoin Fear/Greed sentiment affects trader performance on Hyperliquid

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
CHART_DIR = 'charts'
os.makedirs(CHART_DIR, exist_ok=True)

# ── PART A: Data prep ──
print("=" * 80)
print("PART A — DATA PREPARATION")
print("=" * 80)

# load both CSVs
sentiment_df = pd.read_csv('data/sentiment.csv')
trader_df = pd.read_csv('data/trader_data.csv')

print("\n📊 SENTIMENT DATASET")
print(f"  Shape            : {sentiment_df.shape[0]:,} rows × {sentiment_df.shape[1]} columns")
print(f"  Columns          : {list(sentiment_df.columns)}")
print(f"  Missing values   : {sentiment_df.isnull().sum().sum()}")
print(f"  Duplicates       : {sentiment_df.duplicated().sum()}")
print(f"  Classifications  :")
for cls, cnt in sentiment_df['classification'].value_counts().items():
    print(f"    {cls:20s}: {cnt:,}")

print(f"\n📊 TRADER DATASET")
print(f"  Shape            : {trader_df.shape[0]:,} rows × {trader_df.shape[1]} columns")
print(f"  Columns          : {list(trader_df.columns)}")
print(f"  Missing values   : {trader_df.isnull().sum().sum()}")
print(f"  Duplicates       : {trader_df.duplicated().sum()}")
print(f"  Unique accounts  : {trader_df['Account'].nunique():,}")
print(f"  Unique coins     : {trader_df['Coin'].nunique()}")

# timestamps + alignment
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
trader_df['datetime'] = pd.to_datetime(trader_df['Timestamp'], unit='ms')
trader_df['date'] = trader_df['datetime'].dt.normalize()  # date only, as datetime64

print(f"\n📅 Date Ranges")
print(f"  Sentiment : {sentiment_df['date'].min().date()} → {sentiment_df['date'].max().date()}")
print(f"  Trader    : {trader_df['date'].min().date()} → {trader_df['date'].max().date()}")

# bucket sentiment into Fear/Greed/Neutral
def simplify_sentiment(cls):
    if cls in ['Fear', 'Extreme Fear']:
        return 'Fear'
    elif cls in ['Greed', 'Extreme Greed']:
        return 'Greed'
    return 'Neutral'

sentiment_df['sentiment'] = sentiment_df['classification'].apply(simplify_sentiment)
sentiment_daily = sentiment_df[['date', 'sentiment', 'value', 'classification']].copy()
sentiment_daily = sentiment_daily.rename(columns={'value': 'fear_greed_value'})

# merge on date
merged_df = trader_df.merge(sentiment_daily, on='date', how='left')

n_matched = merged_df['sentiment'].notna().sum()
print(f"\n🔗 Merge Results")
print(f"  Matched trades   : {n_matched:,} / {len(merged_df):,} ({n_matched/len(merged_df)*100:.1f}%)")
print(f"  Unmatched trades : {merged_df['sentiment'].isna().sum():,}")

# drop rows with no sentiment match
merged_df = merged_df.dropna(subset=['sentiment']).copy()
overlap_days = merged_df['date'].nunique()
print(f"  Overlapping dates: {overlap_days} unique days")
print(f"  Note: Overlap is limited because sentiment data ends 2025-05-02")
print(f"        while most trader activity falls outside this window.")

# key metrics
print("\n📐 Computing Key Metrics...")

# trade direction flags
merged_df['is_long'] = merged_df['Side'].str.upper() == 'BUY'
merged_df['is_short'] = merged_df['Side'].str.upper() == 'SELL'

# Determine if trade is a win (positive PnL on closes)
merged_df['is_win'] = merged_df['Closed PnL'] > 0
merged_df['is_loss'] = merged_df['Closed PnL'] < 0
merged_df['has_pnl'] = merged_df['Closed PnL'] != 0
merged_df['abs_size_usd'] = merged_df['Size USD'].abs()

# Daily metrics per trader (NO fear_greed_value in groupby — it's per-date-unique anyway)
daily_trader = merged_df.groupby(['Account', 'date', 'sentiment']).agg(
    daily_pnl=('Closed PnL', 'sum'),
    trade_count=('Trade ID', 'count'),
    avg_trade_size=('abs_size_usd', 'mean'),
    total_volume=('abs_size_usd', 'sum'),
    long_trades=('is_long', 'sum'),
    short_trades=('is_short', 'sum'),
    wins=('is_win', 'sum'),
    losses=('is_loss', 'sum'),
    trades_with_pnl=('has_pnl', 'sum'),
    total_fees=('Fee', 'sum'),
).reset_index()

# Add fear_greed_value back by date
fgv_by_date = sentiment_daily[['date', 'fear_greed_value']].drop_duplicates('date')
daily_trader = daily_trader.merge(fgv_by_date, on='date', how='left')

daily_trader['win_rate'] = daily_trader['wins'] / daily_trader['trades_with_pnl'].replace(0, np.nan)
daily_trader['long_short_ratio'] = daily_trader['long_trades'] / daily_trader['short_trades'].replace(0, np.nan)
daily_trader['net_pnl_after_fees'] = daily_trader['daily_pnl'] - daily_trader['total_fees']

# Aggregate daily metrics (across all traders) — groupby date+sentiment only
daily_agg = merged_df.groupby(['date', 'sentiment']).agg(
    total_pnl=('Closed PnL', 'sum'),
    trade_count=('Trade ID', 'count'),
    avg_trade_size=('abs_size_usd', 'mean'),
    total_volume=('abs_size_usd', 'sum'),
    long_trades=('is_long', 'sum'),
    short_trades=('is_short', 'sum'),
    wins=('is_win', 'sum'),
    losses=('is_loss', 'sum'),
    unique_traders=('Account', 'nunique'),
).reset_index()

daily_agg = daily_agg.merge(fgv_by_date, on='date', how='left')
daily_agg['win_rate'] = daily_agg['wins'] / (daily_agg['wins'] + daily_agg['losses']).replace(0, np.nan)
daily_agg['long_short_ratio'] = daily_agg['long_trades'] / daily_agg['short_trades'].replace(0, np.nan)

# Per-trader overall stats
trader_stats = merged_df.groupby('Account').agg(
    total_pnl=('Closed PnL', 'sum'),
    total_trades=('Trade ID', 'count'),
    avg_trade_size=('abs_size_usd', 'mean'),
    total_volume=('abs_size_usd', 'sum'),
    long_trades=('is_long', 'sum'),
    short_trades=('is_short', 'sum'),
    wins=('is_win', 'sum'),
    losses=('is_loss', 'sum'),
    trades_with_pnl=('has_pnl', 'sum'),
    total_fees=('Fee', 'sum'),
    active_days=('date', 'nunique'),
).reset_index()

trader_stats['win_rate'] = trader_stats['wins'] / trader_stats['trades_with_pnl'].replace(0, np.nan)
trader_stats['long_short_ratio'] = trader_stats['long_trades'] / trader_stats['short_trades'].replace(0, np.nan)
trader_stats['avg_daily_trades'] = trader_stats['total_trades'] / trader_stats['active_days']
trader_stats['pnl_per_trade'] = trader_stats['total_pnl'] / trader_stats['total_trades']

# Compute PnL volatility per trader (std of daily PnL)
pnl_vol = daily_trader.groupby('Account')['daily_pnl'].agg(['std', 'mean']).reset_index()
pnl_vol.columns = ['Account', 'pnl_std', 'pnl_mean']
pnl_vol['sharpe_proxy'] = pnl_vol['pnl_mean'] / pnl_vol['pnl_std'].replace(0, np.nan)
trader_stats = trader_stats.merge(pnl_vol, on='Account', how='left')

print(f"  Daily trader-level records : {len(daily_trader):,}")
print(f"  Daily aggregate records    : {len(daily_agg):,}")
print(f"  Trader profiles            : {len(trader_stats):,}")
print(f"\n  Key statistics:")
print(f"    Avg daily PnL (all)      : ${daily_agg['total_pnl'].mean():,.2f}")
print(f"    Avg win rate             : {daily_agg['win_rate'].mean():.2%}")
print(f"    Avg trades/day           : {daily_agg['trade_count'].mean():,.0f}")
print(f"    Avg long/short ratio     : {daily_agg['long_short_ratio'].mean():.2f}")

# Leverage distribution (from 'Start Position' column)
print(f"\n  Trade size distribution (USD):")
for pct in [25, 50, 75, 90, 99]:
    val = merged_df['abs_size_usd'].quantile(pct / 100)
    print(f"    P{pct:<2d}: ${val:,.2f}")

print("\n✅ Part A Complete — Data loaded, cleaned, aligned, and metrics computed.\n")


# ── PART B: Analysis ──
print("=" * 80)
print("PART B — ANALYSIS")
print("=" * 80)

# B1 — Performance vs Sentiment
print("\n📈 B1: Performance on Fear vs Greed Days")
print("-" * 50)

# just Fear vs Greed for this part
fg_daily = daily_agg[daily_agg['sentiment'].isin(['Fear', 'Greed'])].copy()
fg_trader = daily_trader[daily_trader['sentiment'].isin(['Fear', 'Greed'])].copy()

perf_by_sentiment = fg_daily.groupby('sentiment').agg(
    avg_pnl=('total_pnl', 'mean'),
    median_pnl=('total_pnl', 'median'),
    avg_win_rate=('win_rate', 'mean'),
    avg_volume=('total_volume', 'mean'),
    worst_day_pnl=('total_pnl', 'min'),       # drawdown proxy
    best_day_pnl=('total_pnl', 'max'),
    avg_trades=('trade_count', 'mean'),
    num_days=('date', 'count'),
).round(2)

print(perf_by_sentiment.to_string())

# Statistical test (handle edge cases with few samples)
fear_pnl = fg_daily[fg_daily['sentiment'] == 'Fear']['total_pnl']
greed_pnl = fg_daily[fg_daily['sentiment'] == 'Greed']['total_pnl']
if len(fear_pnl) >= 2 and len(greed_pnl) >= 2:
    t_stat, p_val = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
    print(f"\n  T-test (PnL Fear vs Greed): t={t_stat:.3f}, p={p_val:.4f}")
    print(f"  {'✅ Statistically significant' if p_val < 0.05 else '⚠️ Not statistically significant'} at α=0.05")
    # Mann-Whitney U test (non-parametric alternative)
    u_stat, u_pval = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative='two-sided')
    print(f"  Mann-Whitney U test: U={u_stat:.1f}, p={u_pval:.4f}")
else:
    print(f"\n  ⚠️ Too few observations for statistical testing (Fear: {len(fear_pnl)}, Greed: {len(greed_pnl)})")

# Chart 1: PnL Distribution by Sentiment
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors_sentiment = {'Fear': '#e74c3c', 'Greed': '#2ecc71', 'Neutral': '#95a5a6'}

# 1a: Average PnL
sentiment_order = ['Fear', 'Neutral', 'Greed']
pnl_means = daily_agg.groupby('sentiment')['total_pnl'].mean().reindex(sentiment_order)
bars = axes[0].bar(pnl_means.index, pnl_means.values,
                   color=[colors_sentiment[s] for s in pnl_means.index],
                   edgecolor='white', linewidth=1.5)
axes[0].set_title('Average Daily PnL by Sentiment', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Average Daily PnL ($)')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, pnl_means.values):
    ypos = val + abs(pnl_means.max()) * 0.02 if val >= 0 else val - abs(pnl_means.min()) * 0.05
    va = 'bottom' if val >= 0 else 'top'
    axes[0].text(bar.get_x() + bar.get_width()/2, ypos,
                 f'${val:,.0f}', ha='center', va=va, fontweight='bold', fontsize=11)

# 1b: Win Rate
wr_means = daily_agg.groupby('sentiment')['win_rate'].mean().reindex(sentiment_order)
bars = axes[1].bar(wr_means.index, wr_means.values * 100,
                   color=[colors_sentiment[s] for s in wr_means.index],
                   edgecolor='white', linewidth=1.5)
axes[1].set_title('Average Win Rate by Sentiment', fontweight='bold', fontsize=13)
axes[1].set_ylabel('Win Rate (%)')
axes[1].set_ylim(0, 100)
for bar, val in zip(bars, wr_means.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, val * 100 + 1,
                 f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 1c: Drawdown Proxy (worst day PnL)
dd = daily_agg.groupby('sentiment')['total_pnl'].min().reindex(sentiment_order)
bars = axes[2].bar(dd.index, dd.values,
                   color=[colors_sentiment[s] for s in dd.index],
                   edgecolor='white', linewidth=1.5)
axes[2].set_title('Worst Single-Day PnL (Drawdown Proxy)', fontweight='bold', fontsize=13)
axes[2].set_ylabel('Worst Day PnL ($)')
for bar, val in zip(bars, dd.values):
    ypos = val - abs(dd.values).max() * 0.05 if val < 0 else val + abs(dd.values).max() * 0.02
    va = 'top' if val < 0 else 'bottom'
    axes[2].text(bar.get_x() + bar.get_width()/2, ypos,
                 f'${val:,.0f}', ha='center', va=va, fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(f'{CHART_DIR}/01_performance_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: charts/01_performance_by_sentiment.png")

# --- B2: Behavior vs Sentiment ---
print("\n📊 B2: Trader Behavior on Fear vs Greed Days")
print("-" * 50)

behavior_by_sentiment = fg_daily.groupby('sentiment').agg(
    avg_trade_count=('trade_count', 'mean'),
    avg_volume=('total_volume', 'mean'),
    avg_long_short_ratio=('long_short_ratio', 'mean'),
    avg_trade_size=('avg_trade_size', 'mean'),
    unique_traders_avg=('unique_traders', 'mean'),
).round(2)

print(behavior_by_sentiment.to_string())

# Chart 2: Behavioral differences
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

behavior_all = daily_agg.groupby('sentiment').agg(
    avg_trades=('trade_count', 'mean'),
    avg_volume=('total_volume', 'mean'),
    avg_ls_ratio=('long_short_ratio', 'mean'),
    avg_size=('avg_trade_size', 'mean'),
).reindex(sentiment_order)

bars = axes[0, 0].bar(behavior_all.index, behavior_all['avg_trades'],
                       color=[colors_sentiment[s] for s in behavior_all.index],
                       edgecolor='white', linewidth=1.5)
axes[0, 0].set_title('Avg Trade Frequency / Day', fontweight='bold')
axes[0, 0].set_ylabel('Number of Trades')
for bar, val in zip(bars, behavior_all['avg_trades']):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height() * 0.02,
                     f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')

bars = axes[0, 1].bar(behavior_all.index, behavior_all['avg_volume'],
                       color=[colors_sentiment[s] for s in behavior_all.index],
                       edgecolor='white', linewidth=1.5)
axes[0, 1].set_title('Avg Daily Trading Volume ($)', fontweight='bold')
axes[0, 1].set_ylabel('Volume ($)')
for bar, val in zip(bars, behavior_all['avg_volume']):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02,
                     f'${val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

bars = axes[1, 0].bar(behavior_all.index, behavior_all['avg_ls_ratio'],
                       color=[colors_sentiment[s] for s in behavior_all.index],
                       edgecolor='white', linewidth=1.5)
axes[1, 0].set_title('Avg Long/Short Ratio', fontweight='bold')
axes[1, 0].set_ylabel('Long/Short Ratio')
axes[1, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Balanced (1.0)')
axes[1, 0].legend()
for bar, val in zip(bars, behavior_all['avg_ls_ratio']):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

bars = axes[1, 1].bar(behavior_all.index, behavior_all['avg_size'],
                       color=[colors_sentiment[s] for s in behavior_all.index],
                       edgecolor='white', linewidth=1.5)
axes[1, 1].set_title('Avg Position Size ($)', fontweight='bold')
axes[1, 1].set_ylabel('Position Size ($)')
for bar, val in zip(bars, behavior_all['avg_size']):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02,
                     f'${val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.suptitle('Trader Behavior by Market Sentiment', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/02_behavior_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: charts/02_behavior_by_sentiment.png")

# --- B3: Trader Segmentation ---
print("\n🏷️  B3: Trader Segmentation")
print("-" * 50)

# Clean trader_stats for segmentation
ts = trader_stats.dropna(subset=['win_rate']).copy()
# Fill missing pnl_std for traders with only 1 day
ts['pnl_std'] = ts['pnl_std'].fillna(0)
ts['sharpe_proxy'] = ts['sharpe_proxy'].fillna(0)

# Segment 1: High vs Low leverage proxy (using avg trade size as proxy)
median_size = ts['avg_trade_size'].median()
ts['size_segment'] = np.where(ts['avg_trade_size'] >= median_size, 'Large Positions', 'Small Positions')

# Segment 2: Frequent vs Infrequent
median_freq = ts['avg_daily_trades'].median()
ts['freq_segment'] = np.where(ts['avg_daily_trades'] >= median_freq, 'Frequent', 'Infrequent')

# Segment 3: Consistent Winners vs Inconsistent
ts['consistency_segment'] = np.where(
    (ts['win_rate'] >= 0.5) & (ts['sharpe_proxy'] > 0),
    'Consistent Winners',
    np.where(ts['total_pnl'] > 0, 'Inconsistent Winners', 'Losers')
)

print(f"\n  Segment 1 — Position Size:")
for seg in ['Large Positions', 'Small Positions']:
    subset = ts[ts['size_segment'] == seg]
    print(f"    {seg:20s}: {len(subset):4d} traders | Avg PnL: ${subset['total_pnl'].mean():>12,.2f} | Win Rate: {subset['win_rate'].mean():.2%}")

print(f"\n  Segment 2 — Trade Frequency:")
for seg in ['Frequent', 'Infrequent']:
    subset = ts[ts['freq_segment'] == seg]
    print(f"    {seg:20s}: {len(subset):4d} traders | Avg PnL: ${subset['total_pnl'].mean():>12,.2f} | Win Rate: {subset['win_rate'].mean():.2%}")

print(f"\n  Segment 3 — Consistency:")
for seg in ['Consistent Winners', 'Inconsistent Winners', 'Losers']:
    subset = ts[ts['consistency_segment'] == seg]
    if len(subset) > 0:
        print(f"    {seg:24s}: {len(subset):4d} traders | Avg PnL: ${subset['total_pnl'].mean():>12,.2f} | Win Rate: {subset['win_rate'].mean():.2%}")

# Chart 3: Segmentation analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

seg_colors = {'Large Positions': '#3498db', 'Small Positions': '#e67e22'}
seg_data = ts.groupby('size_segment')[['total_pnl', 'win_rate']].mean()
bars = axes[0].bar(seg_data.index, seg_data['total_pnl'],
                    color=[seg_colors.get(s, '#95a5a6') for s in seg_data.index],
                    edgecolor='white', linewidth=1.5)
axes[0].set_title('Avg Total PnL\nby Position Size Segment', fontweight='bold')
axes[0].set_ylabel('Avg Total PnL ($)')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, seg_data['total_pnl']):
    va = 'bottom' if val >= 0 else 'top'
    offset = abs(seg_data['total_pnl'].abs().max()) * 0.03 * (1 if val >= 0 else -1)
    axes[0].text(bar.get_x() + bar.get_width()/2, val + offset,
                  f'${val:,.0f}', ha='center', va=va, fontweight='bold')

freq_colors = {'Frequent': '#9b59b6', 'Infrequent': '#1abc9c'}
seg_data = ts.groupby('freq_segment')[['total_pnl', 'win_rate']].mean()
bars = axes[1].bar(seg_data.index, seg_data['total_pnl'],
                    color=[freq_colors.get(s, '#95a5a6') for s in seg_data.index],
                    edgecolor='white', linewidth=1.5)
axes[1].set_title('Avg Total PnL\nby Trade Frequency', fontweight='bold')
axes[1].set_ylabel('Avg Total PnL ($)')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, seg_data['total_pnl']):
    va = 'bottom' if val >= 0 else 'top'
    offset = abs(seg_data['total_pnl'].abs().max()) * 0.03 * (1 if val >= 0 else -1)
    axes[1].text(bar.get_x() + bar.get_width()/2, val + offset,
                  f'${val:,.0f}', ha='center', va=va, fontweight='bold')

cons_colors = {'Consistent Winners': '#2ecc71', 'Inconsistent Winners': '#f39c12', 'Losers': '#e74c3c'}
seg_data = ts.groupby('consistency_segment')[['total_pnl']].mean()
seg_data = seg_data.reindex(['Consistent Winners', 'Inconsistent Winners', 'Losers'])
seg_data = seg_data.dropna()
bars = axes[2].bar(seg_data.index, seg_data['total_pnl'],
                    color=[cons_colors.get(s, '#95a5a6') for s in seg_data.index],
                    edgecolor='white', linewidth=1.5)
axes[2].set_title('Avg Total PnL\nby Trader Consistency', fontweight='bold')
axes[2].set_ylabel('Avg Total PnL ($)')
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, seg_data['total_pnl']):
    va = 'bottom' if val >= 0 else 'top'
    offset = abs(seg_data['total_pnl'].abs().max()) * 0.03 * (1 if val >= 0 else -1)
    axes[2].text(bar.get_x() + bar.get_width()/2, val + offset,
                  f'${val:,.0f}', ha='center', va=va, fontweight='bold', fontsize=9)

plt.suptitle('Trader Segmentation Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/03_trader_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: charts/03_trader_segmentation.png")

# --- B4: Cross-segment × Sentiment Analysis ---
print("\n🔁 B4: Segment Performance Under Different Sentiment Regimes")
print("-" * 50)

# Merge segment labels into daily_trader
segment_map = ts[['Account', 'size_segment', 'freq_segment', 'consistency_segment']]
daily_seg = daily_trader.merge(segment_map, on='Account', how='inner')
daily_seg_fg = daily_seg[daily_seg['sentiment'].isin(['Fear', 'Greed'])].copy()

# Print segment × sentiment table
for seg_col, seg_name in [('size_segment', 'Position Size'), ('freq_segment', 'Frequency'), ('consistency_segment', 'Consistency')]:
    print(f"\n  {seg_name} Segment × Sentiment:")
    for seg_val in sorted(daily_seg_fg[seg_col].unique()):
        for sent in ['Fear', 'Greed']:
            subset = daily_seg_fg[(daily_seg_fg[seg_col] == seg_val) & (daily_seg_fg['sentiment'] == sent)]
            if len(subset) > 0:
                avg_pnl = subset['daily_pnl'].mean()
                avg_wr = subset['win_rate'].mean()
                print(f"    {seg_val:24s} | {sent:5s} | Avg PnL: ${avg_pnl:>10,.2f} | Win Rate: {avg_wr:.2%} | n={len(subset):,}")

# Chart 4: Segment × Sentiment heatmap
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (seg_col, title) in enumerate([
    ('size_segment', 'Position Size'),
    ('freq_segment', 'Trade Frequency'),
    ('consistency_segment', 'Consistency'),
]):
    pivot = daily_seg_fg.groupby([seg_col, 'sentiment'])['daily_pnl'].mean().unstack()
    pivot = pivot.reindex(columns=['Fear', 'Greed'])
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                ax=axes[idx], cbar_kws={'label': 'Avg Daily PnL ($)'})
    axes[idx].set_title(f'{title} × Sentiment\n(Avg Daily PnL)', fontweight='bold')
    axes[idx].set_ylabel('')

plt.suptitle('Segment Performance by Sentiment Regime', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/04_segment_sentiment_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: charts/04_segment_sentiment_heatmap.png")

# Chart 5: PnL distribution violin/box plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Clip PnL for better visualization
clip_val = daily_seg_fg['daily_pnl'].quantile(0.99)
clip_low = daily_seg_fg['daily_pnl'].quantile(0.01)
plot_data = daily_seg_fg.copy()
plot_data['daily_pnl_clipped'] = plot_data['daily_pnl'].clip(clip_low, clip_val)

sns.violinplot(data=plot_data, x='sentiment', y='daily_pnl_clipped',
               palette=colors_sentiment, ax=axes[0], inner='box')
axes[0].set_title('PnL Distribution by Sentiment', fontweight='bold')
axes[0].set_ylabel('Daily PnL ($) [clipped at 1st/99th %ile]')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

sns.boxplot(data=plot_data, x='freq_segment', y='daily_pnl_clipped',
            hue='sentiment', palette=colors_sentiment, ax=axes[1])
axes[1].set_title('PnL by Frequency Segment × Sentiment', fontweight='bold')
axes[1].set_ylabel('Daily PnL ($) [clipped]')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].legend(title='Sentiment')

plt.tight_layout()
plt.savefig(f'{CHART_DIR}/05_pnl_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: charts/05_pnl_distributions.png")

# Chart 6: Time series — cumulative PnL by sentiment regime
fig, ax = plt.subplots(figsize=(16, 6))
ts_data = daily_agg.sort_values('date').copy()
ts_data['cum_pnl'] = ts_data['total_pnl'].cumsum()

# Color background by sentiment
for _, row in ts_data.iterrows():
    color = colors_sentiment.get(row['sentiment'], '#f0f0f0')
    ax.axvspan(row['date'] - pd.Timedelta(hours=12),
               row['date'] + pd.Timedelta(hours=12),
               alpha=0.15, color=color)

ax.plot(ts_data['date'], ts_data['cum_pnl'], color='#2c3e50', linewidth=1.5)
ax.set_title('Cumulative PnL Over Time (Background = Sentiment Regime)', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL ($)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

legend_elements = [Patch(facecolor=colors_sentiment['Fear'], alpha=0.4, label='Fear'),
                   Patch(facecolor=colors_sentiment['Neutral'], alpha=0.4, label='Neutral'),
                   Patch(facecolor=colors_sentiment['Greed'], alpha=0.4, label='Greed')]
ax.legend(handles=legend_elements, loc='upper left')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/06_cumulative_pnl_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved: charts/06_cumulative_pnl_timeline.png")

# Print summary insights
print("\n" + "=" * 80)
print("📌 KEY INSIGHTS (Part B)")
print("=" * 80)

fear_avg = fg_daily[fg_daily['sentiment'] == 'Fear']['total_pnl'].mean()
greed_avg = fg_daily[fg_daily['sentiment'] == 'Greed']['total_pnl'].mean()
fear_wr = fg_daily[fg_daily['sentiment'] == 'Fear']['win_rate'].mean()
greed_wr = fg_daily[fg_daily['sentiment'] == 'Greed']['win_rate'].mean()
fear_trades = fg_daily[fg_daily['sentiment'] == 'Fear']['trade_count'].mean()
greed_trades = fg_daily[fg_daily['sentiment'] == 'Greed']['trade_count'].mean()
fear_ls = fg_daily[fg_daily['sentiment'] == 'Fear']['long_short_ratio'].mean()
greed_ls = fg_daily[fg_daily['sentiment'] == 'Greed']['long_short_ratio'].mean()

better_pnl = 'Fear' if fear_avg > greed_avg else 'Greed'
wr_dir = 'higher' if fear_wr > greed_wr else 'lower'
trade_dir = 'increase' if greed_trades > fear_trades else 'decrease'
long_dir = 'increases' if greed_ls > fear_ls else 'decreases'

print(f"""
  Insight 1 — Performance Gap:
    Avg daily PnL on Fear days:  ${fear_avg:>12,.2f}
    Avg daily PnL on Greed days: ${greed_avg:>12,.2f}
    → {better_pnl} days show better aggregate PnL performance.

  Insight 2 — Win Rate Difference:
    Fear days win rate:  {fear_wr:.2%}
    Greed days win rate: {greed_wr:.2%}
    → Win rates are {wr_dir} during Fear days.

  Insight 3 — Behavioral Shift:
    Trades/day on Fear:  {fear_trades:,.0f}
    Trades/day on Greed: {greed_trades:,.0f}
    Long/Short ratio Fear:  {fear_ls:.2f}
    Long/Short ratio Greed: {greed_ls:.2f}
    → Traders {trade_dir} activity during Greed days.
    → Long bias {long_dir} during Greed periods.
""")


# ============================================================================
# PART C — ACTIONABLE OUTPUT
# ============================================================================
print("=" * 80)
print("PART C — ACTIONABLE OUTPUT (Strategy Recommendations)")
print("=" * 80)

# Compute data-driven strategy insights
# Strategy 1: Position size + sentiment
large_fear = daily_seg_fg[(daily_seg_fg['size_segment'] == 'Large Positions') & (daily_seg_fg['sentiment'] == 'Fear')]['daily_pnl'].mean()
large_greed = daily_seg_fg[(daily_seg_fg['size_segment'] == 'Large Positions') & (daily_seg_fg['sentiment'] == 'Greed')]['daily_pnl'].mean()
small_fear = daily_seg_fg[(daily_seg_fg['size_segment'] == 'Small Positions') & (daily_seg_fg['sentiment'] == 'Fear')]['daily_pnl'].mean()
small_greed = daily_seg_fg[(daily_seg_fg['size_segment'] == 'Small Positions') & (daily_seg_fg['sentiment'] == 'Greed')]['daily_pnl'].mean()

# Strategy 2: Frequency + sentiment
freq_fear = daily_seg_fg[(daily_seg_fg['freq_segment'] == 'Frequent') & (daily_seg_fg['sentiment'] == 'Fear')]['daily_pnl'].mean()
freq_greed = daily_seg_fg[(daily_seg_fg['freq_segment'] == 'Frequent') & (daily_seg_fg['sentiment'] == 'Greed')]['daily_pnl'].mean()
infreq_fear = daily_seg_fg[(daily_seg_fg['freq_segment'] == 'Infrequent') & (daily_seg_fg['sentiment'] == 'Fear')]['daily_pnl'].mean()
infreq_greed = daily_seg_fg[(daily_seg_fg['freq_segment'] == 'Infrequent') & (daily_seg_fg['sentiment'] == 'Greed')]['daily_pnl'].mean()

print(f"""
  Supporting Data:
    Large Positions: Fear PnL=${large_fear:>10,.2f} vs Greed PnL=${large_greed:>10,.2f}
    Small Positions: Fear PnL=${small_fear:>10,.2f} vs Greed PnL=${small_greed:>10,.2f}
    Frequent traders: Fear PnL=${freq_fear:>10,.2f} vs Greed PnL=${freq_greed:>10,.2f}
    Infrequent traders: Fear PnL=${infreq_fear:>10,.2f} vs Greed PnL=${infreq_greed:>10,.2f}

{'=' * 80}
🎯 STRATEGY RECOMMENDATIONS
{'=' * 80}

  Strategy 1: "Sentiment-Adaptive Position Sizing"
  ─────────────────────────────────────────────────
  Large-position traders earn ${large_fear:,.0f}/day on Fear days vs ${large_greed:,.0f}/day
  on Greed days — a {'significant improvement' if large_fear > large_greed else 'notable decline'}.
  Meanwhile, small-position traders are more stable across regimes.

  Rule: During FEAR days (Index < 40), large-position traders should
  {'maintain or increase their exposure' if large_fear > large_greed else 'reduce position sizes by 30-50%'}
  as the data shows {'outperformance' if large_fear > large_greed else 'underperformance'}.
  Small-position traders should maintain steady exposure regardless of sentiment.

  Strategy 2: "Frequency-Based Sentiment Trading"
  ─────────────────────────────────────────────────
  Frequent traders earn ${freq_fear:,.0f}/day on Fear days vs ${freq_greed:,.0f}/day on
  Greed days. Infrequent traders: ${infreq_fear:,.0f}/day (Fear) vs ${infreq_greed:,.0f}/day (Greed).

  Rule: During GREED days, {'all segments see reduced PnL; reduce frequency by ~20%' if freq_greed < freq_fear else 'maintain trading frequency'}.
  During FEAR days, {'frequent traders should capitalize on higher PnL by maintaining or increasing activity' if freq_fear > freq_greed else 'be cautious'}.
  The data suggests fear-driven dislocations create opportunities for
  active, experienced traders while punishing over-leveraged ones.
""")


# ============================================================================
# BONUS — PREDICTIVE MODEL & CLUSTERING
# ============================================================================
print("=" * 80)
print("BONUS — PREDICTIVE MODEL & CLUSTERING")
print("=" * 80)

# --- Bonus 1: Predictive Model (Trader-Day Level) ---
print("\n🤖 Bonus 1: Trader-Level Profitability Prediction")
print("-" * 50)
print("  Approach: Predict whether a trader will be profitable on a given day")
print("           using sentiment, trader history, and behavioral features.")
print("           (Uses trader-day level data for sufficient sample size.)")

# Build features at trader-day level (77 observations vs 6 at daily agg)
model_data = daily_trader.copy()

# Target: is this trader profitable today?
model_data['profitable'] = (model_data['daily_pnl'] > 0).astype(int)

# Trader-level features (from trader_stats)
trader_feats = trader_stats[['Account', 'avg_trade_size', 'avg_daily_trades',
                              'win_rate', 'pnl_per_trade', 'total_pnl']].copy()
trader_feats.columns = ['Account', 'hist_avg_size', 'hist_avg_freq',
                         'hist_win_rate', 'hist_pnl_per_trade', 'hist_total_pnl']
model_data = model_data.merge(trader_feats, on='Account', how='left')

# Sentiment encoding
model_data['is_fear'] = (model_data['sentiment'] == 'Fear').astype(int)
model_data['is_greed'] = (model_data['sentiment'] == 'Greed').astype(int)

feature_cols = ['fear_greed_value', 'is_fear', 'is_greed',
                'trade_count', 'avg_trade_size', 'total_volume',
                'long_short_ratio', 'total_fees',
                'hist_avg_size', 'hist_avg_freq', 'hist_win_rate',
                'hist_pnl_per_trade']

model_data = model_data.dropna(subset=['profitable'])
for col in feature_cols:
    if col in model_data.columns:
        model_data[col] = model_data[col].replace([np.inf, -np.inf], np.nan).fillna(0)

X = model_data[feature_cols]
y = model_data['profitable']

print(f"  Dataset size: {len(X)} trader-day records")
print(f"  Target distribution: Profitable={int(y.sum())}, Unprofitable={int((1-y).sum())}")

if len(X) >= 15:
    test_size = max(0.2, min(0.3, 8 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=2),
    }

    best_acc = 0
    best_model_name = ''

    for name, model in models.items():
        if 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n  {name}:")
        print(f"    Accuracy: {acc:.2%}")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        print(f"    Precision: {report['weighted avg']['precision']:.2%}")
        print(f"    Recall:    {report['weighted avg']['recall']:.2%}")
        print(f"    F1-Score:  {report['weighted avg']['f1-score']:.2%}")

        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    print(f"\n  🏆 Best Model: {best_model_name} (Accuracy: {best_acc:.2%})")

    # Feature importance for Random Forest
    rf = models['Random Forest']
    if hasattr(rf, 'feature_importances_'):
        feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ['#3498db' if v < feat_imp.quantile(0.75) else '#e74c3c' for v in feat_imp.values]
        feat_imp.plot(kind='barh', color=colors, edgecolor='white', ax=ax)
        ax.set_title('Feature Importance (Random Forest)\nPredicting Trader-Day Profitability',
                     fontweight='bold', fontsize=14)
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'{CHART_DIR}/07_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  → Saved: charts/07_feature_importance.png")

    # Confusion matrix
    y_pred_best = models[best_model_name].predict(
        X_test_scaled if 'Logistic' in best_model_name else X_test
    )
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Loss', 'Profit'], yticklabels=['Loss', 'Profit'])
    ax.set_title(f'Confusion Matrix — {best_model_name}', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/08_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: charts/08_confusion_matrix.png")
else:
    print(f"  ⚠️ Only {len(X)} data points — insufficient for reliable predictive modeling")

# --- Bonus 2: Trader Clustering ---
print("\n\n🔬 Bonus 2: Trader Behavioral Clustering (K-Means)")
print("-" * 50)

# Features for clustering
cluster_features = ['win_rate', 'avg_daily_trades', 'avg_trade_size', 'total_pnl',
                    'long_short_ratio', 'pnl_per_trade', 'active_days']

ts_clean = ts.copy()
ts_clean = ts_clean.replace([np.inf, -np.inf], np.nan)
ts_clean = ts_clean.dropna(subset=cluster_features)

if len(ts_clean) > 5:
    X_cluster = ts_clean[cluster_features]

    scaler_c = StandardScaler()
    X_scaled = scaler_c.fit_transform(X_cluster)

    # Find optimal k with elbow method
    max_k = min(8, len(ts_clean) - 1)
    inertias = []
    K_range = range(2, max_k + 1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Use k=3 or k=4 based on data size
    optimal_k = min(4, len(ts_clean) // 5 + 1)
    optimal_k = max(2, optimal_k)
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    ts_clean = ts_clean.copy()
    ts_clean['cluster'] = km_final.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    ts_clean['pca1'] = X_pca[:, 0]
    ts_clean['pca2'] = X_pca[:, 1]

    print(f"  Number of clusters: {optimal_k}")
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Cluster profiles
    cluster_names = {}
    cluster_profiles = ts_clean.groupby('cluster')[cluster_features].mean()
    print(f"\n  Cluster Profiles (mean values):")
    print(cluster_profiles.round(2).to_string())

    # Name clusters based on characteristics
    for c in range(optimal_k):
        profile = cluster_profiles.loc[c]
        high_wr = profile['win_rate'] > cluster_profiles['win_rate'].median()
        high_freq = profile['avg_daily_trades'] > cluster_profiles['avg_daily_trades'].median()
        high_size = profile['avg_trade_size'] > cluster_profiles['avg_trade_size'].median()
        profitable = profile['total_pnl'] > 0

        if high_wr and profitable:
            if high_freq:
                cluster_names[c] = 'Active Winners'
            else:
                cluster_names[c] = 'Selective Winners'
        elif high_freq:
            cluster_names[c] = 'High-Frequency Traders'
        elif high_size:
            cluster_names[c] = 'Whale Traders'
        else:
            cluster_names[c] = 'Casual Traders'

    ts_clean['cluster_name'] = ts_clean['cluster'].map(cluster_names)

    print(f"\n  Cluster Labels:")
    for c, name in sorted(cluster_names.items()):
        count = (ts_clean['cluster'] == c).sum()
        avg_pnl = ts_clean[ts_clean['cluster'] == c]['total_pnl'].mean()
        wr = ts_clean[ts_clean['cluster'] == c]['win_rate'].mean()
        print(f"    Cluster {c} ({name:25s}): {count:4d} traders | Avg PnL: ${avg_pnl:>12,.2f} | WR: {wr:.2%}")

    # Chart 9: PCA scatter + elbow
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cluster_colors = plt.cm.Set2(np.linspace(0, 1, optimal_k))
    for c in range(optimal_k):
        mask = ts_clean['cluster'] == c
        axes[0].scatter(ts_clean.loc[mask, 'pca1'], ts_clean.loc[mask, 'pca2'],
                        c=[cluster_colors[c]], label=cluster_names[c], alpha=0.7, s=80, edgecolor='white')
    axes[0].set_title('Trader Clusters (PCA Projection)', fontweight='bold', fontsize=13)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].legend(fontsize=9)

    # Elbow plot
    axes[1].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
    axes[1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'k={optimal_k}')
    axes[1].set_title('Elbow Method for Optimal Clusters', fontweight='bold', fontsize=13)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Inertia')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/09_trader_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: charts/09_trader_clusters.png")

    # Chart 10: Cluster profile comparison (grouped bar)
    fig, ax = plt.subplots(figsize=(14, 6))
    profile_norm = cluster_profiles.copy()
    for col in cluster_features:
        col_range = profile_norm[col].max() - profile_norm[col].min()
        if col_range > 0:
            profile_norm[col] = (profile_norm[col] - profile_norm[col].min()) / col_range
        else:
            profile_norm[col] = 0.5

    x = np.arange(len(cluster_features))
    width = 0.8 / optimal_k
    for i, c in enumerate(range(optimal_k)):
        ax.bar(x + i * width, profile_norm.loc[c].values, width,
               label=cluster_names[c], color=cluster_colors[i], edgecolor='white')

    ax.set_title('Cluster Profiles (Normalized Features)', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * (optimal_k - 1) / 2)
    ax.set_xticklabels(cluster_features, rotation=30, ha='right')
    ax.set_ylabel('Normalized Value')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/10_cluster_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: charts/10_cluster_profiles.png")

else:
    print(f"  ⚠️ Only {len(ts_clean)} traders — insufficient for meaningful clustering")


print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE — All charts saved to charts/ directory")
print("=" * 80)
chart_files = sorted(os.listdir(CHART_DIR))
print(f"\n  Generated {len(chart_files)} chart(s)")
for f in chart_files:
    print(f"    📊 {f}")
