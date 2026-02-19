# Write-Up: Trader Performance vs Market Sentiment

## Approach

I started by loading the two datasets — the Bitcoin Fear/Greed Index (~2600 daily readings) and Hyperliquid trade data (~211K trades from 32 accounts). After converting timestamps and aligning on date, I merged them to see which trades happened on which sentiment days.

The main limitation hit early: only **6 days overlap** between the datasets. The Fear/Greed data stops at May 2, 2025, but most of the trader activity falls after that. So results should be taken as directional signals, not hard conclusions. Still, with 184K matched trades and 77 trader-day records, there's enough to work with.

I bucketed sentiment into three groups (Fear = Fear + Extreme Fear, Greed = Greed + Extreme Greed, Neutral), then computed metrics at three levels: per-trade, per-trader-day, and daily aggregate.

## What I Found

### 1. Fear days are way more profitable

| | Fear | Greed |
|---|---|---|
| Avg daily PnL | $6.7M | $842K |
| Win rate | 86% | 75% |
| Unique traders active | 32 | ~9 |
| Long/short ratio | 0.97 | 0.85 |

This was the biggest surprise. Conventional wisdom says fear = bad, but experienced traders on Hyperliquid clearly capitalize on fear-driven dislocations. The long/short ratio being near 1.0 during fear suggests they're taking balanced positions rather than panic-selling.

### 2. Traders change their behavior based on sentiment

On Fear days, pretty much all 32 traders are active and trading heavily (134K trades/day vs ~11K on Greed days). On Greed days, activity drops dramatically and the remaining traders lean slightly short. This could reflect profit-taking or a belief that Greed periods precede corrections.

### 3. Trader segments react differently

I split traders three ways (by position size, frequency, and win consistency) and cross-referenced:

- **Large-position traders** see the biggest Fear vs Greed gap: $288K/day vs $98K/day
- **Frequent traders** earn $324K/day during Fear vs $142K during Greed
- **Consistent winners** outperform in both regimes but the gap narrows during Greed

### 4. Four trader archetypes (clustering)

K-Means (k=4) on behavioral features revealed:

| Archetype | Count | Avg PnL | Win Rate |
|---|---|---|---|
| Casual Traders | 15 | $135K | 83% |
| High-Freq Traders | 2 | $876K | 88% |
| Active Winners | 9 | $122K | 91% |
| Selective Winners | 6 | $893K | 91% |

The two "whale" traders (High-Frequency cluster) stand out with enormous volume. The Selective Winners are interesting — fewer trades but very high PnL per trade, suggesting they pick their spots carefully.

## Predictive Model (Bonus)

The daily aggregate level only gave 6 data points, which obviously isn't enough for any ML model. So I reframed the problem: instead of predicting next-day market performance, I predict **whether a specific trader will be profitable on a given day**, using their behavioral features + sentiment.

This gives 77 observations. I tried three models:

- Logistic Regression: 87.5%
- Random Forest: 87.5%
- **Gradient Boosting: 93.75%**

The top features were historical win rate and trade volume, with sentiment features (fear/greed value, binary indicators) also contributing. A caveat: the model uses same-day features (trade count, position size), so it's more of a profitability classifier than a true predictor. A production version would need to use only lagged features.

## Strategy Recommendations

**Strategy 1 — Sentiment-Adaptive Position Sizing:**
Large-position traders should maintain or increase exposure during Fear days (Index < 40). The data shows they earn nearly 3x more on Fear days. Small-position traders are more stable across regimes, so they should keep steady sizing.

**Strategy 2 — Frequency-Based Sentiment Trading:**
During Greed days, everyone sees reduced PnL. Frequent traders should cut activity by ~20% and be more selective. During Fear days, active traders should capitalize — fear-driven dislocations create the best alpha opportunities.

## Limitations

- **6-day overlap** is the elephant in the room. All percentage comparisons should be treated carefully.
- The predictive model uses same-day features, not strictly forward-looking.
- Trader-day level (77 obs) is more robust than daily aggregate (6 obs), but still a limited sample.
- No out-of-sample validation across different time periods.

---

*Tools: Python, pandas, scikit-learn, matplotlib/seaborn, Streamlit + Plotly for the dashboard.*
