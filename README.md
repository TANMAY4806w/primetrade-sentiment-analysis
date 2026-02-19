# Trader Performance vs Market Sentiment

Primetrade.ai — Data Science Intern (Round 0)

This project studies how Bitcoin market sentiment (the Fear/Greed Index) relates to trader behavior and performance on Hyperliquid.

## Project Structure

```
data/
  sentiment.csv       — daily Fear/Greed Index
  trader_data.csv     — Hyperliquid trade history
charts/               — generated plots (10 PNGs)
analysis.py           — all the analysis (Parts A through C, plus bonus)
dashboard.py          — Streamlit dashboard for interactive exploration
WRITEUP.md            — methodology + findings + strategy ideas
requirements.txt      — pip dependencies
```

## How to Run

```bash
pip install -r requirements.txt

# run the main analysis
python analysis.py

# launch the dashboard (optional)
streamlit run dashboard.py
```

The analysis script prints results to the console and saves charts to `charts/`.
The dashboard opens at `http://localhost:8501`.

## Data Overview

| Dataset | Rows | Cols | Dates |
|---------|------|------|-------|
| Sentiment | 2,644 | 4 | Feb 2018 – May 2025 |
| Trader | 211,224 | 16 | Mar 2023 – Jun 2025 |

Only **6 days overlap** between the two datasets (the FGI data ends before most trading activity), so some conclusions are directional rather than statistically conclusive. This is discussed more in the writeup.

## Headline Results

- Fear days produce ~8x higher aggregate PnL ($6.7M vs $842K daily average)
- Win rate is noticeably higher during Fear periods (86% vs 75%)
- Long/short ratio is near-balanced on Fear days (0.97) but tilts short on Greed days (0.85)
- K-Means clustering identifies 4 distinct trader archetypes
- Predictive model achieves 93.75% accuracy at the trader-day level (Gradient Boosting)

See [WRITEUP.md](WRITEUP.md) for full details and strategy recommendations.
