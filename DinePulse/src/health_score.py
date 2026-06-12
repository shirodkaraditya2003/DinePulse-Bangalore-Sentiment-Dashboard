"""
health_score.py  —  Restaurant health scores from real DinePulse data
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def clean_cost(cost_str) -> float:
    try:
        return float(str(cost_str).replace(',', '').strip())
    except Exception:
        return float('nan')


def budget_segment(cost):
    if pd.isna(cost):
        return "Unknown"
    elif cost < 500:
        return "Budget"
    elif cost < 1000:
        return "Mid-range"
    elif cost < 2000:
        return "Premium"
    else:
        return "Luxury"


def calculate_health_score(group: pd.DataFrame) -> float:
    total = len(group)
    if total == 0:
        return 0.0
    pos = (group['vader_sentiment'] == 'Positive').sum()
    neu = (group['vader_sentiment'] == 'Neutral').sum()
    return round((pos * 1.0 + neu * 0.5) / total * 100, 1)


def top_complaints(group: pd.DataFrame, n: int = 5) -> str:
    neg = group[group['vader_sentiment'] == 'Negative']
    col = 'clean_review_no_sw' if 'clean_review_no_sw' in neg.columns else 'clean_review'
    neg_text = neg[col].dropna()
    if len(neg_text) < 3:
        return 'Not enough data'
    try:
        vec = CountVectorizer(stop_words='english', ngram_range=(1,2), max_features=30)
        X = vec.fit_transform(neg_text)
        counts = X.toarray().sum(axis=0)
        top = sorted(zip(vec.get_feature_names_out(), counts), key=lambda x: -x[1])[:n]
        return ', '.join(kw for kw, _ in top)
    except Exception:
        return ''


def build_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name, group in df.groupby('restaurant_name'):
        cost_raw = group['cost'].dropna().iloc[0] if 'cost' in group.columns and not group['cost'].dropna().empty else ''
        cost_clean = clean_cost(cost_raw)
        records.append({
            'restaurant_name': name,
            'location':        group['location'].mode().iloc[0] if not group['location'].mode().empty else '',
            'cuisines':        group['cuisines'].mode().iloc[0] if not group['cuisines'].mode().empty else '',
            'health_score':    calculate_health_score(group),
            'total_reviews':   len(group),
            'positive_pct':    round((group['vader_sentiment']=='Positive').mean()*100, 1),
            'negative_pct':    round((group['vader_sentiment']=='Negative').mean()*100, 1),
            'neutral_pct':     round((group['vader_sentiment']=='Neutral').mean()*100, 1),
            'avg_vader_score': round(group['vader_score'].mean(), 3),
            'approx_cost':     cost_clean,
            'budget_segment':  budget_segment(cost_clean),
            'top_complaints':  top_complaints(group),
        })

    scorecard = pd.DataFrame(records).sort_values('health_score', ascending=False)
    scorecard['rank'] = range(1, len(scorecard)+1)
    return scorecard


import sys

if __name__ == '__main__':
    # Fix for Windows console emoji encoding issues
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    df = pd.read_csv('data/sentiment_reviews.csv')
    scorecard = build_scorecard(df)
    scorecard.to_csv('data/restaurant_scorecard.csv', index=False)
    print(f"✅ Scorecard saved → data/restaurant_scorecard.csv")
    print(f"   Total restaurants scored: {len(scorecard)}")
    print("\n🏆 Top 10 Restaurants:")
    print(scorecard[['rank','restaurant_name','location','health_score','total_reviews']].head(10).to_string(index=False))
