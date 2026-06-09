"""
sentiment_model.py  —  VADER sentiment on real DinePulse reviews
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)


def get_vader_score(text: str, sia) -> float:
    return sia.polarity_scores(str(text))['compound']


def score_to_label(score: float) -> str:
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def apply_vader(df: pd.DataFrame, text_col: str = 'clean_review') -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    print("⏳ Running VADER sentiment analysis...")
    df['vader_score']     = df[text_col].apply(lambda x: get_vader_score(x, sia))
    df['vader_sentiment'] = df['vader_score'].apply(score_to_label)

    # Cross-check: if review_rating_clean exists, add rating-based label too
    if 'review_rating_clean' in df.columns:
        df['rating_sentiment'] = df['review_rating_clean'].apply(
            lambda r: 'Positive' if r >= 4 else ('Negative' if r <= 2 else 'Neutral')
            if pd.notna(r) else 'Unknown'
        )

    print("✅ VADER sentiment applied.")
    return df


if __name__ == '__main__':
    df = pd.read_csv('data/cleaned_reviews.csv')
    df = apply_vader(df)
    df.to_csv('data/sentiment_reviews.csv', index=False)
    print("Saved → data/sentiment_reviews.csv")

    print("\n📊 Sentiment Distribution (%):")
    print(df['vader_sentiment'].value_counts(normalize=True).mul(100).round(1).to_string())

    print("\n📍 Top Locations by Review Count:")
    print(df['location'].value_counts().head(10).to_string())

    print("\n🍽️  Top Cuisines:")
    print(df['cuisines'].value_counts().head(10).to_string())
