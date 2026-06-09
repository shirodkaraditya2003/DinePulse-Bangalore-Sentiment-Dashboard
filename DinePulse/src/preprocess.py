"""
preprocess.py  —  Cleans real DinePulse Kaggle dataset
Columns used: name, rate, votes, location, cuisines,
              approx_cost(for two people), reviews_list
"""

import pandas as pd
import re
import ast
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text: str) -> str:
    return ' '.join(w for w in text.split() if w not in STOP_WORDS)


# ── Parse reviews_list column ─────────────────────────────────────────────────
# Format in Kaggle dataset:
# "[('Rated 4.0', 'Great food!'), ('Rated 3.0', 'Okay place')]"

def parse_reviews(reviews_str: str) -> list:
    """Convert stringified list of tuples → list of (rating, review) tuples."""
    try:
        parsed = ast.literal_eval(str(reviews_str))
        return [(str(r[0]), str(r[1])) for r in parsed if len(r) == 2]
    except Exception:
        return []

def extract_reviews_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode reviews_list so each row = one review.
    Returns DataFrame with columns:
      restaurant_name, location, cuisines, cost, rate,
      review_rating, review_text, clean_review
    """
    rows = []
    for _, row in df.iterrows():
        reviews = parse_reviews(row.get('reviews_list', '[]'))
        for rev_rating, rev_text in reviews:
            if rev_text.strip() and rev_text.lower() not in ['nan', 'none', '']:
                rows.append({
                    'restaurant_name': str(row.get('name', '')).strip(),
                    'location':        str(row.get('location', '')).strip(),
                    'cuisines':        str(row.get('cuisines', '')).strip(),
                    'cost':            str(row.get('approx_cost(for two people)', '')),
                    'overall_rate':    str(row.get('rate', '')),
                    'votes':           row.get('votes', 0),
                    'rest_type':       str(row.get('rest_type', '')).strip(),
                    'online_order':    str(row.get('online_order', '')).strip(),
                    'review_rating':   rev_rating,
                    'review_text':     rev_text
                })

    reviews_df = pd.DataFrame(rows)
    print(f"📦 Extracted {len(reviews_df):,} individual reviews from {df['name'].nunique():,} restaurants")
    return reviews_df


# ── Clean Rate Column ─────────────────────────────────────────────────────────

def clean_rate(rate_str: str) -> float:
    """Convert '4.1/5' or '4.1' → 4.1. Returns NaN if invalid."""
    try:
        return float(str(rate_str).replace('/5', '').strip())
    except Exception:
        return float('nan')


# ── Clean Cost Column ─────────────────────────────────────────────────────────

def clean_cost(cost_str: str) -> float:
    try:
        return float(str(cost_str).replace(',', '').strip())
    except Exception:
        return float('nan')


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    print(f"📂 Loading {filepath} ...")
    df = pd.read_csv(filepath)
    print(f"   Raw shape: {df.shape}")

    # Drop duplicates
    df = df.drop_duplicates(subset=['name', 'location'])

    # Clean rate and cost
    df['rate_clean'] = df['rate'].apply(clean_rate)
    df['cost_clean'] = df['approx_cost(for two people)'].apply(clean_cost)

    # Extract individual reviews
    reviews_df = extract_reviews_df(df)

    # Clean review text
    reviews_df = reviews_df[reviews_df['review_text'].str.len() > 10]
    reviews_df['clean_review'] = reviews_df['review_text'].apply(clean_text)
    reviews_df['clean_review_no_sw'] = reviews_df['clean_review'].apply(remove_stopwords)

    # Clean review rating (e.g. "Rated 4.0" → 4.0)
    reviews_df['review_rating_clean'] = reviews_df['review_rating'].apply(
        lambda x: float(re.findall(r'\d+\.?\d*', str(x))[0])
        if re.findall(r'\d+\.?\d*', str(x)) else float('nan')
    )

    print(f"✅ Final reviews DataFrame: {len(reviews_df):,} rows")
    return reviews_df, df


if __name__ == '__main__':
    reviews_df, raw_df = load_and_clean('data/zomato.csv')
    reviews_df.to_csv('data/cleaned_reviews.csv', index=False)
    raw_df.to_csv('data/cleaned_restaurants.csv', index=False)
    print("Saved → data/cleaned_reviews.csv")
    print("Saved → data/cleaned_restaurants.csv")
    print("\n📊 Sample:")
    print(reviews_df[['restaurant_name', 'location', 'review_text']].head(3))
