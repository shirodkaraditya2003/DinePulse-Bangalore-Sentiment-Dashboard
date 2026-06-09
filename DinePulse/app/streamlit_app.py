"""
streamlit_app.py  —  DinePulse Bangalore Sentiment Dashboard
Uses real Kaggle restaurant dataset with all columns.
Run: streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bangalore_coords import BANGALORE_COORDS

nltk.download('vader_lexicon', quiet=True)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DinePulse Sentiment Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #111111; }
    [data-testid="stSidebar"] { background-color: #1a1a1a; }
    h1,h2,h3 { color: #e23744 !important; }
    .metric-box {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

@st.cache_data
def load_reviews():
    path = os.path.join(BASE, 'sentiment_reviews.csv')
    if not os.path.exists(path):
        st.error(f"❌ File not found at: {path}")
        st.stop()
    df = pd.read_csv(path)
    df = df.dropna(subset=['restaurant_name','vader_sentiment'])
    df['restaurant_name'] = df['restaurant_name'].str.strip()
    df['location'] = df['location'].str.strip()
    return df

@st.cache_data
def load_scorecard():
    path = os.path.join(BASE, 'restaurant_scorecard.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def build_map_data(review_df):
    """Build a DataFrame of restaurants with coordinates for map display."""
    rest_stats = review_df.groupby(['restaurant_name', 'location']).agg(
        total_reviews  = ('vader_sentiment', 'count'),
        positive_pct   = ('vader_sentiment', lambda x: round((x == 'Positive').mean() * 100, 1)),
        negative_pct   = ('vader_sentiment', lambda x: round((x == 'Negative').mean() * 100, 1)),
        neutral_pct    = ('vader_sentiment', lambda x: round((x == 'Neutral').mean() * 100, 1)),
        avg_score      = ('vader_score', 'mean'),
    ).reset_index()

    # Add coordinates with slight jitter so restaurants in the same area don't overlap
    coords = []
    for _, row in rest_stats.iterrows():
        loc = row['location']
        base = BANGALORE_COORDS.get(loc, None)
        if base:
            # add small random offset so markers don't stack
            jitter_lat = np.random.uniform(-0.003, 0.003)
            jitter_lng = np.random.uniform(-0.003, 0.003)
            coords.append((base[0] + jitter_lat, base[1] + jitter_lng))
        else:
            coords.append((None, None))

    rest_stats['lat'] = [c[0] for c in coords]
    rest_stats['lng'] = [c[1] for c in coords]
    rest_stats = rest_stats.dropna(subset=['lat', 'lng'])
    rest_stats['avg_score'] = rest_stats['avg_score'].round(3)
    return rest_stats

@st.cache_resource
def get_sia():
    return SentimentIntensityAnalyzer()

df         = load_reviews()
scorecard  = load_scorecard()
map_data   = build_map_data(df)
sia        = get_sia()

SENTIMENT_COLORS = {
    'Positive': '#2ecc71',
    'Neutral':  '#f39c12',
    'Negative': '#e74c3c'
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ DinePulse Analyzer")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔍 Analyze Review",
        "📊 Restaurant Deep Dive",
        "📍 Location Analysis",
        "🗺️ Map Explorer",
        "🏆 Leaderboard"
    ])
    st.markdown("---")
    st.markdown(f"**Total Reviews:** {len(df):,}")
    st.markdown(f"**Restaurants:** {df['restaurant_name'].nunique():,}")
    st.markdown(f"**Locations:** {df['location'].nunique():,}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🍽️ DinePulse Bangalore — Sentiment Intelligence")
    st.markdown("Real data from thousands of Bangalore restaurants")
    st.markdown("---")

    # KPI Row
    c1,c2,c3,c4,c5 = st.columns(5)
    pos_pct = (df['vader_sentiment']=='Positive').mean()*100
    neg_pct = (df['vader_sentiment']=='Negative').mean()*100
    neu_pct = (df['vader_sentiment']=='Neutral').mean()*100

    c1.metric("📝 Total Reviews",    f"{len(df):,}")
    c2.metric("✅ Positive",         f"{pos_pct:.1f}%")
    c3.metric("❌ Negative",         f"{neg_pct:.1f}%")
    c4.metric("😐 Neutral",          f"{neu_pct:.1f}%")
    c5.metric("🏪 Restaurants",      f"{df['restaurant_name'].nunique():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Overall sentiment pie
    with col1:
        st.markdown("#### Overall Sentiment Split")
        counts = df['vader_sentiment'].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.45
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#f5f5f5')
        st.plotly_chart(fig, use_container_width=True)

    # Top cuisines by review count (split combos into individual cuisines)
    with col2:
        st.markdown("#### Top 10 Cuisines by Reviews")
        cuisine_exploded = df['cuisines'].dropna().str.split(',').explode().str.strip()
        top_cuisines = cuisine_exploded.value_counts().head(10).reset_index()
        top_cuisines.columns = ['Cuisine','Reviews']
        fig2 = px.bar(
            top_cuisines, x='Reviews', y='Cuisine',
            orientation='h',
            color='Reviews',
            color_continuous_scale='Reds'
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f5f5f5',
            yaxis={'categoryorder':'total ascending'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Sentiment by cuisine (split combos into individual cuisines)
    st.markdown("#### Sentiment by Cuisine Type")
    df_cuisine_split = df[['cuisines','vader_sentiment']].dropna(subset=['cuisines']).copy()
    df_cuisine_split['cuisine_single'] = df_cuisine_split['cuisines'].str.split(',')
    df_cuisine_split = df_cuisine_split.explode('cuisine_single')
    df_cuisine_split['cuisine_single'] = df_cuisine_split['cuisine_single'].str.strip()
    top_c = df_cuisine_split['cuisine_single'].value_counts().head(8).index
    cuisine_sentiment = (
        df_cuisine_split[df_cuisine_split['cuisine_single'].isin(top_c)]
        .groupby(['cuisine_single','vader_sentiment'])
        .size().reset_index(name='count')
    )
    fig3 = px.bar(
        cuisine_sentiment, x='cuisine_single', y='count',
        color='vader_sentiment',
        color_discrete_map=SENTIMENT_COLORS,
        barmode='group',
        title='Positive vs Negative Reviews per Cuisine',
        labels={'cuisine_single': 'Cuisine'}
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#f5f5f5',
        xaxis={'gridcolor':'#333'},
        yaxis={'gridcolor':'#333'}
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Online order vs sentiment — two pie charts side by side
    if 'online_order' in df.columns:
        st.markdown("#### 🛵 Online Order vs Sentiment")
        pie_col1, pie_col2 = st.columns(2)

        for col, label in zip([pie_col1, pie_col2], ['Yes', 'No']):
            subset = df[df['online_order'] == label]['vader_sentiment'].value_counts().reset_index()
            subset.columns = ['Sentiment', 'Count']
            fig_pie = px.pie(
                subset, values='Count', names='Sentiment',
                color='Sentiment',
                color_discrete_map=SENTIMENT_COLORS,
                hole=0.45,
                title=f"{'✅ Accepts' if label == 'Yes' else '❌ No'} Online Orders"
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>'
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f5f5f5',
                title_font_color='#f5f5f5',
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
                margin=dict(t=50, b=40, l=10, r=10),
                height=340
            )
            col.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYZE REVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyze Review":
    st.title("🔍 Instant Review Analyzer")
    st.markdown("Paste any restaurant review to get instant sentiment analysis")
    st.markdown("---")

    col1, col2 = st.columns([2,1])

    with col1:
        review_input = st.text_area("Paste your review:", height=160,
            placeholder="e.g. The butter chicken was amazing! Fast delivery, hot food...")

        if st.button("🔍 Analyze", use_container_width=True):
            if review_input.strip():
                score = sia.polarity_scores(review_input)['compound']
                if score >= 0.05:
                    sentiment, icon, color = 'Positive', '✅', '#2ecc71'
                elif score <= -0.05:
                    sentiment, icon, color = 'Negative', '❌', '#e74c3c'
                else:
                    sentiment, icon, color = 'Neutral', '😐', '#f39c12'

                st.markdown(f"""
                <div style='background:#1e1e1e;border-left:5px solid {color};
                            border-radius:10px;padding:20px;margin:15px 0'>
                    <h2 style='color:{color};margin:0'>{icon} {sentiment}</h2>
                    <p style='color:#aaa;margin:8px 0 0'>
                        Sentiment Score: <b style='color:{color}'>{score:+.3f}</b>
                        &nbsp;|&nbsp; Confidence: <b style='color:{color}'>{abs(score)*100:.1f}%</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={'font':{'color':color}},
                    gauge={
                        'axis': {'range':[-1,1], 'tickcolor':'#aaa'},
                        'bar':  {'color': color},
                        'steps':[
                            {'range':[-1,-0.05],'color':'#2d1117'},
                            {'range':[-0.05,0.05],'color':'#1e1e1e'},
                            {'range':[0.05,1],'color':'#112d1a'},
                        ]
                    },
                    title={'text':"Sentiment Score",'font':{'color':'#f5f5f5'}}
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f5f5f5', height=260
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 💡 Try Examples")
        examples = [
            ("Positive 😋", "Absolutely loved the food! Best paneer tikka I've had. Fast delivery and great packaging."),
            ("Negative 😤", "Terrible experience. Food was cold, arrived 2 hours late. Never ordering again."),
            ("Neutral 😐",  "Food was okay, nothing special. Delivery was on time at least."),
        ]
        for label, ex in examples:
            if st.button(label, use_container_width=True):
                st.info(ex)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESTAURANT DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Restaurant Deep Dive":
    st.title("📊 Restaurant Deep Dive")
    st.markdown("---")

    # Location filter → Restaurant filter
    locations = ['All Locations'] + sorted(df['location'].dropna().unique().tolist())
    sel_location = st.selectbox("📍 Filter by Location", locations)

    if sel_location != 'All Locations':
        filtered_df = df[df['location'] == sel_location]
    else:
        filtered_df = df

    restaurants_list = sorted(filtered_df['restaurant_name'].unique().tolist())
    if not restaurants_list:
        st.warning("No restaurants found for this filter.")
        st.stop()

    sel_restaurant = st.selectbox("🏪 Select Restaurant", restaurants_list)
    rest_df = filtered_df[filtered_df['restaurant_name'] == sel_restaurant]

    st.markdown("---")

    # KPIs
    pos_p = (rest_df['vader_sentiment']=='Positive').mean()*100
    neg_p = (rest_df['vader_sentiment']=='Negative').mean()*100
    neu_p = (rest_df['vader_sentiment']=='Neutral').mean()*100
    avg_s = rest_df['vader_score'].mean()
    health = (pos_p*1.0 + (100-pos_p-neg_p)*0.5)/100

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📝 Reviews",       len(rest_df))
    c2.metric("✅ Positive",      f"{pos_p:.1f}%")
    c3.metric("❌ Negative",      f"{neg_p:.1f}%")
    c4.metric("😐 Neutral",       f"{neu_p:.1f}%")
    c5.metric("💯 Health Score",  f"{health*100:.1f}/100")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sentiment Breakdown")
        counts = rest_df['vader_sentiment'].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index, color_discrete_map=SENTIMENT_COLORS, hole=0.4
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#f5f5f5')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Rating Distribution")
        if 'review_rating_clean' in rest_df.columns:
            fig2 = px.histogram(
                rest_df.dropna(subset=['review_rating_clean']),
                x='review_rating_clean', nbins=5,
                color_discrete_sequence=['#e23744'],
                labels={'review_rating_clean':'Rating'}
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f5f5f5',
                bargap=0.1
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Word Clouds
    st.markdown("#### 💬 What Customers Say")
    wc1, wc2 = st.columns(2)

    def make_wc(text_series, cmap):
        text = ' '.join(text_series.dropna())
        if len(text.strip()) < 20:
            return None
        wc = WordCloud(width=700,height=320,background_color='#111111',
                       colormap=cmap, max_words=60).generate(text)
        fig, ax = plt.subplots(figsize=(7,3))
        fig.patch.set_facecolor('#111111')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        return fig

    with wc1:
        st.markdown("**Positive Reviews**")
        pos_text = rest_df[rest_df['vader_sentiment']=='Positive']['clean_review']
        wc_fig = make_wc(pos_text, 'Greens')
        if wc_fig:
            st.pyplot(wc_fig)
        else:
            st.info("Not enough positive reviews")

    with wc2:
        st.markdown("**Negative Reviews**")
        neg_text = rest_df[rest_df['vader_sentiment']=='Negative']['clean_review']
        wc_fig2 = make_wc(neg_text, 'Reds')
        if wc_fig2:
            st.pyplot(wc_fig2)
        else:
            st.info("Not enough negative reviews")

    # Sample Reviews
    st.markdown("#### 📋 Sample Reviews")
    tab1, tab2 = st.tabs(["✅ Positive", "❌ Negative"])
    with tab1:
        pos_samples = rest_df[rest_df['vader_sentiment']=='Positive']['review_text'].head(5)
        for i, r in enumerate(pos_samples, 1):
            st.markdown(f"**{i}.** {r}")
    with tab2:
        neg_samples = rest_df[rest_df['vader_sentiment']=='Negative']['review_text'].head(5)
        for i, r in enumerate(neg_samples, 1):
            st.markdown(f"**{i}.** {r}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LOCATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📍 Location Analysis":
    st.title("📍 Location-wise Sentiment Analysis")
    st.markdown("Which areas of Bangalore have the happiest diners?")
    st.markdown("---")

    loc_stats = df.groupby('location').agg(
        total_reviews   = ('vader_sentiment','count'),
        positive_pct    = ('vader_sentiment', lambda x: (x=='Positive').mean()*100),
        negative_pct    = ('vader_sentiment', lambda x: (x=='Negative').mean()*100),
        neutral_pct     = ('vader_sentiment', lambda x: (x=='Neutral').mean()*100),
        avg_score       = ('vader_score','mean')
    ).reset_index()
    loc_stats = loc_stats[loc_stats['total_reviews'] >= 10].sort_values('positive_pct', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 15 Locations — Positive Review %")
        top15 = loc_stats.head(15).sort_values('positive_pct')
        fig = px.bar(
            top15, x='positive_pct', y='location',
            orientation='h', color='positive_pct',
            color_continuous_scale='Greens',
            labels={'positive_pct':'Positive %','location':'Area'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f5f5f5',
            coloraxis_showscale=False, height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Bottom 15 Locations — Most Complaints")
        bot15 = loc_stats.tail(15).sort_values('negative_pct', ascending=False)
        fig2 = px.bar(
            bot15, x='negative_pct', y='location',
            orientation='h', color='negative_pct',
            color_continuous_scale='Reds',
            labels={'negative_pct':'Negative %','location':'Area'}
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f5f5f5',
            coloraxis_showscale=False, height=500
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: reviews vs positive %
    st.markdown("#### Review Volume vs Sentiment Quality")
    fig3 = px.scatter(
        loc_stats, x='total_reviews', y='positive_pct',
        size='total_reviews', color='positive_pct',
        hover_name='location',
        color_continuous_scale='RdYlGn',
        labels={'total_reviews':'Total Reviews','positive_pct':'Positive %'}
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#f5f5f5'
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Full table
    st.markdown("#### 📋 Full Location Scorecard")
    st.dataframe(
        loc_stats[['location','total_reviews','positive_pct','negative_pct','neutral_pct','avg_score']]
        .round(2).reset_index(drop=True),
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Leaderboard":
    st.title("🏆 Restaurant Health Leaderboard")
    st.markdown("Ranked by sentiment health score across all reviews")
    st.markdown("---")

    if scorecard.empty:
        st.warning("Run `python src/health_score.py` to generate the scorecard first.")
        st.stop()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        loc_filter = st.selectbox("Filter by Location",
            ['All'] + sorted(scorecard['location'].dropna().unique().tolist()))
    with col2:
        min_reviews = st.slider("Minimum Reviews", 1, 50, 5)

    filtered_sc = scorecard[scorecard['total_reviews'] >= min_reviews]
    if loc_filter != 'All':
        filtered_sc = filtered_sc[filtered_sc['location'] == loc_filter]

    filtered_sc = filtered_sc.sort_values('health_score', ascending=False).reset_index(drop=True)
    filtered_sc['rank'] = range(1, len(filtered_sc)+1)
    filtered_sc['grade'] = filtered_sc['health_score'].apply(
        lambda s: '🥇 Excellent' if s>=75 else ('🥈 Good' if s>=55 else '🥉 Needs Work')
    )

    # Bar chart top 20
    top20 = filtered_sc.head(20).sort_values('health_score')
    fig = px.bar(
        top20, x='health_score', y='restaurant_name',
        orientation='h', color='health_score',
        color_continuous_scale=['#e74c3c','#f39c12','#2ecc71'],
        text='health_score',
        labels={'health_score':'Health Score','restaurant_name':'Restaurant'}
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#f5f5f5',
        coloraxis_showscale=False,
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.markdown("#### Full Scorecard")
    st.dataframe(
        filtered_sc[['rank','restaurant_name','location','health_score',
                     'total_reviews','positive_pct','negative_pct','neutral_pct','grade','top_complaints']],
        use_container_width=True,
        hide_index=True
    )

    # Download button
    csv = filtered_sc.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Scorecard CSV",
        csv, "restaurant_scorecard.csv", "text/csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MAP EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Map Explorer":
    st.title("🗺️ Bangalore Restaurant Map")
    st.markdown("Explore all restaurants across Bangalore — click any area to dive into reviews")
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    filter_col1, filter_col2 = st.columns([1, 1])

    with filter_col1:
        map_locations = ['All Locations'] + sorted(map_data['location'].unique().tolist())
        sel_map_loc = st.selectbox("📍 Filter by Area", map_locations, key='map_loc')

    with filter_col2:
        sentiment_filter = st.selectbox("🎯 Filter by Sentiment",
            ['All', 'Mostly Positive (>60%)', 'Mostly Negative (>30%)', 'Mixed'], key='map_sent')

    # Apply filters
    display_data = map_data.copy()
    if sel_map_loc != 'All Locations':
        display_data = display_data[display_data['location'] == sel_map_loc]

    if sentiment_filter == 'Mostly Positive (>60%)':
        display_data = display_data[display_data['positive_pct'] > 60]
    elif sentiment_filter == 'Mostly Negative (>30%)':
        display_data = display_data[display_data['negative_pct'] > 30]
    elif sentiment_filter == 'Mixed':
        display_data = display_data[
            (display_data['positive_pct'] <= 60) & (display_data['negative_pct'] <= 30)
        ]

    # KPIs for filtered view
    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric("🏪 Restaurants", f"{display_data['restaurant_name'].nunique():,}")
    kc2.metric("📍 Branches", f"{len(display_data):,}")
    kc3.metric("✅ Avg Positive %", f"{display_data['positive_pct'].mean():.1f}%")
    kc4.metric("❌ Avg Negative %", f"{display_data['negative_pct'].mean():.1f}%")
    kc5.metric("😐 Avg Neutral %", f"{display_data['neutral_pct'].mean():.1f}%")

    st.markdown("---")

    # ── Color-code by sentiment ───────────────────────────────────────────────
    def get_color(pos_pct):
        if pos_pct >= 70:
            return [46, 204, 113, 180]     # green
        elif pos_pct >= 50:
            return [243, 156, 18, 180]      # orange
        else:
            return [231, 76, 60, 180]       # red

    display_data = display_data.copy()
    display_data['color'] = display_data['positive_pct'].apply(get_color)
    display_data['radius'] = display_data['total_reviews'].clip(1, 100) * 8

    # ── Pydeck Map ────────────────────────────────────────────────────────────
    map_col, detail_col = st.columns([3, 2])

    with map_col:
        st.markdown("#### 📍 Restaurant Locations")
        st.markdown("🟢 Positive (>70%)     🟠 Mixed (50-70%)     🔴 Negative (<50%)")

        view_state = pdk.ViewState(
            latitude=12.9716,
            longitude=77.5946,
            zoom=11,
            pitch=45,
        )

        # Scatterplot layer
        scatter_layer = pdk.Layer(
            'ScatterplotLayer',
            data=display_data,
            get_position='[lng, lat]',
            get_color='color',
            get_radius='radius',
            pickable=True,
            opacity=0.8,
            auto_highlight=True,
        )

        # Column layer for 3D effect
        column_layer = pdk.Layer(
            'ColumnLayer',
            data=display_data,
            get_position='[lng, lat]',
            get_elevation='total_reviews * 5',
            elevation_scale=2,
            get_fill_color='color',
            radius=50,
            pickable=True,
            auto_highlight=True,
        )

        tooltip = {
            "html": """
            <div style='padding:8px;font-family:Arial;'>
                <b style='font-size:14px;'>{restaurant_name}</b><br/>
                <span style='color:#aaa'>{location}</span><br/>
                <hr style='margin:4px 0;border-color:#333'/>
                <span style='color:#2ecc71'>✅ {positive_pct}%</span> &nbsp;
                <span style='color:#e74c3c'>❌ {negative_pct}%</span> &nbsp;
                <span style='color:#f39c12'>😐 {neutral_pct}%</span><br/>
                <span>📝 {total_reviews} reviews</span>
            </div>
            """,
            "style": {"backgroundColor": "#1e1e1e", "color": "#f5f5f5"}
        }

        deck = pdk.Deck(
            layers=[scatter_layer, column_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style='mapbox://styles/mapbox/dark-v10',
        )
        st.pydeck_chart(deck, use_container_width=True)

    # ── Restaurant selector + reviews ─────────────────────────────────────────
    with detail_col:
        st.markdown("#### 🏪 Select a Restaurant")

        # Restaurant picker from filtered data
        rest_names = sorted(display_data['restaurant_name'].unique().tolist())
        if not rest_names:
            st.warning("No restaurants match the current filters.")
        else:
            sel_rest = st.selectbox("Pick a restaurant", rest_names, key='map_rest')

            # Get stats for selected restaurant
            rest_info = display_data[display_data['restaurant_name'] == sel_rest].iloc[0]

            # Sentiment card
            pos_color = '#2ecc71' if rest_info['positive_pct'] > 60 else ('#f39c12' if rest_info['positive_pct'] > 40 else '#e74c3c')
            st.markdown(f"""
            <div style='background:#1e1e1e;border-radius:12px;padding:16px;margin-bottom:12px;
                        border-left:4px solid {pos_color}'>
                <h3 style='margin:0;color:#f5f5f5 !important'>{sel_rest}</h3>
                <p style='color:#888;margin:4px 0'>📍 {rest_info['location']}</p>
                <hr style='border-color:#333;margin:8px 0'/>
                <div style='display:flex;justify-content:space-between'>
                    <span style='color:#2ecc71'>✅ {rest_info['positive_pct']}%</span>
                    <span style='color:#e74c3c'>❌ {rest_info['negative_pct']}%</span>
                    <span style='color:#f39c12'>😐 {rest_info['neutral_pct']}%</span>
                </div>
                <p style='color:#aaa;margin:8px 0 0'>📝 {rest_info['total_reviews']} reviews  |  Score: {rest_info['avg_score']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

            # Sentiment pie chart
            rest_reviews = df[df['restaurant_name'] == sel_rest]
            counts = rest_reviews['vader_sentiment'].value_counts()
            fig_pie = px.pie(
                values=counts.values, names=counts.index,
                color=counts.index, color_discrete_map=SENTIMENT_COLORS,
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font_color='#f5f5f5',
                height=250, margin=dict(t=10, b=10, l=10, r=10),
                showlegend=True, legend=dict(font=dict(size=10))
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Sample reviews
            st.markdown("#### 💬 Recent Reviews")
            tab_pos, tab_neg, tab_neu = st.tabs(["✅ Positive", "❌ Negative", "😐 Neutral"])

            with tab_pos:
                pos_rev = rest_reviews[rest_reviews['vader_sentiment'] == 'Positive']['review_text'].head(5)
                if len(pos_rev) == 0:
                    st.info("No positive reviews")
                for i, r in enumerate(pos_rev, 1):
                    st.markdown(f"**{i}.** {r}")

            with tab_neg:
                neg_rev = rest_reviews[rest_reviews['vader_sentiment'] == 'Negative']['review_text'].head(5)
                if len(neg_rev) == 0:
                    st.info("No negative reviews")
                for i, r in enumerate(neg_rev, 1):
                    st.markdown(f"**{i}.** {r}")

            with tab_neu:
                neu_rev = rest_reviews[rest_reviews['vader_sentiment'] == 'Neutral']['review_text'].head(5)
                if len(neu_rev) == 0:
                    st.info("No neutral reviews")
                for i, r in enumerate(neu_rev, 1):
                    st.markdown(f"**{i}.** {r}")
