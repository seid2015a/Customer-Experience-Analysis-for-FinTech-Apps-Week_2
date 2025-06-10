import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from collections import defaultdict
import numpy as np


from google.colab import drive
drive.mount('/content/drive')


# Initialize NLP tools
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_sentiment(text):
    """Analyze sentiment using DistilBERT model"""
    try:
        result = sentiment_analyzer(text[:512])[0]  # Truncate to model max length
        return {
            'sentiment_label': result['label'],
            'sentiment_score': result['score']
        }
    except:
        return {'sentiment_label': 'ERROR', 'sentiment_score': 0}

def preprocess_text(text):
    """Basic text preprocessing with spaCy"""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens)

def extract_keywords(texts, n=10):
    """Extract top keywords using TF-IDF"""
    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    tfidf.fit(texts)
    return tfidf.get_feature_names_out()[:n]

def cluster_reviews_into_themes(reviews, n_themes=5):
    """Group reviews into themes based on keywords"""
    # Preprocess all reviews
    processed = [preprocess_text(r) for r in reviews]
    
    # Get top keywords
    keywords = extract_keywords(processed)
    
    # Create theme clusters (simplified example)
    themes = {
        'Account Access': ['login', 'password', 'account', 'access'],
        'Transactions': ['transfer', 'payment', 'send money', 'transaction'],
        'UI/UX': ['app', 'interface', 'design', 'use'],
        'Customer Support': ['support', 'service', 'help', 'response'],
        'Performance': ['slow', 'crash', 'bug', 'freeze']
    }
    
    # Assign reviews to themes
    review_themes = []
    for review in reviews:
        matched_themes = []
        review_lower = review.lower()
        for theme, keywords in themes.items():
            if any(keyword in review_lower for keyword in keywords):
                matched_themes.append(theme)
        review_themes.append(", ".join(matched_themes) if matched_themes else "Other")
    
    return review_themes, themes

def process_bank_reviews(input_csv, bank_name):
    """Full processing pipeline for a bank's reviews"""
    # Load cleaned data
    df = pd.read_csv(input_csv)
    
    # Sentiment Analysis
    print(f"Analyzing sentiment for {bank_name}...")
    sentiment_results = df['review'].apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df, sentiment_results], axis=1)
    
    # Thematic Analysis
    print(f"Performing thematic analysis for {bank_name}...")
    themes, theme_keywords = cluster_reviews_into_themes(df['review'].tolist())
    df['theme'] = themes
    
    # Save results
    output_cols = [
        'review', 'rating', 'date', 'bank', 'source',
        'sentiment_label', 'sentiment_score', 'theme'
    ]
    output_filename = f"{bank_name.replace(' ', '_')}_analyzed_reviews.csv"
    df[output_cols].to_csv(output_filename, index=False)
    
    # Aggregate statistics
    agg_stats = df.groupby(['rating', 'theme']).agg({
        'sentiment_score': 'mean',
        'review': 'count'
    }).rename(columns={'review': 'review_count'})
    
    return df, agg_stats, theme_keywords

# Example usage
if __name__ == "__main__":
    banks = {
        '/content/drive/MyDrive/Colab Notebooks/BOA_reviews_pre_processed.csv': 'Bank of Abyssiniya (BOA)',
        '/content/drive/MyDrive/Colab Notebooks/CBE_reviews_pre_processed.csv': 'Commercial Bank of Ethiopia (CBE)',
        '/content/drive/MyDrive/Colab Notebooks/Dashen_reviews_pre_processed.csv': 'Dashen Bank'
    }
    
    all_results = {}
    for file, name in banks.items():
        print(f"\nProcessing {name}...")
        results, stats, themes = process_bank_reviews(file, name)
        all_results[name] = {
            'data': results,
            'stats': stats,
            'themes': themes
        }
        print(f"Completed analysis for {name}")
        print(f"Themes identified: {list(themes.keys())}")
