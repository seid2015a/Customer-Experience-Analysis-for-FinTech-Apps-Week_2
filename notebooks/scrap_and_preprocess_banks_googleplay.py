from google_play_scraper import reviews, Sort
import pandas as pd
from datetime import datetime

def preprocess_banks_play_store_reviews(bank_name, app_id, review_count=500):
  
    try:
        # 1. Scrape raw reviews
        print(f"Scraping {review_count} reviews for {bank_name}...")
        raw_reviews, _ = reviews(
            app_id,
            count=review_count,
            lang='en',
            country='ET',
            sort=Sort.NEWEST
        )
        
        # 2. Convert to DataFrame
        df = pd.DataFrame(raw_reviews)
         # 3. Data Cleaning Pipeline
        
        # Remove duplicates based on key fields
        initial_count = len(df)
        df = df.drop_duplicates(
            subset=['content', 'score', 'at', 'userName'],
            keep='first'
        )
        print(f"Removed {initial_count - len(df)} duplicates")
        
        # Handle missing values
        df['content'] = df['content'].fillna('[No review text]')
        df['score'] = df['score'].fillna(df['score'].median()).astype(int)
        df['userName'] = df['userName'].fillna('Anonymous')
        
        # 4. Date Normalization (to YYYY-MM-DD)
        df['date'] = pd.to_datetime(df['at'], errors='coerce')
        df['date'] = df['date'].fillna(pd.to_datetime('today'))
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # 5. Select and rename required columns
        final_df = df.rename(columns={
            'content': 'review',
            'score': 'rating'
        })[['userName','review', 'at','rating', 'date']]
        
        # Add metadata columns
        final_df['bank'] = bank_name
        final_df['source'] = 'Google Play'

        #Text Cleaning 

        
        import re
        final_df['review'] = final_df['review'].apply(
            lambda x: re.sub(r'\s+', ' ', x).strip() if pd.notnull(x) else x
        )
        
        #Rating Distribution Check:

        print("Rating distribution:")
        print(final_df['rating'].value_counts().sort_index())
        
        #  Save to CSV
        filename = f"{bank_name.replace(' ', '_')}_reviews_pre_processed.csv"
        final_df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Saved {len(final_df)} cleaned reviews to {filename}")
        
        return final_df
    
    except Exception as e:
        print(f"Error processing {bank_name}: {str(e)}")
        return None
if __name__ == "__main__":
  # Scrape data for all banks
  banks = {'CBE':'com.combanketh.mobilebanking','BOA':'com.boa.boaMobileBanking','Dashen':'com.dashen.dashensuperapp'}
  for bank_name, app_id in banks.items():
      preprocess_banks_play_store_reviews(bank_name, app_id,review_count=500)

  print("Scraping completed for all banks!")