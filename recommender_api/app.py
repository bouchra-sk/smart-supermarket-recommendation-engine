# app.py
from flask import Flask, request, jsonify
import pickle
import joblib
import pandas as pd
import numpy as np
import random  # ← ADD THIS

app = Flask(__name__)

# Load all models
print("Loading models...")
tfidf = joblib.load('tfidf.pkl')
sim_df = pickle.load(open('sim_df.pkl', 'rb'))
products_catalog = pickle.load(open('products_catalog.pkl', 'rb'))
hour_scores_norm = pickle.load(open('hour_scores_norm.pkl', 'rb'))
season_popularity_norm = pickle.load(open('season_popularity_norm.pkl', 'rb'))
user_health_profile = pickle.load(open('user_health_profile.pkl', 'rb'))
print("✅ Models loaded successfully")

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        user_id = data.get('user_id')
        
        # Get products from user's cart and purchases
        cart_products = [item['product_id'] for item in data.get('current_cart', [])]
        purchased_products = [item['product_id'] for item in data.get('purchases', [])]
        
        user_products = list(set(cart_products + purchased_products))
        recommendations = []
        
        if user_products:
            for pid in user_products[:5]:
                if pid in sim_df.columns:
                    similar = sim_df[pid].sort_values(ascending=False).head(5)
                    for sim_pid, score in similar.items():
                        if sim_pid not in user_products:
                            product = products_catalog[(products_catalog['Product_ID'] == sim_pid)&
    (products_catalog['stock'] > 0)]
                            if not product.empty:
                                recommendations.append({
                                    'product_id': int(sim_pid),
                                    'name': product.iloc[0]['Name'],
                                    'price': float(product.iloc[0]['Price_DA']),
                                    'score': float(score)
                                })
        
        # Remove duplicates by NAME
        seen_names = set()
        unique_recs = []
        for rec in recommendations:
            if rec['name'] not in seen_names:
                seen_names.add(rec['name'])
                unique_recs.append(rec)
        
        # Shuffle for variety
        random.shuffle(unique_recs)
        
        return jsonify({'recommendations': unique_recs[:10]})
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'recommendations': [], 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)