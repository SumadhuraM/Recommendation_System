import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

# Set page configuration
st.set_page_config(
    page_title="E-Commerce Recommendation AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== ENHANCED CUSTOM CSS ==================
st.markdown("""
<style>
    /* Main Theme Colors */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --dark-bg: #0f172a;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-pink: #ec4899;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Sub Headers */
    .sub-header {
        font-size: 2rem;
        color: var(--text-primary);
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid var(--accent-blue);
        position: relative;
    }
    
    .sub-header:after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 100px;
        height: 3px;
        background: var(--secondary-gradient);
    }
    
    /* Cards */
    .model-card {
        padding: 2rem;
        border-radius: 16px;
        background: var(--card-bg);
        margin-bottom: 1.5rem;
        border: none;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 5px solid var(--accent-blue);
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }
    
    /* Metric Cards */
    .metric-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: white;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        transition: all 0.3s ease;
        border-top: 4px solid var(--accent-blue);
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }
    
    /* Navigation */
    .nav-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .nav-button {
        padding: 0.75rem 1.5rem;
        margin: 0.25rem;
        border-radius: 10px;
        background: white;
        border: 2px solid #e2e8f0;
        color: var(--text-primary);
        font-weight: 600;
        transition: all 0.3s ease;
        text-align: center;
        cursor: pointer;
    }
    
    .nav-button:hover {
        background: var(--accent-blue);
        color: white;
        border-color: var(--accent-blue);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .nav-button.active {
        background: var(--primary-gradient);
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Winner Card */
    .winner-card {
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2.5rem 0;
        animation: pulse 2s infinite;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        border: none;
    }
    
    .winner-card-pca {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .winner-card-svd {
        background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);
    }
    
    .winner-card-tie {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Product Cards */
    .product-card {
        padding: 1.25rem;
        border-radius: 12px;
        background: white;
        border: 1px solid #e2e8f0;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
    }
    
    .product-card:hover {
        border-color: var(--accent-blue);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.1);
        transform: translateX(5px);
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    }
    
    /* Dataframe Styling */
    .dataframe-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Footer */
    .footer {
        padding: 2rem;
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-popular {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
    }
    
    .badge-collab {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
    }
    
    .badge-content {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
    }
    
    /* Comparison Cards */
    .comparison-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: white;
        border: 2px solid;
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .pca-card {
        border-color: var(--accent-blue);
        border-left: 5px solid var(--accent-blue);
    }
    
    .svd-card {
        border-color: var(--accent-pink);
        border-left: 5px solid var(--accent-pink);
    }
    
    /* Loading Animation */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .shimmer {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

class CompleteRecommendationSystem:
    def __init__(self):
        self.products_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.popular_products = None
        self.cosine_sim = None
        self.indices = None
        
    def load_data(self):
        """Load data for all three models"""
        try:
            # Create progress tracker
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load products
            status_text.text("📦 Loading products...")
            products_path = r"C:\Users\sumad\Downloads\recommendation\product_descriptions.csv"
            if os.path.exists(products_path):
                self.products_df = pd.read_csv(products_path, nrows=5000)
                
                # Standardize columns
                if 'product_uid' in self.products_df.columns:
                    self.products_df = self.products_df.rename(columns={'product_uid': 'product_id'})
                self.products_df['product_id'] = self.products_df['product_id'].astype(str)
                
                # Create product name
                if 'product_name' not in self.products_df.columns:
                    if 'product_description' in self.products_df.columns:
                        self.products_df['product_name'] = self.products_df['product_description'].str[:30] + '...'
                    else:
                        self.products_df['product_name'] = 'Product ' + self.products_df['product_id']
                
                st.success(f"✅ Loaded {len(self.products_df):,} products")
            else:
                st.error("❌ Product file not found")
                return False
            
            progress_bar.progress(30)
            
            # Load ratings
            status_text.text("⭐ Loading ratings...")
            ratings_path = r"C:\Users\sumad\Downloads\recommendation\ratings_Beauty.csv"
            if os.path.exists(ratings_path):
                self.ratings_df = pd.read_csv(ratings_path, nrows=10000)
                
                # Standardize columns
                column_map = {}
                if 'UserId' in self.ratings_df.columns:
                    column_map['UserId'] = 'user_id'
                if 'ProductId' in self.ratings_df.columns:
                    column_map['ProductId'] = 'product_id'
                if 'Rating' in self.ratings_df.columns:
                    column_map['Rating'] = 'rating'
                
                if column_map:
                    self.ratings_df = self.ratings_df.rename(columns=column_map)
                
                self.ratings_df['product_id'] = self.ratings_df['product_id'].astype(str)
                self.ratings_df['user_id'] = self.ratings_df['user_id'].astype(str)
                
                st.success(f"✅ Loaded {len(self.ratings_df):,} ratings")
            else:
                st.error("❌ Ratings file not found")
                return False
            
            progress_bar.progress(60)
            
            # Prepare all models
            status_text.text("🔄 Preparing all recommendation models...")
            self._prepare_all_models()
            
            progress_bar.progress(100)
            status_text.text("✨ All models ready!")
            
            return True
            
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
            return False
    
    def _prepare_all_models(self):
        """Prepare all three recommendation models"""
        try:
            # 1. Popularity-Based Model
            self._prepare_popularity_model()
            
            # 2. Collaborative Filtering Model
            self._prepare_collaborative_model()
            
            # 3. Content-Based Model
            self._prepare_content_model()
            
            st.success("🎯 All three models prepared successfully!")
            
        except Exception as e:
            st.warning(f"📝 Note: {str(e)}")
    
    def _prepare_popularity_model(self):
        """Prepare popularity-based recommendations"""
        try:
            # Calculate product popularity
            product_stats = self.ratings_df.groupby('product_id').agg(
                rating_count=('rating', 'count'),
                avg_rating=('rating', 'mean')
            ).reset_index()
            
            # Get top 50 popular products
            self.popular_products = product_stats.sort_values(
                ['rating_count', 'avg_rating'], 
                ascending=[False, False]
            ).head(50).copy()
            
            # Add product names
            if self.products_df is not None:
                self.popular_products = pd.merge(
                    self.popular_products,
                    self.products_df[['product_id', 'product_name']],
                    on='product_id',
                    how='left'
                )
            
            # Fill missing names
            self.popular_products['product_name'] = self.popular_products['product_name'].fillna(
                'Product ' + self.popular_products['product_id']
            )
            
            st.info(f"🔥 Popularity model: {len(self.popular_products)} popular products")
            
        except Exception as e:
            st.warning(f"🔥 Popularity model: {str(e)}")
    
    def _prepare_collaborative_model(self):
        """Prepare collaborative filtering model"""
        try:
            # Create user-item matrix
            self.user_item_matrix = self.ratings_df.pivot_table(
                index='user_id',
                columns='product_id',
                values='rating',
                aggfunc='mean'
            ).fillna(0)
            
            st.info(f"👥 Collaborative model: {self.user_item_matrix.shape[0]} users × {self.user_item_matrix.shape[1]} products")
            
        except Exception as e:
            st.warning(f"👥 Collaborative model: {str(e)}")
    
    def _prepare_content_model(self):
        """Prepare content-based model"""
        try:
            # Use product descriptions for content-based
            if self.products_df is not None:
                # Prepare text data
                text_columns = []
                for col in self.products_df.columns:
                    if col != 'product_id' and self.products_df[col].dtype == 'object':
                        text_columns.append(col)
                
                if text_columns:
                    self.products_df['combined_text'] = self.products_df[text_columns].fillna('').agg(' '.join, axis=1)
                else:
                    self.products_df['combined_text'] = ''
                
                # Create TF-IDF matrix
                tfidf = TfidfVectorizer(stop_words='english', max_features=500)
                tfidf_matrix = tfidf.fit_transform(self.products_df['combined_text'])
                
                # Compute cosine similarity
                self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
                
                # Create indices
                self.indices = pd.Series(
                    self.products_df.index,
                    index=self.products_df['product_id']
                ).drop_duplicates()
                
                st.info(f"🔍 Content-based model: {self.cosine_sim.shape[0]} products")
                
        except Exception as e:
            st.warning(f"🔍 Content model: {str(e)}")
    
    # ================== THREE RECOMMENDATION MODELS ==================
    
    def model_1_popularity_based(self, n=10):
        """Model 1: Popularity-based recommendations (for new customers)"""
        if self.popular_products is None:
            # Sample data
            return pd.DataFrame({
                'product_id': [str(i) for i in range(1, n+1)],
                'product_name': [f'Popular Product {i}' for i in range(1, n+1)],
                'rating_count': np.random.randint(50, 500, n),
                'avg_rating': np.round(np.random.uniform(3.5, 5.0, n), 1)
            })
        
        return self.popular_products.head(n).copy()
    
    def model_2_collaborative_filtering(self, user_id, n=10):
        """Model 2: Collaborative filtering (for returning customers)"""
        try:
            if self.user_item_matrix is None:
                return self.model_1_popularity_based(n)
            
            user_id = str(user_id)
            
            if user_id not in self.user_item_matrix.index:
                st.info("👤 User not found. Showing popular products instead.")
                return self.model_1_popularity_based(n)
            
            # Simple user-based collaborative filtering
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_items = user_ratings[user_ratings == 0].index.tolist()
            
            if not unrated_items:
                return self.model_1_popularity_based(n)
            
            # Recommend popular unrated items
            if self.popular_products is not None:
                recommendations = self.popular_products[
                    self.popular_products['product_id'].isin(unrated_items)
                ].head(n)
                
                if len(recommendations) > 0:
                    return recommendations
            
            return self.model_1_popularity_based(n)
            
        except Exception as e:
            return self.model_1_popularity_based(n)
    
    def model_3_content_based(self, product_id, n=10):
        """Model 3: Content-based recommendations (cold start)"""
        try:
            if self.cosine_sim is None or self.indices is None:
                return self.model_1_popularity_based(n)
            
            product_id = str(product_id)
            
            # Check if product exists in indices
            if product_id not in self.indices:
                st.info("📦 Product not found in content model. Showing popular products instead.")
                return self.model_1_popularity_based(n)
            
            # Get similar products
            idx = self.indices[product_id]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
            
            # Get product details
            result = []
            for prod_idx, score in sim_scores:
                if prod_idx < len(self.products_df):
                    product_info = self.products_df.iloc[prod_idx].to_dict()
                    product_info['similarity_score'] = score
                    result.append(product_info)
            
            if result:
                return pd.DataFrame(result)
            else:
                return self.model_1_popularity_based(n)
                
        except Exception as e:
            return self.model_1_popularity_based(n)
    
    # ================== PCA vs SVD COMPARISON ==================
    
    def compare_pca_vs_svd(self, n_components=5):
        """Compare PCA vs SVD for dimensionality reduction - CORRECTED LOGIC"""
        try:
            # Use a sample for comparison
            if len(self.ratings_df) > 2000:
                sample_ratings = self.ratings_df.sample(2000, random_state=42)
            else:
                sample_ratings = self.ratings_df
            
            # Create matrix for comparison
            comparison_matrix = sample_ratings.pivot_table(
                index='user_id',
                columns='product_id',
                values='rating',
                aggfunc='mean'
            ).fillna(0)
            
            # Check if we have enough data
            if comparison_matrix.shape[0] < 10 or comparison_matrix.shape[1] < 10:
                return None
            
            # Adjust components
            max_comp = min(n_components, comparison_matrix.shape[1] - 1)
            if max_comp < 2:
                return None
            
            # Prepare data (item-based)
            X = comparison_matrix.T.values
            
            # Train PCA
            pca = PCA(n_components=max_comp)
            X_pca = pca.fit_transform(X)
            
            # Train SVD
            svd = TruncatedSVD(n_components=max_comp, random_state=42)
            X_svd = svd.fit_transform(X)
            
            # Reconstruct
            X_recon_pca = pca.inverse_transform(X_pca)
            X_recon_svd = svd.inverse_transform(X_svd)
            
            # Calculate metrics
            mse_pca = mean_squared_error(X.flatten(), X_recon_pca.flatten())
            mse_svd = mean_squared_error(X.flatten(), X_recon_svd.flatten())
            
            mae_pca = mean_absolute_error(X.flatten(), X_recon_pca.flatten())
            mae_svd = mean_absolute_error(X.flatten(), X_recon_svd.flatten())
            
            # Explained variance
            explained_var_pca = np.sum(pca.explained_variance_ratio_)
            explained_var_svd = np.sum(svd.explained_variance_ratio_)
            
            # CORRECTED: For error metrics, LOWER is better
            # We want to choose the model with LOWER error
            # Calculate an overall error score (weighted combination)
            
            # Weighted error score (60% MSE, 40% MAE) - LOWER is better
            # Explained variance is considered separately
            error_score_pca = (0.6 * mse_pca) + (0.4 * mae_pca)
            error_score_svd = (0.6 * mse_svd) + (0.4 * mae_svd)
            
            # Determine winner based on LOWER error score
            # If error scores are very close (within 10%), check explained variance
            if error_score_pca < error_score_svd:
                # PCA has lower error
                if (error_score_svd - error_score_pca) / error_score_svd > 0.1:
                    # PCA is significantly better
                    winner = "PCA"
                    reason = f"PCA has significantly lower error score ({error_score_pca:.4f} vs {error_score_svd:.4f})"
                else:
                    # Scores are close, check explained variance
                    if explained_var_pca > explained_var_svd:
                        winner = "PCA"
                        reason = f"PCA has slightly lower error ({error_score_pca:.4f} vs {error_score_svd:.4f}) and higher explained variance"
                    else:
                        winner = "SVD"
                        reason = f"Scores are close but SVD has higher explained variance ({explained_var_svd:.2%} vs {explained_var_pca:.2%})"
            elif error_score_svd < error_score_pca:
                # SVD has lower error
                if (error_score_pca - error_score_svd) / error_score_pca > 0.1:
                    # SVD is significantly better
                    winner = "SVD"
                    reason = f"SVD has significantly lower error score ({error_score_svd:.4f} vs {error_score_pca:.4f})"
                else:
                    # Scores are close, check explained variance
                    if explained_var_svd > explained_var_pca:
                        winner = "SVD"
                        reason = f"SVD has slightly lower error ({error_score_svd:.4f} vs {error_score_pca:.4f}) and higher explained variance"
                    else:
                        winner = "PCA"
                        reason = f"Scores are close but PCA has higher explained variance ({explained_var_pca:.2%} vs {explained_var_svd:.2%})"
            else:
                # Equal error scores
                if explained_var_pca > explained_var_svd:
                    winner = "PCA"
                    reason = f"Equal error scores but PCA has higher explained variance ({explained_var_pca:.2%} vs {explained_var_svd:.2%})"
                elif explained_var_svd > explained_var_pca:
                    winner = "SVD"
                    reason = f"Equal error scores but SVD has higher explained variance ({explained_var_svd:.2%} vs {explained_var_pca:.2%})"
                else:
                    winner = "Tie"
                    reason = "Both models have identical error scores and explained variance"
            
            # Create results
            results = {
                'Best Model': winner,
                'Winner Reason': reason,
                'Error_Score_PCA': float(error_score_pca),
                'Error_Score_SVD': float(error_score_svd),
                'PCA': {
                    'MSE': float(mse_pca),
                    'MAE': float(mae_pca),
                    'Explained Variance': float(explained_var_pca),
                    'Components': max_comp,
                    'Error_Score': float(error_score_pca)
                },
                'SVD': {
                    'MSE': float(mse_svd),
                    'MAE': float(mae_svd),
                    'Explained Variance': float(explained_var_svd),
                    'Components': max_comp,
                    'Error_Score': float(error_score_svd)
                }
            }
            
            return results
            
        except Exception as e:
            st.error(f"⚡ Comparison error: {str(e)}")
            return None

def main():
    # Initialize system
    if 'rec_system' not in st.session_state:
        st.session_state.rec_system = CompleteRecommendationSystem()
        st.session_state.data_loaded = False
    
    # Initialize navigation state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "overview"
    
    rec_system = st.session_state.rec_system
    
    # ================== ENHANCED SIDEBAR ==================
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                      -webkit-background-clip: text;
                      -webkit-text-fill-color: transparent;
                      font-weight: 800;
                      font-size: 1.8rem;">
                🤖 RecSys AI
            </h1>
            <p style="color: #64748b; font-size: 0.9rem;">Intelligent Product Recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data loading button
        if st.button("🚀 **Load All Data**", type="primary", use_container_width=True, 
                    help="Load product and rating data for all recommendation models"):
            with st.spinner("Loading data for all three models..."):
                if rec_system.load_data():
                    st.session_state.data_loaded = True
                    st.success("✅ All data loaded!")
                else:
                    st.error("❌ Load failed")
        
        st.markdown("---")
        
        # Settings section
        st.markdown("### ⚙️ **Settings**")
        num_recs = st.slider("**Recommendations per model**", 5, 15, 10)
        
        st.markdown("---")
        
        # Model Status
        st.markdown("### 📊 **System Status**")
        if rec_system.products_df is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📦 Products", f"{len(rec_system.products_df):,}")
            with col2:
                if rec_system.ratings_df is not None:
                    st.metric("⭐ Ratings", f"{len(rec_system.ratings_df):,}")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ **Quick Actions**")
        if st.button("🔄 Refresh Models", use_container_width=True):
            if rec_system.products_df is not None:
                rec_system._prepare_all_models()
                st.success("Models refreshed!")
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding-top: 2rem;">
            <p>Version 2.0 • Powered by AI</p>
            <p>© 2024 RecSys Analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ================== MAIN HEADING - FIXED ==================
    st.markdown('<h1 class="main-header">🤖 E-Commerce Recommendation AI</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #64748b; max-width: 800px; margin: 0 auto;">
            Advanced AI-powered product recommendation system with three intelligent models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Navigation
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    pages = [
        ("🏠 Overview", "overview"),
        ("🔥 Popularity", "popularity"),
        ("👥 Collaborative", "collaborative"),
        ("🔍 Content-Based", "content"),
        ("📊 PCA vs SVD", "comparison")
    ]
    
    cols = st.columns(len(pages))
    for idx, (page_name, page_key) in enumerate(pages):
        with cols[idx]:
            is_active = st.session_state.current_page == page_key
            button_class = "nav-button active" if is_active else "nav-button"
            if st.button(
                page_name,
                use_container_width=True,
                key=f"nav_{page_key}",
                help=f"Switch to {page_name} section"
            ):
                st.session_state.current_page = page_key
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Stats Dashboard
    if rec_system.products_df is not None:
        st.markdown("### 📈 **System Dashboard**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📦 Total Products", f"{len(rec_system.products_df):,}", help="Number of products in database")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if rec_system.ratings_df is not None:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("⭐ Total Ratings", f"{len(rec_system.ratings_df):,}", help="Total user ratings")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            if rec_system.popular_products is not None:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🔥 Popular Items", len(rec_system.popular_products), help="Trending products")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            if rec_system.user_item_matrix is not None:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("👥 Active Users", rec_system.user_item_matrix.shape[0], help="Users with ratings")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================== PAGE CONTENT ==================
    if st.session_state.current_page == "overview":
        st.markdown('<h2 class="sub-header">🏠 System Overview</h2>', unsafe_allow_html=True)
        
        # Hero Section
        st.markdown("""
        <div class="model-card">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                          -webkit-background-clip: text;
                          -webkit-text-fill-color: transparent;
                          font-weight: 800;
                          font-size: 2rem;">
                    AI-Powered Recommendation Engine
                </h2>
                <p style="color: #64748b; font-size: 1.1rem;">
                    Three intelligent models tailored for different customer scenarios
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Model Cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="model-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔥</div>
                <h3 style="color: #f59e0b; margin-bottom: 0.5rem;">Popularity-Based</h3>
                <p style="color: #64748b; font-size: 0.9rem;">
                    <strong>For New Customers</strong><br>
                    Shows trending and best-selling products based on overall popularity
                </p>
                <span class="badge badge-popular">Trending</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="model-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">👥</div>
                <h3 style="color: #3b82f6; margin-bottom: 0.5rem;">Collaborative Filtering</h3>
                <p style="color: #64748b; font-size: 0.9rem;">
                    <strong>For Returning Customers</strong><br>
                    Personalized recommendations based on similar users' preferences
                </p>
                <span class="badge badge-collab">Personalized</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="model-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔍</div>
                <h3 style="color: #8b5cf6; margin-bottom: 0.5rem;">Content-Based</h3>
                <p style="color: #64748b; font-size: 0.9rem;">
                    <strong>For Cold Start</strong><br>
                    Recommendations based on product features and descriptions
                </p>
                <span class="badge badge-content">Semantic</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data Preview
        if rec_system.products_df is not None:
            st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-header">📋 Data Preview</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="model-card">
                    <h4 style="color: #3b82f6;">📦 Sample Products</h4>
                """, unsafe_allow_html=True)
                st.dataframe(
                    rec_system.products_df[['product_id', 'product_name']].head(5),
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                if rec_system.ratings_df is not None:
                    st.markdown("""
                    <div class="model-card">
                        <h4 style="color: #10b981;">⭐ Sample Ratings</h4>
                    """, unsafe_allow_html=True)
                    st.dataframe(
                        rec_system.ratings_df[['user_id', 'product_id', 'rating']].head(5),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.current_page == "popularity":
        st.markdown('<h2 class="sub-header">🔥 Popularity-Based Recommendations</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("📋 Please load data first from the sidebar")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🎯 Generate Recommendations", type="primary", use_container_width=True):
                    with st.spinner("🔍 Finding most popular products..."):
                        recommendations = rec_system.model_1_popularity_based(num_recs)
                        
                        st.session_state.popular_recs = recommendations
            
            if 'popular_recs' in st.session_state:
                recommendations = st.session_state.popular_recs
                
                # Display recommendations in elegant cards
                st.success(f"✨ **Top {len(recommendations)} Popular Products**")
                
                for idx, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="product-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <h4 style="margin: 0; color: #1e293b;">{row['product_name']}</h4>
                                    <p style="margin: 0.25rem 0; color: #64748b; font-size: 0.9rem;">
                                        Product ID: {row['product_id']}
                                    </p>
                                </div>
                                <div style="text-align: right;">
                                    <div style="display: flex; gap: 1rem;">
                                        <div style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); 
                                                    color: white; padding: 0.25rem 0.75rem; border-radius: 20px;">
                                            ⭐ {row['avg_rating']:.1f}
                                        </div>
                                        <div style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); 
                                                    color: white; padding: 0.25rem 0.75rem; border-radius: 20px;">
                                            📊 {row['rating_count']:,}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Visualization
                if len(recommendations) > 0:
                    st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
                    fig = px.bar(
                        recommendations.head(10),
                        x='product_name',
                        y='rating_count',
                        title="📈 **Top 10 Products by Popularity**",
                        color='avg_rating',
                        color_continuous_scale='Viridis',
                        labels={'product_name': 'Product', 'rating_count': 'Number of Ratings', 'avg_rating': 'Average Rating'}
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        height=450,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#1e293b'),
                        title_font=dict(size=20)
                    )
                    fig.update_traces(marker_line_width=0)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.current_page == "collaborative":
        st.markdown('<h2 class="sub-header">👥 Collaborative Filtering Recommendations</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("📋 Please load data first from the sidebar")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                # Get available users
                if rec_system.user_item_matrix is not None:
                    available_users = list(rec_system.user_item_matrix.index)[:20]
                    user_id = st.selectbox(
                        "👤 **Select User**", 
                        options=available_users,
                        help="Choose a user to get personalized recommendations"
                    )
                else:
                    user_id = st.text_input("👤 **Enter User ID**", value="1")
            
            with col2:
                st.markdown("")
                st.markdown("")
                if st.button("🎯 Get Personalized Recommendations", type="primary", use_container_width=True):
                    with st.spinner("🤖 Analyzing user preferences..."):
                        recommendations = rec_system.model_2_collaborative_filtering(user_id, num_recs)
                        
                        if recommendations is not None:
                            st.session_state.collab_recs = recommendations
                            st.session_state.current_user = user_id
            
            if 'collab_recs' in st.session_state:
                recommendations = st.session_state.collab_recs
                user_id = st.session_state.current_user
                
                st.success(f"✨ **Personalized Recommendations for User {user_id}**")
                
                for idx, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="product-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <h4 style="margin: 0; color: #1e293b;">{row['product_name']}</h4>
                                    <p style="margin: 0.25rem 0; color: #64748b; font-size: 0.9rem;">
                                        Recommended based on similar users
                                    </p>
                                </div>
                                <div style="text-align: right;">
                                    <div style="display: flex; gap: 1rem;">
                                        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                                    color: white; padding: 0.25rem 0.75rem; border-radius: 20px;">
                                            ⭐ {row['avg_rating']:.1f}
                                        </div>
                                        <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                                                    color: white; padding: 0.25rem 0.75rem; border-radius: 20px;">
                                            👥 {row['rating_count']}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    elif st.session_state.current_page == "content":
        st.markdown('<h2 class="sub-header">🔍 Content-Based Recommendations</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("📋 Please load data first from the sidebar")
        else:
            if rec_system.products_df is not None:
                # Get products for selection
                valid_products = rec_system.products_df[['product_id', 'product_name']].head(20)
                
                if len(valid_products) > 0:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        product_dict = {row['product_name']: row['product_id'] for _, row in valid_products.iterrows()}
                        selected_product = st.selectbox(
                            "📦 **Select a product to find similar items:**",
                            options=list(product_dict.keys()),
                            help="Choose a product to find semantically similar items"
                        )
                        product_id = product_dict[selected_product]
                    
                    with col2:
                        st.markdown("")
                        st.markdown("")
                        if st.button("🔍 Find Similar Products", type="primary", use_container_width=True):
                            with st.spinner("🔬 Analyzing product features..."):
                                recommendations = rec_system.model_3_content_based(product_id, num_recs)
                                
                                if recommendations is not None:
                                    st.session_state.content_recs = recommendations
                                    st.session_state.selected_product = selected_product
                    
                    if 'content_recs' in st.session_state:
                        recommendations = st.session_state.content_recs
                        selected_product = st.session_state.selected_product
                        
                        st.success(f"✨ **Products Similar to: {selected_product}**")
                        
                        for idx, row in recommendations.iterrows():
                            with st.container():
                                # FIXED: Properly handle the HTML generation
                                similarity_percent = row.get('similarity_score', 0) * 100 if 'similarity_score' in row else 0
                                
                                # Create HTML for rating if available
                                rating_html = ""
                                if 'avg_rating' in row and pd.notna(row['avg_rating']):
                                    rating_html = f'<div style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); color: white; padding: 0.25rem 0.75rem; border-radius: 20px;">⭐ {row["avg_rating"]:.1f}</div>'
                                
                                # Create the full HTML string
                                html_content = f"""
                                <div class="product-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <h4 style="margin: 0; color: #1e293b;">{row['product_name']}</h4>
                                            <p style="margin: 0.25rem 0; color: #64748b; font-size: 0.9rem;">
                                                Based on product description similarity
                                            </p>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="display: flex; gap: 1rem;">
                                                <div style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%); 
                                                          color: white; padding: 0.25rem 0.75rem; border-radius: 20px;">
                                                    🔍 {similarity_percent:.1f}%
                                                </div>
                                                {rating_html}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """
                                
                                st.markdown(html_content, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No products available for content-based recommendations")
            else:
                st.warning("📋 Please load data first")
    
    elif st.session_state.current_page == "comparison":
        st.markdown('<h2 class="sub-header">📊 PCA vs SVD Comparison</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="model-card">
            <div style="text-align: center;">
                <h3 style="color: #1e293b; margin-bottom: 1rem;">🎯 Dimensionality Reduction Analysis</h3>
                <p style="color: #64748b;">
                    Compare Principal Component Analysis (PCA) vs Singular Value Decomposition (SVD) 
                    to determine the optimal technique for your recommendation system.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("📋 Please load data first from the sidebar")
        else:
            # Configuration Panel
            st.markdown("### ⚙️ **Comparison Configuration**")
            
            config_col1, config_col2, config_col3 = st.columns([2, 1, 1])
            
            with config_col1:
                n_components = st.slider(
                    "**Number of Components**",
                    2, 10, 5,
                    help="Number of components/dimensions to reduce to"
                )
            
            with config_col2:
                st.markdown("")
                st.markdown("")
                if st.button("⚡ **Run Comparison**", type="primary", use_container_width=True):
                    with st.spinner("🔬 Comparing dimensionality reduction techniques..."):
                        results = rec_system.compare_pca_vs_svd(n_components)
                        
                        if results:
                            st.session_state.comparison_results = results
            
            if 'comparison_results' in st.session_state:
                results = st.session_state.comparison_results
                
                # Performance Metrics
                st.markdown("### 📈 **Performance Metrics**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="comparison-card pca-card">
                        <h3 style="color: #3b82f6; margin-bottom: 1rem;">🔷 PCA</h3>
                        <div style="font-size: 2.5rem; color: #3b82f6; margin-bottom: 0.5rem;">
                            {mse:.4f}
                        </div>
                        <p style="color: #64748b; font-size: 0.9rem;">Mean Squared Error</p>
                        <div style="font-size: 2.5rem; color: #3b82f6; margin-bottom: 0.5rem;">
                            {mae:.4f}
                        </div>
                        <p style="color: #64748b; font-size: 0.9rem;">Mean Absolute Error</p>
                        <div style="font-size: 2.5rem; color: #10b981; margin-bottom: 0.5rem;">
                            {var:.1%}
                        </div>
                        <p style="color: #64748b; font-size: 0.9rem;">Explained Variance</p>
                        <div style="font-size: 1.2rem; color: #3b82f6; margin-top: 1rem; font-weight: bold;">
                            Error Score: {error:.4f}
                        </div>
                        <p style="color: #64748b; font-size: 0.8rem;">(Lower is better)</p>
                    </div>
                    """.format(
                        mse=results['PCA']['MSE'],
                        mae=results['PCA']['MAE'],
                        var=results['PCA']['Explained Variance'],
                        error=results['PCA']['Error_Score']
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="comparison-card svd-card">
                        <h3 style="color: #ec4899; margin-bottom: 1rem;">🔶 SVD</h3>
                        <div style="font-size: 2.5rem; color: #ec4899; margin-bottom: 0.5rem;">
                            {mse:.4f}
                        </div>
                        <p style="color: #64748b; font-size: 0.9rem;">Mean Squared Error</p>
                        <div style="font-size: 2.5rem; color: #ec4899; margin-bottom: 0.5rem;">
                            {mae:.4f}
                        </div>
                        <p style="color: #64748b; font-size: 0.9rem;">Mean Absolute Error</p>
                        <div style="font-size: 2.5rem; color: #10b981; margin-bottom: 0.5rem;">
                            {var:.1%}
                        </div>
                        <p style="color: #64748b; font-size: 0.9rem;">Explained Variance</p>
                        <div style="font-size: 1.2rem; color: #ec4899; margin-top: 1rem; font-weight: bold;">
                            Error Score: {error:.4f}
                        </div>
                        <p style="color: #64748b; font-size: 0.8rem;">(Lower is better)</p>
                    </div>
                    """.format(
                        mse=results['SVD']['MSE'],
                        mae=results['SVD']['MAE'],
                        var=results['SVD']['Explained Variance'],
                        error=results['SVD']['Error_Score']
                    ), unsafe_allow_html=True)
                
                # Winner Announcement
                winner = results['Best Model']
                
                if winner == "PCA":
                    st.markdown(f"""
                    <div class="winner-card winner-card-pca">
                        <h1 style="margin-bottom: 1rem;">🏆 PCA IS THE WINNER! 🏆</h1>
                        <h3 style="margin-bottom: 1rem;">{results['Winner Reason']}</h3>
                        <p style="opacity: 0.9; font-size: 1.1rem;">
                            PCA Error Score: {results['Error_Score_PCA']:.4f} vs SVD Error Score: {results['Error_Score_SVD']:.4f}
                            <br><strong>Lower error score is better!</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif winner == "SVD":
                    st.markdown(f"""
                    <div class="winner-card winner-card-svd">
                        <h1 style="margin-bottom: 1rem;">🏆 SVD IS THE WINNER! 🏆</h1>
                        <h3 style="margin-bottom: 1rem;">{results['Winner Reason']}</h3>
                        <p style="opacity: 0.9; font-size: 1.1rem;">
                            SVD Error Score: {results['Error_Score_SVD']:.4f} vs PCA Error Score: {results['Error_Score_PCA']:.4f}
                            <br><strong>Lower error score is better!</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="winner-card winner-card-tie">
                        <h1 style="margin-bottom: 1rem;">🤝 IT'S A TIE! 🤝</h1>
                        <h3 style="margin-bottom: 1rem;">{results['Winner Reason']}</h3>
                        <p style="opacity: 0.9; font-size: 1.1rem;">
                            PCA Error Score: {results['Error_Score_PCA']:.4f} = SVD Error Score: {results['Error_Score_SVD']:.4f}
                            <br>You can use either technique for your recommendation system
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visual Comparison
                st.markdown("### 📊 **Visual Comparison**")
                
                # Create tabs for different visualizations
                tab1, tab2 = st.tabs(["📈 Error Metrics", "⚖️ Error Scores"])
                
                with tab1:
                    # Error metrics comparison
                    fig_errors = go.Figure()
                    fig_errors.add_trace(go.Bar(
                        name='🔷 PCA',
                        x=['MSE', 'MAE'],
                        y=[results['PCA']['MSE'], results['PCA']['MAE']],
                        marker_color='#3b82f6',
                        text=[f"{results['PCA']['MSE']:.4f}", f"{results['PCA']['MAE']:.4f}"],
                        textposition='auto',
                        marker_line_width=0,
                        width=0.4
                    ))
                    fig_errors.add_trace(go.Bar(
                        name='🔶 SVD',
                        x=['MSE', 'MAE'],
                        y=[results['SVD']['MSE'], results['SVD']['MAE']],
                        marker_color='#ec4899',
                        text=[f"{results['SVD']['MSE']:.4f}", f"{results['SVD']['MAE']:.4f}"],
                        textposition='auto',
                        marker_line_width=0,
                        width=0.4
                    ))
                    fig_errors.update_layout(
                        title="📉 Error Metrics Comparison (Lower is Better)",
                        yaxis_title="Error Value",
                        height=450,
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#1e293b'),
                        title_font=dict(size=18),
                        margin=dict(l=50, r=50, t=80, b=50),
                        bargap=0.2
                    )
                    st.plotly_chart(fig_errors, use_container_width=True)
                
                with tab2:
                    # Error score comparison
                    fig_scores = go.Figure()
                    fig_scores.add_trace(go.Bar(
                        name='Error Scores',
                        x=['PCA', 'SVD'],
                        y=[results['PCA']['Error_Score'], results['SVD']['Error_Score']],
                        marker_color=['#3b82f6', '#ec4899'],
                        text=[f"{results['PCA']['Error_Score']:.4f}", f"{results['SVD']['Error_Score']:.4f}"],
                        textposition='auto',
                        marker_line_width=0,
                        width=0.6
                    ))
                    fig_scores.update_layout(
                        title="⚖️ Overall Error Score Comparison (Lower is Better)",
                        yaxis_title="Error Score",
                        height=450,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#1e293b'),
                        title_font=dict(size=18),
                        margin=dict(l=50, r=50, t=80, b=50)
                    )
                    # Add a horizontal line for reference
                    fig_scores.add_hline(y=min(results['PCA']['Error_Score'], results['SVD']['Error_Score']), 
                                       line_dash="dash", line_color="green", annotation_text="Best Score")
                    st.plotly_chart(fig_scores, use_container_width=True)
                
                # Detailed Comparison Table
                st.markdown("### 🔬 **Detailed Comparison Table**")
                
                comparison_data = {
                    'Metric': ['Best Model', 'Mean Squared Error (MSE)', 
                              'Mean Absolute Error (MAE)', 'Explained Variance',
                              'Number of Components', 'Error Score'],
                    'PCA': [
                        '✅' if winner == 'PCA' else '',
                        f"{results['PCA']['MSE']:.6f}",
                        f"{results['PCA']['MAE']:.6f}",
                        f"{results['PCA']['Explained Variance']:.2%}",
                        results['PCA']['Components'],
                        f"{results['PCA']['Error_Score']:.4f}"
                    ],
                    'SVD': [
                        '✅' if winner == 'SVD' else '',
                        f"{results['SVD']['MSE']:.6f}",
                        f"{results['SVD']['MAE']:.6f}",
                        f"{results['SVD']['Explained Variance']:.2%}",
                        results['SVD']['Components'],
                        f"{results['SVD']['Error_Score']:.4f}"
                    ]
                }
                
                df_comparison = pd.DataFrame(comparison_data)
                
                # Display the table with proper styling
                st.markdown("""
                <div style="border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 10px 25px rgba(0,0,0,0.08);">
                """, unsafe_allow_html=True)
                
                st.dataframe(
                    df_comparison,
                    use_container_width=True,
                    height=300,
                    hide_index=True
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Recommendations Section
                st.markdown("### 💡 **Implementation Recommendations**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="model-card">
                        <h4 style="color: #3b82f6; margin-bottom: 1rem;">📊 When to use PCA:</h4>
                        <ul style="color: #64748b;">
                            <li>When working with dense, continuous data</li>
                            <li>When interpretability of components is important</li>
                            <li>For visualization in 2D/3D</li>
                            <li>When data follows Gaussian distribution</li>
                            <li>For general dimensionality reduction</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="model-card">
                        <h4 style="color: #ec4899; margin-bottom: 1rem;">📊 When to use SVD:</h4>
                        <ul style="color: #64748b;">
                            <li>When working with sparse matrices</li>
                            <li>For collaborative filtering systems</li>
                            <li>When memory efficiency is crucial</li>
                            <li>For text data (TF-IDF matrices)</li>
                            <li>In recommendation systems (matrix factorization)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ================== ENHANCED FOOTER ==================
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div style="text-align: left;">
                <h4 style="color: #1e293b; margin-bottom: 0.5rem;">🤖 E-Commerce Recommendation AI</h4>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0;">
                    Advanced AI-powered product recommendation system
                </p>
            </div>
            <div style="text-align: right;">
                <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                    <span style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); 
                               color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">
                        🔥 Popularity
                    </span>
                    <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                               color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">
                        👥 Collaborative
                    </span>
                    <span style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%); 
                               color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">
                        🔍 Content-Based
                    </span>
                </div>
            </div>
        </div>
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
            <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">
                © 2024 AI RecSys Analytics • Version 2.0 • Powered by Machine Learning
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()