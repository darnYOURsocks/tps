#!/usr/bin/env python3
"""
TPS ML Analytics Pipeline - Advanced Pattern Analysis
Machine learning system for analyzing TPS reasoning patterns and optimizing performance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import logging
import psycopg2
import redis
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import joblib
import optuna
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class TPSPattern:
    """Data structure for TPS reasoning patterns"""
    input_text: str
    tps_scores: Dict[str, float]
    domain: str
    success_score: float
    wave_stages: int
    processing_time: float
    configuration: str
    insights_generated: int
    user_satisfaction: Optional[float] = None
    failure_points: List[str] = field(default_factory=list)
    meta_observations: List[str] = field(default_factory=list)

@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    training_time: float
    feature_importance: Dict[str, float] = field(default_factory=dict)

class TPSDataCollector:
    """Collect and preprocess TPS reasoning data"""
    
    def __init__(self, db_config: Dict, redis_config: Dict):
        self.db_config = db_config
        self.redis_client = redis.Redis(**redis_config)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
    def collect_session_data(self, days_back: int = 30) -> pd.DataFrame:
        """Collect TPS session data from database"""
        
        try:
            conn = psycopg2.connect(**self.db_config)
            
            query = """
            SELECT 
                rs.id,
                rs.input_text,
                rs.tps_scores,
                rs.wave_progression,
                rs.insights,
                rs.success_metrics,
                rs.meta_observations,
                rs.configuration_used,
                rs.processing_time_ms,
                rs.created_at,
                u.username,
                u.role
            FROM reasoning_sessions rs
            LEFT JOIN users u ON rs.user_id = u.id
            WHERE rs.created_at >= %s
            ORDER BY rs.created_at DESC
            """
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            df = pd.read_sql(query, conn, params=[cutoff_date])
            conn.close()
            
            logger.info(f"Collected {len(df)} sessions from last {days_back} days")
            return df
            
        except Exception as e:
            logger.error(f"Database collection error: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw session data for ML"""
        
        processed_data = []
        
        for _, row in df.iterrows():
            try:
                # Parse TPS scores
                tps_scores = json.loads(row['tps_scores']) if row['tps_scores'] else {}
                
                # Parse success metrics
                success_metrics = json.loads(row['success_metrics']) if row['success_metrics'] else {}
                
                # Parse wave progression
                wave_data = json.loads(row['wave_progression']) if row['wave_progression'] else {}
                
                # Parse insights
                insights = json.loads(row['insights']) if row['insights'] else []
                
                # Extract features
                processed_row = {
                    'session_id': row['id'],
                    'input_text': row['input_text'],
                    'input_length': len(row['input_text']) if row['input_text'] else 0,
                    'word_count': len(row['input_text'].split()) if row['input_text'] else 0,
                    
                    # TPS scores
                    'emotional_score': self.extract_tps_score(tps_scores, 'E'),
                    'logical_score': self.extract_tps_score(tps_scores, 'L'),
                    'holding_score': self.extract_tps_score(tps_scores, 'H'),
                    'integration_score': self.extract_tps_score(tps_scores, 'I'),
                    
                    # Derived TPS features
                    'tps_balance': self.calculate_tps_balance(tps_scores),
                    'tps_total': sum([self.extract_tps_score(tps_scores, k) for k in ['E', 'L', 'H']]),
                    'dominant_sense': self.identify_dominant_sense(tps_scores),
                    
                    # Success metrics
                    'overall_success': success_metrics.get('overall_success', 0),
                    'wave_quality': success_metrics.get('wave_quality', 0),
                    'integration_quality': success_metrics.get('integration_quality', 0),
                    'meta_awareness': success_metrics.get('meta_awareness', 0),
                    
                    # Wave progression features
                    'stages_completed': wave_data.get('stages_completed', 0),
                    'wave_energy': wave_data.get('wave_energy', 0),
                    
                    # Processing features
                    'processing_time': row['processing_time_ms'] / 1000.0 if row['processing_time_ms'] else 0,
                    'configuration': row['configuration_used'] or 'default',
                    'insights_count': len(insights),
                    
                    # Temporal features
                    'hour_of_day': row['created_at'].hour,
                    'day_of_week': row['created_at'].weekday(),
                    'is_weekend': row['created_at'].weekday() >= 5,
                    
                    # User features
                    'user_role': row['role'] or 'user',
                    'username_hash': hash(row['username'] or 'anonymous') % 1000
                }
                
                processed_data.append(processed_row)
                
            except Exception as e:
                logger.warning(f"Error processing row {row.get('id', 'unknown')}: {str(e)}")
                continue
        
        return pd.DataFrame(processed_data)
    
    def extract_tps_score(self, tps_scores: Dict, sense: str) -> float:
        """Extract TPS score for specific sense"""
        if isinstance(tps_scores, dict) and sense in tps_scores:
            return float(tps_scores[sense])
        return 5.0  # Default neutral score
    
    def calculate_tps_balance(self, tps_scores: Dict) -> float:
        """Calculate TPS balance score (lower variance = better balance)"""
        scores = [self.extract_tps_score(tps_scores, k) for k in ['E', 'L', 'H']]
        return 1.0 - (np.var(scores) / 25.0)  # Normalized variance
    
    def identify_dominant_sense(self, tps_scores: Dict) -> str:
        """Identify dominant TPS sense"""
        scores = {
            'E': self.extract_tps_score(tps_scores, 'E'),
            'L': self.extract_tps_score(tps_scores, 'L'),
            'H': self.extract_tps_score(tps_scores, 'H')
        }
        return max(scores, key=scores.get)
    
    def extract_text_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract text features using TF-IDF"""
        texts = df['input_text'].fillna('').values
        return self.vectorizer.fit_transform(texts).toarray()

class TPSPatternAnalyzer:
    """Analyze TPS reasoning patterns using ML"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.cluster_model = None
        self.success_predictor = None
        self.configuration_optimizer = None
        
    def analyze_success_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns that lead to successful reasoning"""
        
        logger.info("Analyzing success patterns...")
        
        # Prepare features for success prediction
        feature_cols = [
            'input_length', 'word_count', 'emotional_score', 'logical_score',
            'holding_score', 'integration_score', 'tps_balance', 'tps_total',
            'hour_of_day', 'insights_count', 'processing_time'
        ]
        
        X = df[feature_cols].fillna(0)
        y = (df['overall_success'] > 0.7).astype(int)  # Binary success classification
        
        # Train Random Forest classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(rf_model, X, y, cv=5)
        
        rf_model.fit(X, y)
        self.models['success_classifier'] = rf_model
        
        # Feature importance
        importance = dict(zip(feature_cols, rf_model.feature_importances_))
        self.feature_importance['success_patterns'] = importance
        
        # Success rate by configuration
        config_success = df.groupby('configuration')['overall_success'].agg(['mean', 'count']).reset_index()
        
        # Success rate by dominant sense
        sense_success = df.groupby('dominant_sense')['overall_success'].agg(['mean', 'count']).reset_index()
        
        results = {
            'model_accuracy': scores.mean(),
            'feature_importance': importance,
            'config_performance': config_success.to_dict('records'),
            'sense_performance': sense_success.to_dict('records'),
            'overall_success_rate': df['overall_success'].mean()
        }
        
        logger.info(f"Success analysis complete. Accuracy: {scores.mean():.3f}")
        return results
    
    def cluster_reasoning_patterns(self, df: pd.DataFrame) -> Dict:
        """Cluster similar reasoning patterns"""
        
        logger.info("Clustering reasoning patterns...")
        
        # Feature engineering for clustering
        cluster_features = [
            'emotional_score', 'logical_score', 'holding_score',
            'tps_balance', 'input_length', 'processing_time',
            'insights_count', 'overall_success'
        ]
        
        X = df[cluster_features].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        
        # Optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Choose k with best silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        self.cluster_model = kmeans
        df['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = df.groupby('cluster').agg({
            'overall_success': ['mean', 'std', 'count'],
            'emotional_score': 'mean',
            'logical_score': 'mean',
            'holding_score': 'mean',
            'tps_balance': 'mean',
            'processing_time': 'mean',
            'insights_count': 'mean'
        }).round(3)
        
        results = {
            'optimal_clusters': optimal_k,
            'silhouette_score': silhouette_scores[optimal_k - 2],
            'cluster_analysis': cluster_analysis.to_dict(),
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
        
        logger.info(f"Clustering complete. Optimal clusters: {optimal_k}")
        return results
    
    def optimize_configurations(self, df: pd.DataFrame) -> Dict:
        """Optimize TPS configurations using ML"""
        
        logger.info("Optimizing configurations...")
        
        def objective(trial):
            """Optuna objective function for configuration optimization"""
            
            # Suggest hyperparameters for TPS configuration
            emotional_sensitivity = trial.suggest_float('emotional_sensitivity', 0.5, 2.0)
            logical_sensitivity = trial.suggest_float('logical_sensitivity', 0.5, 2.0)
            holding_sensitivity = trial.suggest_float('holding_sensitivity', 0.5, 2.0)
            
            domain_weights = {
                'chemistry': trial.suggest_float('chemistry_weight', 0.0, 1.0),
                'biology': trial.suggest_float('biology_weight', 0.0, 1.0),
                'psychology': trial.suggest_float('psychology_weight', 0.0, 1.0),
                'physics': trial.suggest_float('physics_weight', 0.0, 1.0)
            }
            
            # Normalize domain weights
            total_weight = sum(domain_weights.values())
            if total_weight > 0:
                domain_weights = {k: v/total_weight for k, v in domain_weights.items()}
            
            # Simulate configuration performance using historical data
            config_score = self.simulate_config_performance(
                df, emotional_sensitivity, logical_sensitivity, 
                holding_sensitivity, domain_weights
            )
            
            return config_score
        
        # Optimize using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        best_config = study.best_params
        best_score = study.best_value
        
        results = {
            'optimal_configuration': best_config,
            'expected_performance': best_score,
            'optimization_history': [(trial.value, trial.params) for trial in study.trials[:10]]
        }
        
        logger.info(f"Configuration optimization complete. Best score: {best_score:.3f}")
        return results
    
    def simulate_config_performance(self, df: pd.DataFrame, e_sens: float, 
                                   l_sens: float, h_sens: float, domain_weights: Dict) -> float:
        """Simulate configuration performance on historical data"""
        
        # This is a simplified simulation - in practice, you'd retrain the TPS engine
        performance_scores = []
        
        for _, row in df.sample(min(100, len(df))).iterrows():
            # Adjust TPS scores based on sensitivity
            adjusted_e = row['emotional_score'] * e_sens
            adjusted_l = row['logical_score'] * l_sens
            adjusted_h = row['holding_score'] * h_sens
            
            # Calculate balance with new scores
            adjusted_balance = 1.0 - (np.var([adjusted_e, adjusted_l, adjusted_h]) / 25.0)
            
            # Estimate success based on balance and other factors
            estimated_success = min(1.0, (adjusted_balance + row['overall_success']) / 2.0)
            performance_scores.append(estimated_success)
        
        return np.mean(performance_scores)
    
    def predict_session_outcome(self, session_features: Dict) -> Dict:
        """Predict outcome for a new reasoning session"""
        
        if 'success_classifier' not in self.models:
            return {'error': 'Success prediction model not trained'}
        
        model = self.models['success_classifier']
        
        # Prepare features
        feature_vector = np.array([[
            session_features.get('input_length', 0),
            session_features.get('word_count', 0),
            session_features.get('emotional_score', 5.0),
            session_features.get('logical_score', 5.0),
            session_features.get('holding_score', 5.0),
            session_features.get('integration_score', 5.0),
            session_features.get('tps_balance', 0.5),
            session_features.get('tps_total', 15.0),
            session_features.get('hour_of_day', 12),
            session_features.get('insights_count', 1),
            session_features.get('processing_time', 5.0)
        ]])
        
        # Predict success probability
        success_probability = model.predict_proba(feature_vector)[0][1]
        success_prediction = model.predict(feature_vector)[0]
        
        return {
            'success_probability': float(success_probability),
            'predicted_success': bool(success_prediction),
            'confidence': float(max(model.predict_proba(feature_vector)[0]))
        }

class TPSNeuralNetwork(nn.Module):
    """Deep learning model for TPS pattern analysis"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(TPSNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Output layer
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class TPSDeepAnalyzer:
    """Deep learning analyzer for TPS patterns"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        
    def train_deep_model(self, df: pd.DataFrame, epochs: int = 50) -> Dict:
        """Train deep neural network on TPS data"""
        
        logger.info("Training deep neural network...")
        
        # Prepare features
        feature_cols = [
            'input_length', 'word_count', 'emotional_score', 'logical_score',
            'holding_score', 'integration_score', 'tps_balance', 'hour_of_day',
            'processing_time', 'insights_count'
        ]
        
        X = df[feature_cols].fillna(0).values
        y = df['overall_success'].values.reshape(-1, 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Initialize model
        self.model = TPSNeuralNetwork(len(feature_cols)).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            predictions = self.model(X_train_tensor)
            loss = criterion(predictions, y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_test_tensor)
                val_loss = criterion(val_predictions, y_test_tensor)
                val_losses.append(val_loss.item())
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            test_predictions = self.model(X_test_tensor)
            test_mse = nn.MSELoss()(test_predictions, y_test_tensor).item()
            
        results = {
            'final_mse': test_mse,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        logger.info(f"Deep model training complete. Test MSE: {test_mse:.4f}")
        return results
    
    def extract_semantic_features(self, texts: List[str]) -> np.ndarray:
        """Extract semantic features using BERT"""
        
        logger.info("Extracting semantic features with BERT...")
        
        features = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.bert_model(**encoded)
                # Use [CLS] token embedding
                batch_features = outputs.last_hidden_state[:, 0, :].numpy()
                features.extend(batch_features)
        
        return np.array(features)

class TPSVisualizationGenerator:
    """Generate visualizations for TPS analytics"""
    
    def __init__(self):
        self.color_palette = {
            'emotional': '#ff6b6b',
            'logical': '#4ecdc4',
            'holding': '#45b7d1',
            'integration': '#96ceb4',
            'success': '#feca57',
            'failure': '#ff9ff3'
        }
    
    def create_tps_distribution_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create TPS score distribution visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Emotional Scores', 'Logical Scores', 'Holding Scores', 'Integration Scores'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        scores = ['emotional_score', 'logical_score', 'holding_score', 'integration_score']
        colors = [self.color_palette['emotional'], self.color_palette['logical'],
                 self.color_palette['holding'], self.color_palette['integration']]
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (score, color, pos) in enumerate(zip(scores, colors, positions)):
            fig.add_trace(
                go.Histogram(x=df[score], name=score.replace('_', ' ').title(), 
                           marker_color=color, opacity=0.7),
                row=pos[0], col=pos[1]
            )
        
        fig.update_layout(
            title="TPS Score Distributions",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def create_success_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap with success metrics"""
        
        correlation_cols = [
            'emotional_score', 'logical_score', 'holding_score', 'integration_score',
            'tps_balance', 'overall_success', 'wave_quality', 'processing_time',
            'insights_count', 'input_length'
        ]
        
        corr_matrix = df[correlation_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="TPS Feature Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600
        )
        
        return fig
    
    def create_performance_timeline(self, df: pd.DataFrame) -> go.Figure:
        """Create performance timeline visualization"""
        
        # Group by date
        df['date'] = df['created_at'] if 'created_at' in df.columns else pd.Timestamp.now()
        daily_stats = df.groupby(df['date'].dt.date).agg({
            'overall_success': ['mean', 'count'],
            'processing_time': 'mean',
            'insights_count': 'mean'
        }).round(3)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Success Rate Over Time', 'Processing Metrics'],
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
        )
        
        # Success rate
        fig.add_trace(
            go.Scatter(
                x=daily_stats.index,
                y=daily_stats[('overall_success', 'mean')],
                mode='lines+markers',
                name='Success Rate',
                line=dict(color=self.color_palette['success'])
            ),
            row=1, col=1
        )
        
        # Session count
        fig.add_trace(
            go.Bar(
                x=daily_stats.index,
                y=daily_stats[('overall_success', 'count')],
                name='Session Count',
                marker_color=self.color_palette['integration'],
                opacity=0.6
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Processing time
        fig.add_trace(
            go.Scatter(
                x=daily_stats.index,
                y=daily_stats[('processing_time', 'mean')],
                mode='lines+markers',
                name='Avg Processing Time',
                line=dict(color=self.color_palette['logical'])
            ),
            row=2, col=1
        )
        
        # Insights count
        fig.add_trace(
            go.Scatter(
                x=daily_stats.index,
                y=daily_stats[('insights_count', 'mean')],
                mode='lines+markers',
                name='Avg Insights',
                line=dict(color=self.color_palette['holding'])
            ),
            row=2, col=1, secondary_y=True
        )
        
        fig.update_layout(
            title="TPS Performance Timeline",
            height=800
        )
        
        return fig
    
    def create_cluster_visualization(self, df: pd.DataFrame) -> go.Figure:
        """Create 3D cluster visualization"""
        
        if 'cluster' not in df.columns:
            return go.Figure().add_annotation(text="No cluster data available")
        
        fig = go.Figure()
        
        unique_clusters = df['cluster'].unique()
        colors = px.colors.qualitative.Set3[:len(unique_clusters)]
        
        for i, cluster in enumerate(unique_clusters):
            cluster_data = df[df['cluster'] == cluster]
            
            fig.add_trace(go.Scatter3d(
                x=cluster_data['emotional_score'],
                y=cluster_data['logical_score'],
                z=cluster_data['holding_score'],
                mode='markers',
                marker=dict(
                    size=cluster_data['overall_success'] * 10,
                    color=colors[i],
                    opacity=0.7
                ),
                name=f'Cluster {cluster}',
                hovertemplate='<b>Cluster %{text}</b><br>' +
                              'E: %{x:.1f}<br>' +
                              'L: %{y:.1f}<br>' +
                              'H: %{z:.1f}<br>' +
                              'Success: %{marker.size}<extra></extra>',
                text=[cluster] * len(cluster_data)
            ))
        
        fig.update_layout(
            title="TPS Pattern Clusters (3D)",
            scene=dict(
                xaxis_title='Emotional Score',
                yaxis_title='Logical Score',
                zaxis_title='Holding Score'
            ),
            height=600
        )
        
        return fig

class TPSAnalyticsOrchestrator:
    """Main orchestrator for TPS analytics pipeline"""
    
    def __init__(self, db_config: Dict, redis_config: Dict):
        self.data_collector = TPSDataCollector(db_config, redis_config)
        self.pattern_analyzer = TPSPatternAnalyzer()
        self.deep_analyzer = TPSDeepAnalyzer()
        self.visualizer = TPSVisualizationGenerator()
        self.analysis_results = {}
        
    def run_complete_analysis(self, days_back: int = 30) -> Dict:
        """Run complete TPS analytics pipeline"""
        
        logger.info("Starting complete TPS analytics pipeline...")
        
        # Step 1: Collect and preprocess data
        raw_data = self.data_collector.collect_session_data(days_back)
        if raw_data.empty:
            return {'error': 'No data available for analysis'}
        
        processed_data = self.data_collector.preprocess_data(raw_data)
        
        # Step 2: Pattern analysis
        success_analysis = self.pattern_analyzer.analyze_success_patterns(processed_data)
        cluster_analysis = self.pattern_analyzer.cluster_reasoning_patterns(processed_data)
        optimization_results = self.pattern_analyzer.optimize_configurations(processed_data)
        
        # Step 3: Deep learning analysis
        if len(processed_data) > 100:  # Only run if sufficient data
            deep_results = self.deep_analyzer.train_deep_model(processed_data)
        else:
            deep_results = {'note': 'Insufficient data for deep learning analysis'}
        
        # Step 4: Generate visualizations
        visualizations = {
            'tps_distributions': self.visualizer.create_tps_distribution_plot(processed_data),
            'correlation_heatmap': self.visualizer.create_success_correlation_heatmap(processed_data),
            'performance_timeline': self.visualizer.create_performance_timeline(processed_data),
            'cluster_visualization': self.visualizer.create_cluster_visualization(processed_data)
        }
        
        # Step 5: Compile comprehensive results
        results = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_sessions': len(processed_data),
                'date_range': f"{days_back} days",
                'analysis_version': '6.0'
            },
            'data_summary': self.generate_data_summary(processed_data),
            'success_patterns': success_analysis,
            'cluster_analysis': cluster_analysis,
            'configuration_optimization': optimization_results,
            'deep_learning_results': deep_results,
            'visualizations': visualizations,
            'recommendations': self.generate_recommendations(
                success_analysis, cluster_analysis, optimization_results
            )
        }
        
        self.analysis_results = results
        logger.info("Complete analytics pipeline finished successfully")
        
        return results
    
    def generate_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the dataset"""
        
        return {
            'total_sessions': len(df),
            'success_rate': df['overall_success'].mean(),
            'avg_tps_scores': {
                'emotional': df['emotional_score'].mean(),
                'logical': df['logical_score'].mean(),
                'holding': df['holding_score'].mean(),
                'integration': df['integration_score'].mean()
            },
            'configuration_distribution': df['configuration'].value_counts().to_dict(),
            'dominant_sense_distribution': df['dominant_sense'].value_counts().to_dict(),
            'avg_processing_time': df['processing_time'].mean(),
            'avg_insights_per_session': df['insights_count'].mean()
        }
    
    def generate_recommendations(self, success_analysis: Dict, 
                               cluster_analysis: Dict, 
                               optimization_results: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Success pattern recommendations
        feature_importance = success_analysis.get('feature_importance', {})
        if feature_importance:
            top_feature = max(feature_importance, key=feature_importance.get)
            recommendations.append(
                f"Focus on improving {top_feature.replace('_', ' ')} as it has the highest impact on success"
            )
        
        # Configuration recommendations
        best_config = optimization_results.get('optimal_configuration', {})
        if best_config:
            recommendations.append(
                f"Consider updating default configuration with optimized parameters: "
                f"E_sens={best_config.get('emotional_sensitivity', 1.0):.2f}, "
                f"L_sens={best_config.get('logical_sensitivity', 1.0):.2f}, "
                f"H_sens={best_config.get('holding_sensitivity', 1.0):.2f}"
            )
        
        # Cluster-based recommendations
        optimal_clusters = cluster_analysis.get('optimal_clusters', 0)
        if optimal_clusters > 0:
            recommendations.append(
                f"Identified {optimal_clusters} distinct reasoning patterns. "
                f"Consider developing specialized configurations for each pattern."
            )
        
        # Performance recommendations
        overall_success = success_analysis.get('overall_success_rate', 0)
        if overall_success < 0.7:
            recommendations.append(
                "Overall success rate is below 70%. Focus on improving TPS balance and integration quality."
            )
        
        return recommendations
    
    def save_results(self, filepath: str):
        """Save analysis results to file"""
        
        # Convert Plotly figures to JSON for serialization
        serializable_results = self.analysis_results.copy()
        if 'visualizations' in serializable_results:
            for name, fig in serializable_results['visualizations'].items():
                serializable_results['visualizations'][name] = fig.to_json()
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Analysis results saved to {filepath}")

def main():
    """Main execution function"""
    
    # Configuration
    db_config = {
        'host': 'localhost',
        'database': 'tps_db',
        'user': 'tps_user',
        'password': 'tps_password',
        'port': 5432
    }
    
    redis_config = {
        'host': 'localhost',
        'port': 6379,
        'db': 0
    }
    
    # Initialize and run analytics
    orchestrator = TPSAnalyticsOrchestrator(db_config, redis_config)
    results = orchestrator.run_complete_analysis(days_back=30)
    
    # Save results
    orchestrator.save_results('tps_analytics_results.json')
    
    # Print summary
    if 'error' not in results:
        print("\n🧠 TPS Analytics Pipeline Complete!")
        print(f"📊 Analyzed {results['analysis_metadata']['total_sessions']} sessions")
        print(f"✅ Success Rate: {results['data_summary']['success_rate']:.2%}")
        print(f"🎯 Top Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
    else:
        print(f"❌ Analysis failed: {results['error']}")

if __name__ == "__main__":
    main()
