# src/world_cup_predictor.py
"""
World Cup Winner Predictor: Complete Implementation
Production-grade machine learning pipeline for World Cup prediction
"""

import os
import sys
import warnings
import logging
import pickle
import json
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1: DATA GENERATION & COLLECTION
# ============================================================================

class WorldCupDataGenerator:
    """
    Generates synthetic training data based on real World Cup patterns.
    In production, replace with actual API calls to data sources.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize data generator"""
        self.seed = seed
        np.random.seed(seed)
        
    def generate_economic_data(self, n_countries: int = 50, 
                              years: List[int] = None) -> pd.DataFrame:
        """
        Generate economic indicators
        
        Args:
            n_countries: Number of countries to generate
            years: List of years for data
            
        Returns:
            DataFrame with economic indicators
        """
        if years is None:
            years = [2010, 2014, 2018, 2022]
            
        countries = self._get_sample_countries(n_countries)
        data = []
        
        for year in years:
            for country in countries:
                data.append({
                    'country_code': self._country_to_code(country),
                    'country_name': country,
                    'year': year,
                    'gdp_per_capita_usd': np.random.lognormal(10, 1.5),
                    'government_spending_pct_gdp': np.random.uniform(10, 50),
                    'education_spending_pct_gdp': np.random.uniform(2, 8),
                    'military_spending_pct_gdp': np.random.uniform(0.5, 5),
                    'inflation_rate_pct': np.random.uniform(-2, 10),
                    'unemployment_rate_pct': np.random.uniform(2, 15),
                    'gini_index': np.random.uniform(25, 65),
                    'fdi_inflow_usd_millions': np.random.lognormal(15, 2),
                    'total_debt_pct_gdp': np.random.uniform(20, 150),
                    'trade_openness_index': np.random.uniform(30, 150),
                    'economic_stability_score': np.random.uniform(0, 100),
                    'economic_forecast_score': np.random.uniform(40, 100),
                })
                
        return pd.DataFrame(data)
    
    def generate_sports_data(self, n_countries: int = 50,
                            years: List[int] = None) -> pd.DataFrame:
        """
        Generate sports performance metrics
        
        Args:
            n_countries: Number of countries
            years: List of years for data
            
        Returns:
            DataFrame with sports metrics
        """
        if years is None:
            years = [2010, 2014, 2018, 2022]
            
        countries = self._get_sample_countries(n_countries)
        data = []
        
        for year in years:
            for country in countries:
                # Bias stronger nations to have better stats
                strength_bias = np.random.uniform(0, 1)
                
                data.append({
                    'country_code': self._country_to_code(country),
                    'country_name': country,
                    'year': year,
                    'elo_rating': np.random.uniform(1400, 2200) + strength_bias * 500,
                    'win_rate_pct': np.random.uniform(30, 80) + strength_bias * 20,
                    'goals_for_avg': np.random.uniform(0.5, 2.5) + strength_bias * 0.5,
                    'goals_against_avg': np.random.uniform(0.5, 2.0) - strength_bias * 0.3,
                    'clean_sheets_pct': np.random.uniform(20, 60) + strength_bias * 20,
                    'attacking_strength_index': np.random.uniform(40, 100) + strength_bias * 10,
                    'defensive_strength_index': np.random.uniform(40, 100) + strength_bias * 10,
                    'tournament_experience_score': np.random.uniform(20, 100),
                    'recent_form_score': np.random.uniform(30, 100),
                    'head_to_head_strength': np.random.uniform(40, 100),
                    'team_chemistry_score': np.random.uniform(30, 90) + strength_bias * 15,
                    'coaching_experience_score': np.random.uniform(30, 95),
                    'injury_severity_index': np.random.uniform(0, 50),
                    'squad_consistency_index': np.random.uniform(40, 95),
                    'confederation_rank': np.random.randint(1, 50),
                })
                
        return pd.DataFrame(data)
    
    def generate_geopolitical_data(self, n_countries: int = 50,
                                  years: List[int] = None) -> pd.DataFrame:
        """
        Generate geopolitical indicators
        
        Args:
            n_countries: Number of countries
            years: List of years for data
            
        Returns:
            DataFrame with geopolitical factors
        """
        if years is None:
            years = [2010, 2014, 2018, 2022]
            
        countries = self._get_sample_countries(n_countries)
        data = []
        
        for year in years:
            for country in countries:
                data.append({
                    'country_code': self._country_to_code(country),
                    'country_name': country,
                    'year': year,
                    'political_stability_index': np.random.uniform(-2.5, 2.5),
                    'control_of_corruption_index': np.random.uniform(-2.5, 2.5),
                    'rule_of_law_index': np.random.uniform(-2.5, 2.5),
                    'regulatory_quality_index': np.random.uniform(-2.5, 2.5),
                    'voice_accountability_index': np.random.uniform(-2.5, 2.5),
                    'democratic_index': np.random.uniform(0, 10),
                    'regional_conflict_indicator': np.random.uniform(0, 10),
                    'border_dispute_history': np.random.randint(0, 3),
                    'regional_power_dynamics': np.random.uniform(0, 100),
                    'sanctions_status': np.random.choice([0, 1], p=[0.9, 0.1]),
                })
                
        return pd.DataFrame(data)
    
    def generate_financial_data(self, n_countries: int = 50,
                               years: List[int] = None) -> pd.DataFrame:
        """
        Generate financial metrics
        
        Args:
            n_countries: Number of countries
            years: List of years for data
            
        Returns:
            DataFrame with financial data
        """
        if years is None:
            years = [2010, 2014, 2018, 2022]
            
        countries = self._get_sample_countries(n_countries)
        data = []
        
        for year in years:
            for country in countries:
                squad_value = np.random.lognormal(15, 1.5)
                
                data.append({
                    'country_code': self._country_to_code(country),
                    'country_name': country,
                    'year': year,
                    'squad_market_value_eur_millions': squad_value,
                    'avg_player_value_eur_millions': squad_value / 25,
                    'youth_investment_eur_millions': np.random.lognormal(12, 1.5),
                    'federation_budget_eur_millions': np.random.lognormal(13, 1.5),
                    'sponsorship_revenue_eur_millions': np.random.lognormal(12, 1.8),
                    'broadcast_revenue_eur_millions': np.random.lognormal(11, 2),
                    'player_age_average': np.random.uniform(26, 32),
                    'player_experience_avg_caps': np.random.uniform(30, 120),
                })
                
        return pd.DataFrame(data)
    
    def generate_target_variable(self, countries: List[str],
                                year: int) -> Dict[str, int]:
        """
        Generate target variable (tournament winner)
        In production, this would be actual tournament results
        
        Args:
            countries: List of participating countries
            year: Tournament year
            
        Returns:
            Dictionary mapping country to winner status
        """
        targets = {country: 0 for country in countries}
        # Randomly assign winner (in production, use actual results)
        winner_idx = np.random.randint(0, len(countries))
        targets[countries[winner_idx]] = 1
        return targets
    
    @staticmethod
    def _get_sample_countries(n: int = 50) -> List[str]:
        """Get sample countries"""
        all_countries = [
            "Argentina", "Brazil", "France", "Germany", "Spain", "England",
            "Italy", "Netherlands", "Portugal", "Belgium", "Denmark", "Sweden",
            "Poland", "Switzerland", "Austria", "Czech Republic", "Mexico",
            "USA", "Canada", "Uruguay", "Colombia", "Chile", "Japan",
            "South Korea", "Australia", "Iran", "Saudi Arabia", "Egypt",
            "Nigeria", "Senegal", "Morocco", "Turkey", "Ukraine", "Russia",
            "Greece", "Romania", "Hungary", "Serbia", "Croatia", "Slovenia",
            "Slovakia", "Ireland", "Wales", "Scotland", "Iceland", "Norway",
            "Finland", "New Zealand", "Costa Rica", "Panama"
        ]
        return all_countries[:n]
    
    @staticmethod
    def _country_to_code(country: str) -> str:
        """Convert country name to ISO code"""
        codes = {
            "Argentina": "ARG", "Brazil": "BRA", "France": "FRA",
            "Germany": "GER", "Spain": "ESP", "England": "ENG",
            "Italy": "ITA", "Netherlands": "NED", "Portugal": "POR",
            "Belgium": "BEL", "Denmark": "DEN", "Sweden": "SWE",
            "Mexico": "MEX", "USA": "USA", "Canada": "CAN",
        }
        return codes.get(country, country[:3].upper())


# ============================================================================
# SECTION 2: DATA PROCESSING & FEATURE ENGINEERING
# ============================================================================

class DataProcessor:
    """
    Comprehensive data processing pipeline including cleaning and feature engineering
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize data processor"""
        self.verbose = verbose
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
    def combine_datasets(self, economic_df: pd.DataFrame,
                        sports_df: pd.DataFrame,
                        geopolitical_df: pd.DataFrame,
                        financial_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all data sources into single dataset
        
        Args:
            economic_df: Economic indicators
            sports_df: Sports metrics
            geopolitical_df: Geopolitical factors
            financial_df: Financial data
            
        Returns:
            Combined DataFrame
        """
        # Merge on country_code and year
        merged = economic_df.copy()
        merged = merged.merge(sports_df, on=['country_code', 'country_name', 'year'])
        merged = merged.merge(geopolitical_df, on=['country_code', 'country_name', 'year'])
        merged = merged.merge(financial_df, on=['country_code', 'country_name', 'year'])
        
        if self.verbose:
            logger.info(f"Combined dataset shape: {merged.shape}")
            
        return merged
    
    def clean_data(self, df: pd.DataFrame, 
                   missing_threshold: float = 0.3) -> pd.DataFrame:
        """
        Clean data: handle missing values, duplicates, outliers
        
        Args:
            df: Input DataFrame
            missing_threshold: Drop columns with > threshold missing values
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        
        if self.verbose:
            logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index
        
        if self.verbose:
            logger.info(f"Dropping columns with >{missing_threshold*100}% missing: {list(cols_to_drop)}")
        
        df = df.drop(columns=cols_to_drop)
        
        # Fill remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Remove outliers (IQR method)
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        if self.verbose:
            logger.info(f"Data shape after cleaning: {df.shape}")
            
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from raw data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Economic feature engineering
        df['gdp_log'] = np.log1p(df['gdp_per_capita_usd'])
        df['economic_power'] = (
            df['gdp_per_capita_usd'] * df['government_spending_pct_gdp'] / 100
        )
        df['education_investment_intensity'] = (
            df['education_spending_pct_gdp'] / df['government_spending_pct_gdp']
        )
        df['fdi_momentum'] = df['fdi_inflow_usd_millions'].diff().fillna(0)
        
        # Sports feature engineering
        df['goal_differential'] = df['goals_for_avg'] - df['goals_against_avg']
        df['offensive_efficiency'] = df['goals_for_avg'] / (df['goals_for_avg'] + df['goals_against_avg'] + 1)
        df['team_strength_index'] = (
            df['attacking_strength_index'] + df['defensive_strength_index']
        ) / 2
        df['form_momentum'] = df['recent_form_score'] / (df['tournament_experience_score'] + 1)
        
        # Financial feature engineering
        df['squad_value_log'] = np.log1p(df['squad_market_value_eur_millions'])
        df['player_value_variance'] = (
            df['squad_market_value_eur_millions'] / (df['avg_player_value_eur_millions'] * 25)
        )
        df['federation_investment_intensity'] = (
            df['youth_investment_eur_millions'] / df['federation_budget_eur_millions']
        )
        
        # Geopolitical feature engineering
        df['governance_quality'] = (
            df['control_of_corruption_index'] + 
            df['rule_of_law_index'] + 
            df['regulatory_quality_index']
        ) / 3
        df['stability_composite'] = (
            df['political_stability_index'] + df['governance_quality']
        ) / 2
        
        # Interaction features
        df['economic_sports_interaction'] = df['gdp_log'] * df['team_strength_index']
        df['stability_investment_interaction'] = df['stability_composite'] * df['squad_value_log']
        
        if self.verbose:
            logger.info(f"Created {len(df.columns) - 45} new engineered features")
            
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame, 
                            target_col: Optional[str] = None,
                            fit_scaler: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for model training: scaling, encoding
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            fit_scaler: Whether to fit or use existing scaler
            
        Returns:
            Tuple of (features_array, target_array)
        """
        df = df.copy()
        
        # Separate features and target
        feature_cols = [col for col in df.columns 
                       if col not in ['country_code', 'country_name', 'year', target_col]]
        
        X = df[feature_cols].copy()
        y = df[target_col] if target_col else None
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        if fit_scaler:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Set fit_scaler=True on first call.")
            X_scaled = self.scaler.transform(X)
        
        if self.verbose:
            logger.info(f"Features prepared: {X_scaled.shape}")
            
        return X_scaled, y.values if y is not None else None


# ============================================================================
# SECTION 3: MODEL TRAINING & EVALUATION
# ============================================================================

class WorldCupPredictorEnsemble:
    """
    Ensemble model combining XGBoost, Random Forest, Gradient Boosting, and Neural Networks
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """Initialize ensemble model"""
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Individual models
        self.xgb_model = None
        self.rf_model = None
        self.gb_model = None
        self.nn_model = None
        
        # Ensemble
        self.ensemble = None
        
        # Performance tracking
        self.cv_scores = {}
        self.test_metrics = {}
        
    def build_xgboost_model(self, **kwargs) -> None:
        """Build XGBoost classifier"""
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }
        default_params.update(kwargs)
        
        self.xgb_model = xgb.XGBClassifier(**default_params)
        logger.info("XGBoost model initialized")
    
    def build_random_forest_model(self, **kwargs) -> None:
        """Build Random Forest classifier"""
        default_params = {
            'n_estimators': 500,
            'max_depth': 15,
            'min_samples_split': 2,
            'max_features': 'sqrt',
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }
        default_params.update(kwargs)
        
        self.rf_model = RandomForestClassifier(**default_params)
        logger.info("Random Forest model initialized")
    
    def build_gradient_boosting_model(self, **kwargs) -> None:
        """Build Gradient Boosting classifier"""
        default_params = {
            'n_estimators': 300,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'random_state': self.random_state,
        }
        default_params.update(kwargs)
        
        self.gb_model = GradientBoostingClassifier(**default_params)
        logger.info("Gradient Boosting model initialized")
    
    def build_neural_network_model(self, input_dim: int, **kwargs) -> None:
        """
        Build neural network model
        
        Args:
            input_dim: Number of input features
            **kwargs: Additional model parameters
        """
        layers_config = kwargs.get('hidden_layers', [256, 128, 64, 32])
        dropout_rates = kwargs.get('dropout_rates', [0.3, 0.3, 0.2, 0.2])
        
        model = keras.Sequential([
            layers.Dense(layers_config[0], activation='relu', 
                        input_dim=input_dim,
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rates[0]),
        ])
        
        # Add hidden layers
        for units, dropout in zip(layers_config[1:], dropout_rates[1:]):
            model.add(layers.Dense(units, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.001)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC', 'accuracy']
        )
        
        self.nn_model = model
        logger.info("Neural Network model initialized")
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    verbose: bool = True) -> None:
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Whether to print training progress
        """
        # Train XGBoost
        logger.info("Training XGBoost...")
        self.xgb_model.fit(X_train, y_train, 
                          eval_set=[(X_val, y_val)],
                          verbose=False)
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        
        # Train Gradient Boosting
        logger.info("Training Gradient Boosting...")
        self.gb_model.fit(X_train, y_train)
        
        # Train Neural Network
        logger.info("Training Neural Network...")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        self.nn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )
        
        logger.info("All models trained successfully")
    
    def create_ensemble(self) -> None:
        """Create voting ensemble from trained models"""
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgb', self.xgb_model),
                ('rf', self.rf_model),
                ('gb', self.gb_model),
            ],
            voting='soft',
            weights=[0.35, 0.20, 0.15],
            n_jobs=self.n_jobs
        )
        
        logger.info("Ensemble model created")
    
    def get_ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble predictions combining all models
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        # Sklearn models
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        rf_probs = self.rf_model.predict_proba(X)[:, 1]
        gb_probs = self.gb_model.predict_proba(X)[:, 1]
        
        # Neural network predictions
        nn_probs = self.nn_model.predict(X, verbose=0).flatten()
        
        # Weighted average
        ensemble_probs = (
            0.35 * xgb_probs +
            0.20 * rf_probs +
            0.15 * gb_probs +
            0.30 * nn_probs
        )
        
        return ensemble_probs
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                model_name: str = "ensemble") -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            model_name: Name of model to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model_name == "ensemble":
            y_pred_proba = self.get_ensemble_predictions(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif model_name == "xgb":
            y_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
            y_pred = self.xgb_model.predict(X_test)
        elif model_name == "rf":
            y_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
            y_pred = self.rf_model.predict(X_test)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
        }
        
        self.test_metrics[model_name] = metrics
        
        logger.info(f"{model_name.upper()} Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names: List[str],
                              n_top: int = 20) -> pd.DataFrame:
        """
        Get feature importances from XGBoost model
        
        Args:
            feature_names: List of feature names
            n_top: Number of top features to return
            
        Returns:
            DataFrame of feature importances
        """
        importance_scores = self.xgb_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores,
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(n_top)


# ============================================================================
# SECTION 4: VISUALIZATION & ANALYSIS
# ============================================================================

class VisualizationManager:
    """
    Create publication-quality visualizations and interactive dashboards
    """
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame,
                               title: str = "Feature Importance (XGBoost)",
                               n_features: int = 20) -> go.Figure:
        """
        Create feature importance plot
        
        Args:
            importance_df: DataFrame with features and importance scores
            title: Plot title
            n_features: Number of features to display
            
        Returns:
            Plotly figure
        """
        df_top = importance_df.head(n_features).sort_values('importance')
        
        fig = go.Figure(data=[
            go.Bar(
                y=df_top['feature'],
                x=df_top['importance'],
                orientation='h',
                marker=dict(color=df_top['importance'], 
                           colorscale='Viridis',
                           showscale=False)
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=600,
            template="plotly_white",
        )
        
        return fig
    
    @staticmethod
    def plot_roc_curve(y_test: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "Model") -> go.Figure:
        """
        Create ROC curve
        
        Args:
            y_test: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of model
            
        Returns:
            Plotly figure
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        fig = go.Figure(data=[
            go.Scatter(x=fpr, y=tpr, mode='lines',
                      name=f'{model_name} (AUC={auc:.3f})',
                      line=dict(width=2, color='#1f77b4')),
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      name='Random Classifier',
                      line=dict(dash='dash', color='gray'))
        ])
        
        fig.update_layout(
            title=f"ROC Curve - {model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=600,
            template="plotly_white",
        )
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray,
                             model_name: str = "Model") -> go.Figure:
        """
        Create confusion matrix heatmap
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            model_name: Name of model
            
        Returns:
            Plotly figure
        """
        cm = confusion_matrix(y_test, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix - {model_name}",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=500,
            template="plotly_white",
        )
        
        return fig
    
    @staticmethod
    def plot_prediction_distribution(y_pred_proba: np.ndarray,
                                    title: str = "Prediction Probability Distribution") -> go.Figure:
        """
        Plot distribution of predictions
        
        Args:
            y_pred_proba: Predicted probabilities
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[
            go.Histogram(x=y_pred_proba, nbinsx=50, name='Probability',
                        marker=dict(color='#1f77b4'))
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Prediction Probability",
            yaxis_title="Count",
            height=500,
            template="plotly_white",
        )
        
        return fig
    
    @staticmethod
    def plot_model_comparison(metrics_dict: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Compare metrics across models
        
        Args:
            metrics_dict: Dictionary of model names to metrics
            
        Returns:
            Plotly figure
        """
        models = list(metrics_dict.keys())
        metric_names = list(metrics_dict[models[0]].keys())
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=metric_names)
        
        for i, metric in enumerate(metric_names[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            for model in models:
                value = metrics_dict[model].get(metric, 0)
                fig.add_trace(
                    go.Bar(name=model, x=[model], y=[value]),
                    row=row, col=col
                )
            
            fig.update_yaxes(title_text=metric.upper(), row=row, col=col, range=[0, 1])
        
        fig.update_layout(height=800, showlegend=True, title_text="Model Performance Comparison")
        
        return fig


# ============================================================================
# SECTION 5: COMPLETE PIPELINE ORCHESTRATION
# ============================================================================

class WorldCupPredictorPipeline:
    """
    End-to-end pipeline orchestrating all components
    """
    
    def __init__(self):
        """Initialize pipeline"""
        self.data_generator = WorldCupDataGenerator(seed=42)
        self.data_processor = DataProcessor(verbose=True)
        self.model_ensemble = WorldCupPredictorEnsemble()
        self.visualizer = VisualizationManager()
        
        logger.info("Pipeline initialized")
    
    def run_complete_pipeline(self) -> Dict:
        """
        Run complete pipeline from data generation to prediction
        
        Returns:
            Dictionary with all results
        """
        logger.info("=" * 80)
        logger.info("STARTING WORLD CUP PREDICTOR PIPELINE")
        logger.info("=" * 80)
        
        # ---- STEP 1: DATA GENERATION ----
        logger.info("\n[STEP 1] Generating synthetic training data...")
        economic_data = self.data_generator.generate_economic_data()
        sports_data = self.data_generator.generate_sports_data()
        geopolitical_data = self.data_generator.generate_geopolitical_data()
        financial_data = self.data_generator.generate_financial_data()
        
        # ---- STEP 2: DATA COMBINATION & CLEANING ----
        logger.info("\n[STEP 2] Combining and cleaning data...")
        combined_data = self.data_processor.combine_datasets(
            economic_data, sports_data, geopolitical_data, financial_data
        )
        cleaned_data = self.data_processor.clean_data(combined_data)
        
        # ---- STEP 3: FEATURE ENGINEERING ----
        logger.info("\n[STEP 3] Engineering features...")
        engineered_data = self.data_processor.engineer_features(cleaned_data)
        
        # ---- STEP 4: CREATE TARGET VARIABLE ----
        logger.info("\n[STEP 4] Creating target variable...")
        # For demo: randomly assign winners based on team strength
        engineered_data['is_winner'] = 0
        for country in engineered_data['country_code'].unique():
            country_data = engineered_data[engineered_data['country_code'] == country]
            # Bias towards stronger teams (higher ELO)
            strengths = country_data.groupby('country_code')['elo_rating'].mean().values
            winner_idx = np.argmax(strengths) if len(strengths) > 0 else 0
            engineered_data.loc[country_data.index[winner_idx], 'is_winner'] = 1
        
        # ---- STEP 5: DATA PREPARATION ----
        logger.info("\n[STEP 5] Preparing data for modeling...")
        X, y = self.data_processor.prepare_for_modeling(
            engineered_data, target_col='is_winner', fit_scaler=True
        )
        
        # ---- STEP 6: TRAIN-TEST SPLIT ----
        logger.info("\n[STEP 6] Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # ---- STEP 7: BUILD MODELS ----
        logger.info("\n[STEP 7] Building models...")
        self.model_ensemble.build_xgboost_model()
        self.model_ensemble.build_random_forest_model()
        self.model_ensemble.build_gradient_boosting_model()
        self.model_ensemble.build_neural_network_model(input_dim=X_train.shape[1])
        
        # ---- STEP 8: TRAIN MODELS ----
        logger.info("\n[STEP 8] Training models...")
        self.model_ensemble.train_models(X_train, y_train, X_val, y_val, verbose=False)
        
        # ---- STEP 9: CREATE ENSEMBLE ----
        logger.info("\n[STEP 9] Creating ensemble...")
        self.model_ensemble.create_ensemble()
        
        # ---- STEP 10: EVALUATE MODELS ----
        logger.info("\n[STEP 10] Evaluating models...")
        xgb_metrics = self.model_ensemble.evaluate(X_test, y_test, model_name='xgb')
        rf_metrics = self.model_ensemble.evaluate(X_test, y_test, model_name='rf')
        ensemble_metrics = self.model_ensemble.evaluate(X_test, y_test, model_name='ensemble')
        
        # ---- STEP 11: MAKE PREDICTIONS ----
        logger.info("\n[STEP 11] Making predictions...")
        y_pred_proba = self.model_ensemble.get_ensemble_predictions(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # ---- STEP 12: FEATURE IMPORTANCE ----
        logger.info("\n[STEP 12] Computing feature importance...")
        importance_df = self.model_ensemble.get_feature_importance(
            self.data_processor.feature_names, n_top=20
        )
        
        logger.info("\nTop 10 Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # ---- STEP 13: VISUALIZATIONS ----
        logger.info("\n[STEP 13] Creating visualizations...")
        
        results = {
            'data': {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
            },
            'metrics': {
                'xgb': xgb_metrics,
                'rf': rf_metrics,
                'ensemble': ensemble_metrics,
            },
            'predictions': {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
            },
            'feature_importance': importance_df,
            'feature_names': self.data_processor.feature_names,
            'visualizations': {
                'feature_importance': self.visualizer.plot_feature_importance(importance_df),
                'roc_curve': self.visualizer.plot_roc_curve(y_test, y_pred_proba, 'Ensemble'),
                'confusion_matrix': self.visualizer.plot_confusion_matrix(y_test, y_pred, 'Ensemble'),
                'prediction_distribution': self.visualizer.plot_prediction_distribution(y_pred_proba),
                'model_comparison': self.visualizer.plot_model_comparison({
                    'XGBoost': xgb_metrics,
                    'Random Forest': rf_metrics,
                    'Ensemble': ensemble_metrics,
                }),
            },
            'model': self.model_ensemble,
            'processor': self.data_processor,
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Execute complete World Cup Predictor pipeline
    """
    # Initialize and run pipeline
    pipeline = WorldCupPredictorPipeline()
    results = pipeline.run_complete_pipeline()
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 80)
    
    for model_name, metrics in results['metrics'].items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric.upper():15s}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("TOP 10 FEATURE IMPORTANCE")
    print("=" * 80)
    print(results['feature_importance'].head(10).to_string(index=False))
    
    # Save visualizations
    print("\n\nSaving visualizations...")
    output_dir = "./outputs/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    for viz_name, fig in results['visualizations'].items():
        fig.write_html(f"{output_dir}/{viz_name}.html")
        print(f"  ✓ Saved {viz_name}.html")
