# config/constants.py
"""
Global constants and configuration for World Cup predictor
Centralized definitions for all project parameters
"""

import os
from enum import Enum
from typing import Dict, List

# ============================================================================
# PROJECT METADATA
# ============================================================================
PROJECT_NAME = "World Cup Winner Predictor"
PROJECT_VERSION = "1.0.0"
AUTHOR = "Quantitative Development Team"

# ============================================================================
# DATA SOURCES & APIS
# ============================================================================
class DataSources(Enum):
    """Enumeration of data sources"""
    WORLD_BANK = "world_bank"
    IMF = "imf"
    STATSBOMB = "statsbomb"
    ESPN = "espn"
    TRANSFERMARKT = "transfermarkt"
    UN = "un"
    VDEM = "vdem"

# API Endpoints
API_ENDPOINTS = {
    "world_bank": "https://api.worldbank.org/v2",
    "imf": "https://data.imf.org/api/dataapi",
    "statsbomb": "https://raw.githubusercontent.com/statsbomb/data/master",
    "espn": "https://www.espn.com/soccer/",
}

# ============================================================================
# DATA SCHEMA DEFINITIONS
# ============================================================================
ECONOMIC_FEATURES = [
    "gdp_per_capita_usd",
    "government_spending_pct_gdp",
    "education_spending_pct_gdp",
    "military_spending_pct_gdp",
    "inflation_rate_pct",
    "unemployment_rate_pct",
    "gini_index",
    "fdi_inflow_usd_millions",
    "total_debt_pct_gdp",
    "trade_openness_index",
    "economic_stability_score",
    "economic_forecast_score",
]

GEOPOLITICAL_FEATURES = [
    "political_stability_index",
    "control_of_corruption_index",
    "rule_of_law_index",
    "regulatory_quality_index",
    "voice_accountability_index",
    "democratic_index",
    "regional_conflict_indicator",
    "border_dispute_history",
    "regional_power_dynamics",
    "sanctions_status",
]

SPORTS_FEATURES = [
    "elo_rating",
    "win_rate_pct",
    "goals_for_avg",
    "goals_against_avg",
    "clean_sheets_pct",
    "attacking_strength_index",
    "defensive_strength_index",
    "tournament_experience_score",
    "recent_form_score",
    "head_to_head_strength",
    "team_chemistry_score",
    "coaching_experience_score",
    "injury_severity_index",
    "squad_consistency_index",
    "confederation_rank",
]

FINANCIAL_FEATURES = [
    "squad_market_value_eur_millions",
    "avg_player_value_eur_millions",
    "youth_investment_eur_millions",
    "federation_budget_eur_millions",
    "sponsorship_revenue_eur_millions",
    "broadcast_revenue_eur_millions",
    "player_age_average",
    "player_experience_avg_caps",
]

# All features combined
ALL_FEATURES = (
    ECONOMIC_FEATURES + 
    GEOPOLITICAL_FEATURES + 
    SPORTS_FEATURES + 
    FINANCIAL_FEATURES
)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
class ModelConfig:
    """Machine learning model hyperparameters"""
    
    # XGBoost parameters
    XGBOOST_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
    }
    
    # Random Forest parameters
    RANDOM_FOREST_PARAMS = {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "oob_score": True,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    # Gradient Boosting parameters
    GRADIENT_BOOSTING_PARAMS = {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "subsample": 0.8,
        "max_features": "sqrt",
        "random_state": 42,
        "loss": "log_loss",
    }
    
    # Neural Network parameters
    NEURAL_NETWORK_PARAMS = {
        "hidden_layers": [256, 128, 64, 32],
        "dropout_rates": [0.3, 0.3, 0.2, 0.2],
        "activation": "relu",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "validation_split": 0.2,
        "early_stopping_patience": 15,
    }
    
    # Ensemble weights
    ENSEMBLE_WEIGHTS = {
        "xgboost": 0.35,
        "neural_network": 0.30,
        "random_forest": 0.20,
        "gradient_boosting": 0.15,
    }

# ============================================================================
# DATA PROCESSING CONFIGURATION
# ============================================================================
class DataConfig:
    """Data processing and validation parameters"""
    
    # Missing value thresholds
    MISSING_VALUE_THRESHOLD = 0.30  # Drop features with >30% missing
    MISSING_VALUE_IMPUTATION = "forward_fill"  # Strategy for imputation
    
    # Outlier detection
    OUTLIER_METHOD = "iqr"  # IQR or zscore
    IQR_MULTIPLIER = 1.5
    ZSCORE_THRESHOLD = 3.0
    
    # Feature scaling
    SCALING_METHOD = "robust"  # robust, standard, minmax
    
    # Data split ratios
    TRAIN_RATIO = 0.70
    VALIDATION_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Cross-validation
    CV_FOLDS = 5
    STRATIFIED = True

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
class EvaluationConfig:
    """Model evaluation and validation parameters"""
    
    # Performance thresholds
    ACCEPTABLE_ACCURACY = 0.75
    ACCEPTABLE_AUC = 0.80
    ACCEPTABLE_PRECISION = 0.70
    ACCEPTABLE_RECALL = 0.70
    
    # Calibration
    ECE_BINS = 10  # Expected Calibration Error bins
    MAX_ECE = 0.10  # Maximum acceptable ECE
    
    # Backtesting
    BACKTEST_YEARS = [2010, 2014, 2018, 2022]
    MIN_BACKTEST_ACCURACY = 0.70

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
class VisualizationConfig:
    """Plotting and dashboard parameters"""
    
    # Color schemes
    PRIMARY_COLOR = "#1f77b4"
    SECONDARY_COLOR = "#ff7f0e"
    SUCCESS_COLOR = "#2ca02c"
    ERROR_COLOR = "#d62728"
    WARNING_COLOR = "#ff9800"
    
    # Chart defaults
    DEFAULT_WIDTH = 1000
    DEFAULT_HEIGHT = 600
    FONT_SIZE = 12
    TITLE_FONT_SIZE = 16
    
    # SHAP plot settings
    SHAP_MAX_FEATURES = 20
    SHAP_PLOT_WIDTH = 1200
    SHAP_PLOT_HEIGHT = 600

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
class LoggingConfig:
    """Logging parameters"""
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "logs/worldcup_predictor.log"
    MAX_BYTES = 10485760  # 10MB
    BACKUP_COUNT = 5

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================
class OptimizationConfig:
    """Performance tuning parameters"""
    
    # Hyperparameter tuning
    OPTUNA_N_TRIALS = 100
    OPTUNA_TIMEOUT = 3600  # 1 hour
    OPTUNA_N_JOBS = -1
    
    # Parallel processing
    N_JOBS = -1  # Use all cores
    CHUNK_SIZE = 1000  # For processing large datasets
    
    # Caching
    CACHE_ENABLED = True
    CACHE_EXPIRY_DAYS = 7

# ============================================================================
# WORLD CUP TOURNAMENT DATA
# ============================================================================
class TournamentConfig:
    """World Cup tournament parameters"""
    
    # Standard format
    STANDARD_TEAMS = 32
    GROUPS = 8
    TEAMS_PER_GROUP = 4
    
    # Tournament stages
    STAGE_GROUP = "group"
    STAGE_KNOCKOUT = "knockout"
    
    # Participant history (countries that have qualified for 2026+)
    PREDICTED_PARTICIPANTS_2026 = [
        "Argentina", "Brazil", "France", "England", "Germany", "Spain", "Italy",
        "Netherlands", "Portugal", "Belgium", "Denmark", "Switzerland", "Sweden",
        "Austria", "Czech Republic", "Poland", "Ukraine", "Turkey", "Greece",
        "Serbia", "Croatia", "Romania", "Slovenia", "Slovakia", "Hungary",
        "Mexico", "USA", "Canada", "Costa Rica", "Panama", "Ecuador",
        "Uruguay", "Colombia", "Paraguay", "Chile", "Bolivia", "Peru",
        "Venezuela", "Japan", "South Korea", "Australia", "Iran",
        "Saudi Arabia", "UAE", "Morocco", "Senegal", "Ivory Coast",
        "Nigeria", "Ghana", "Egypt", "Tunisia", "Algeria"
    ]

# ============================================================================
# FILE PATHS
# ============================================================================
class FilePaths:
    """Directory and file path constants"""
    
    # Base directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")
    
    # Model directories
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Output directories
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
    REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
    VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, "visualizations")
    
    # Log directory
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    # Create directories if they don't exist
    @staticmethod
    def ensure_directories():
        """Create all required directories"""
        for path in [
            FilePaths.DATA_DIR,
            FilePaths.RAW_DATA_DIR,
            FilePaths.PROCESSED_DATA_DIR,
            FilePaths.EXTERNAL_DATA_DIR,
            FilePaths.MODELS_DIR,
            FilePaths.OUTPUTS_DIR,
            FilePaths.PREDICTIONS_DIR,
            FilePaths.REPORTS_DIR,
            FilePaths.VISUALIZATIONS_DIR,
            FilePaths.LOGS_DIR,
        ]:
            os.makedirs(path, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_feature_groups() -> Dict[str, List[str]]:
    """Get features organized by category"""
    return {
        "economic": ECONOMIC_FEATURES,
        "geopolitical": GEOPOLITICAL_FEATURES,
        "sports": SPORTS_FEATURES,
        "financial": FINANCIAL_FEATURES,
    }

def get_all_features() -> List[str]:
    """Get all features"""
    return ALL_FEATURES

def validate_features(features: List[str]) -> bool:
    """Validate that features are in allowed list"""
    return all(f in ALL_FEATURES for f in features)

# ============================================================================
# RUNTIME INITIALIZATION
# ============================================================================
if __name__ == "__main__":
    # Ensure all directories exist
    FilePaths.ensure_directories()
    
    # Print configuration summary
    print(f"{PROJECT_NAME} v{PROJECT_VERSION}")
    print(f"Features configured: {len(ALL_FEATURES)}")
    print(f"  - Economic: {len(ECONOMIC_FEATURES)}")
    print(f"  - Geopolitical: {len(GEOPOLITICAL_FEATURES)}")
    print(f"  - Sports: {len(SPORTS_FEATURES)}")
    print(f"  - Financial: {len(FINANCIAL_FEATURES)}")
