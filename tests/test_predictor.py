import pytest
import numpy as np
import pandas as pd
from src.world_cup_predictor import WorldCupDataGenerator, DataProcessor

def test_data_generation():
    """Vérifie que la génération de données synthétiques génère le bon format."""
    generator = WorldCupDataGenerator(seed=42)
    economic_df = generator.generate_economic_data()
    
    assert isinstance(economic_df, pd.DataFrame)
    assert not economic_df.empty
    assert 'gdp_per_capita_usd' in economic_df.columns
    assert 'country_name' in economic_df.columns

def test_data_cleaning_and_processing():
    """Vérifie que le processeur combine et nettoie correctement les données."""
    generator = WorldCupDataGenerator(seed=42)
    processor = DataProcessor(verbose=False)
    
    # Génération
    eco = generator.generate_economic_data()
    spo = generator.generate_sports_data()
    geo = generator.generate_geopolitical_data()
    fin = generator.generate_financial_data()
    
    # Merging & Cleaning
    combined = processor.combine_datasets(eco, spo, geo, fin)
    cleaned = processor.clean_data(combined)
    engineered = processor.engineer_features(cleaned)
    
    assert isinstance(engineered, pd.DataFrame)
    assert 'goal_differential' in engineered.columns  # Vérifie qu'une feature créée existe
    assert 'gdp_log' in engineered.columns