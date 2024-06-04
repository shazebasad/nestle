
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import random

# Importing custom modules
from load_data import DataProcessor
from data_labels import label_data_dict
from descriptive import DescriptiveDetails
from feature_importance import FeatureImportance
from segmentation import ClusterSegment
from churn import ChurnPrediction

# Data Processing
processor = DataProcessor('Data V1.xlsx')
processor.clean_data()
processor.handle_missing_data()
processor.handle_duplicates('Customer ID')
processor.add_multiple_labels(label_data_dict)

# Descriptive Analysis
descriptive = DescriptiveDetails(processor.df)
descriptive.create_boxplots()
descriptive.create_histograms()

# Feature Importance Analysis
predictor = FeatureImportance(processor.df)
predictor.prepare_data()
predictor.train_and_evaluate_models()
predictor.model_evaluation()
predictor.feature_importance()
predictor.plot_feature_importance()
predictor.plot_beeswarm_shap_values()
predictor.plot_mean_shap_values()
predictor.plot_columns_shap_values('Income')

# Clustering Analysis
clustering_features = [
    'Age', 'Income', 'Family status', 
    'Confectionary Shopping Value 2022', 'Confectionary Shopping Value 2023', 
    'Coffee Shopping Value 2022', 'Coffee Shopping Value 2023'
]
analyzer = ClusterSegment(processor.df, clustering_features)
segment_profiles_gower = analyzer.run_analysis()
display(segment_profiles_gower)

# Churn Prediction
data = processor.df  
cluster = 2
churn_prediction = ChurnPrediction(data, cluster)
churn_prediction.preprocess_data()
churn_prediction.train_model()
churn_prediction.evaluate_model()
at_risk_customers = churn_prediction.identify_at_risk_customers()
highest_churn_customer = churn_prediction.get_highest_churn_customer()
print(highest_churn_customer)
