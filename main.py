
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import random
from load_data import DataProcessor
from data_labels import label_data_dict
from descriptive import DescriptiveDetails


processor = DataProcessor('Data V1.xlsx')
processor.clean_data()
processor.handle_missing_data()
processor.handle_duplicates('Customer ID')
processor.add_multiple_labels(label_data_dict)

descriptive = DescriptiveDetails(processor.df)
descriptive.create_boxplots()
descriptive.create_histograms()

predictor = FeatureImportance(processor.df)
predictor.prepare_data()
predictor.train_and_evaluate_models()
predictor.model_evaluation()
predictor.feature_importance()
predictor.plot_feature_importance()
predictor.plot_beeswarm_shap_values()
predictor.plot_mean_shap_values()
predictor.plot_columns_shap_values('Income')
