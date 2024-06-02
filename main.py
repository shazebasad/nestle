
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import random
from load_data import DataProcessor
from data_labels import label_data_dict

processor = DataProcessor('Data V1.xlsx')
processor.clean_data()
processor.handle_missing_data()
processor.handle_duplicates('Customer ID')
processor.add_multiple_labels(label_data_dict)
