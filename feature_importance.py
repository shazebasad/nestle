
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
import shap

class FeatureImportance:
    def __init__(self, data):
        self.data = data
        self.encoder = OneHotEncoder(drop='first', sparse=False)
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest Regression': RandomForestRegressor(random_state=42),
            'Decision Tree Regression': DecisionTreeRegressor(random_state=42)
        }
        self.evaluation_metrics = {}
        self.feature_importances_summary = None

    def prepare_data(self):
        self.data['Total Shopping Value 2022'] = self.data['Confectionary Shopping Value 2022'] + self.data['Coffee Shopping Value 2022']
        self.data['Total Shopping Value 2023'] = self.data['Confectionary Shopping Value 2023'] + self.data['Coffee Shopping Value 2023']
        self.data['Total Shopping Value'] = self.data['Total Shopping Value 2022'] + self.data['Total Shopping Value 2023']

        features = ['Education', 'Income', 'Gender_Label', 'Family_Status_Label', 'Age']
        X = self.data[features]
        y = self.data['Total Shopping Value']

        X_encoded = self.encoder.fit_transform(X[['Gender_Label', 'Family_Status_Label']])
        X_encoded_df = pd.DataFrame(X_encoded, columns=self.encoder.get_feature_names_out(['Gender_Label', 'Family_Status_Label']))
        X = X.drop(['Gender_Label', 'Family_Status_Label'], axis=1)
        X = pd.concat([X, X_encoded_df], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_and_evaluate_models(self):
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            self.evaluation_metrics[model_name] = {'MAE': mae, 'MSE': mse, 'R2 Score': r2}

        rf_importances = self.models['Random Forest Regression'].feature_importances_
        dt_importances = self.models['Decision Tree Regression'].feature_importances_

        lr_importances = self.models['Linear Regression'].coef_


        self.feature_importances_summary = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance_random_forest': rf_importances,
            'Importance_decision_tree': dt_importances,
            'Importance_linear_regression': lr_importances,
        }).sort_values(by='Importance_random_forest', ascending=False)

    def model_evaluation(self):
        evaluation_df = pd.DataFrame(self.evaluation_metrics).T.reset_index().rename(columns={'index': 'Model'})
        display(evaluation_df)

    def feature_importance(self):
        display(self.feature_importances_summary)

    def plot_feature_importance(self):
        feature_importance_data = {
            'Random Forest Regression': self.feature_importances_summary['Importance_random_forest'],
            'Decision Tree Regression': self.feature_importances_summary['Importance_decision_tree'],
            'Linear Regression': self.feature_importances_summary['Importance_linear_regression']
        }
        
        # Define the starting colors for each model
        starting_colors = {
            'Random Forest Regression': '#0067a0',
            'Decision Tree Regression': '#9cdbd9',
            'Linear Regression': 'orange'
        }
    
        end_color = 'grey'
    
        for model_name, importances in feature_importance_data.items():
            sorted_features = self.feature_importances_summary[['Feature']].copy()
            sorted_features['Importance'] = importances
            sorted_features = sorted_features.sort_values(by='Importance', ascending=False)
            
            # Select the appropriate starting color
            start_color = starting_colors[model_name]
            
            # Generate a range of colors from the starting color to grey
            colors = [mcolors.to_hex(mcolors.to_rgba(np.array(mcolors.to_rgba(start_color)) * (1 - i/len(sorted_features)) + np.array(mcolors.to_rgba(end_color)) * (i/len(sorted_features)))) for i in range(len(sorted_features))]
            colors  # Reverse the color range
        
            fig = go.Figure(go.Bar(
                x=sorted_features['Importance'],
                y=sorted_features['Feature'],
                orientation='h',
                marker=dict(color=colors),
                text=sorted_features['Importance'].round(2),
                textposition='outside'
            ))
        
            fig.update_layout(
                title=f"Feature Importance for {model_name}",
                yaxis=dict(autorange='reversed', showgrid=False, zeroline=False),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  # Remove x-axis labels
                height=800,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
        
            fig.show()
    
    def plot_mean_shap_values(self):
        shap_values_data = {}
        bar_color = '#9cdbd9'  # RGB color for the bars
    
        for model_name, model in self.models.items():
            explainer = shap.Explainer(model, self.X_train)
            shap_values = explainer(self.X_test)
            shap_values_data[model_name] = shap_values
            
            shap.plots.bar(shap_values, show=False)
            plt.title(f'SHAP Values for {model_name}')
            
            # Get the current axes and customize the plot
            ax = plt.gca()
            
            # Set the color of the bars
            for bar in ax.containers[0]:
                bar.set_color(bar_color)
            
            # Set annotation color to black
            for annotation in ax.texts:
                annotation.set_color('black')
            
            # Remove x-axis labels and line
            ax.set_xticklabels([])
            ax.xaxis.set_visible(False)
            
            plt.show()


    def plot_beeswarm_shap_values(self, color_map=None):
        if color_map is None:
            colors = ["#0067a0", "#a50034"]  # Light blue and light green
            n_bins = 100  # Discretizes the interpolation into bins
            color_map = LinearSegmentedColormap.from_list('custom_blue_green', colors, N=n_bins)
    
        shap_values_data = {}
    
        for model_name, model in self.models.items():
            explainer = shap.Explainer(model, self.X_train)
            shap_values = explainer(self.X_test)
            shap_values_data[model_name] = shap_values
            
            # Set figure size (increase width)
            plt.figure(figsize=(12, 8))  # Adjust width (first value) and height (second value)
            
            shap.summary_plot(shap_values, self.X_test, show=False, cmap=color_map)
            
            # Adjust the size of the x labels
            plt.xticks(fontsize=5)  # Adjust the font size as needed
            
            plt.title(f'SHAP Values for {model_name}')
            plt.show()

    def plot_columns_shap_values(self, column):
        shap_values_data = {}
        
        for model_name, model in self.models.items():
            explainer = shap.Explainer(model, self.X_train)
            shap_values = explainer(self.X_test)
            shap_values_data[model_name] = shap_values
            
            shap.plots.scatter(shap_values[:,column],  show=False)
            plt.title(f'SHAP Values for {model_name}')
            plt.show()
