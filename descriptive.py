
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots
from load_data import DataProcessor


class DescriptiveDetails:
    def __init__(self, cleaned_data):
        self.df = cleaned_data

    def extract_columns(self):
        confectionary_2022 = self.df['Confectionary Shopping Value 2022']
        confectionary_2023 = self.df['Confectionary Shopping Value 2023']
        coffee_2022 = self.df['Coffee Shopping Value 2022']
        coffee_2023 = self.df['Coffee Shopping Value 2023']
        
        plot_data = pd.DataFrame({
            'Confectionary 2022': confectionary_2022,
            'Confectionary 2023': confectionary_2023,
            'Coffee 2022': coffee_2022,
            'Coffee 2023': coffee_2023
        })
        return plot_data

    def create_long_form_data(self, plot_data):
        long_form_data = pd.melt(plot_data.reset_index(), id_vars=['index'], value_vars=plot_data.columns)
        long_form_data.columns = ['index', 'Category', 'Value']
        return long_form_data

    def create_boxplots(self):
        plot_data = self.extract_columns()
        long_form_data = self.create_long_form_data(plot_data)
        
        fig = px.box(long_form_data, x='Category', y='Value', 
                     color='Category',
                     color_discrete_map={
                         'Confectionary 2022': '#9bcbeb',
                         'Confectionary 2023': '#9bcbeb',
                         'Coffee 2022': '#9cdbd9',
                         'Coffee 2023': '#9cdbd9'
                     })
    
        annotations = []
        for category in plot_data.columns:
            category_data = plot_data[category]
            
            median = category_data.median()
            mean = category_data.mean()
            quantiles = category_data.quantile([0.25, 0.75])
            
            annotations.append(dict(
                x=category,
                y=median,
                xref="x",
                yref="y",
                text=f"Median: {median:.2f}",
                ax=0,
                ay=-8
            ))
            
            annotations.append(dict(
                x=category,
                y=mean,
                xref="x",
                yref="y",
                text=f"Mean: {mean:.2f}",
                ax=0,
                ay=8
            ))
            
            annotations.append(dict(
                x=category,
                y=quantiles.iloc[0],
                xref="x",
                yref="y",
                text=f"Q1: {quantiles.iloc[0]:.2f}",
                ax=0,
                ay=15
            ))
            
            annotations.append(dict(
                x=category,
                y=quantiles.iloc[1],
                xref="x",
                yref="y",
                text=f"Q3: {quantiles.iloc[1]:.2f}",
                ax=0,
                ay=-15
            ))
    
        fig.update_layout(
            yaxis_title='Shopping Value',
            xaxis_title=None,
            showlegend=False,
            plot_bgcolor='white',
            yaxis=dict(
                showline=False,
                showgrid=True,
                zeroline=False,
                title=dict(
                    text='Shopping Value',
                    font=dict(size=14),
                    standoff=40  # Adjust standoff to position y-axis label further from the axis
                ),
                automargin=True,
            ),
            xaxis=dict(
                showline=False,
                showgrid=False,
                zeroline=False,
                tickangle=0,
                tickmode='array',
                tickvals=[0, 1, 2, 3],
                ticktext=['Confectionary 2022', 'Confectionary 2023', 'Coffee 2022', 'Coffee 2023'],
                automargin=False
            ),
            margin=dict(l=100, r=20, t=20, b=40),  # Adjust margins to create space for the y-axis label
            width=700,
            height=600,
            boxgap=0.1,
            boxgroupgap=0.1,
            annotations=annotations
        )
    
        fig.update_xaxes(showticklabels=True, ticks="")
        fig.update_yaxes(showticklabels=True, ticks="")
        
        fig.show()


    def create_histograms(self):
        
        def compute_stats(values):
            return {
                'mean': np.mean(values),
                'median': np.median(values),
                'percentile_25': np.percentile(values, 25),
                'percentile_75': np.percentile(values, 75)
            }
    
        stats_confectionary_2022 = compute_stats(self.df['Confectionary Shopping Value 2022'])
        stats_confectionary_2023 = compute_stats(self.df['Confectionary Shopping Value 2023'])
        stats_coffee_2022 = compute_stats(self.df['Coffee Shopping Value 2022'])
        stats_coffee_2023 = compute_stats(self.df['Coffee Shopping Value 2023'])
    
        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Confectionary Shopping", "Coffee Shopping"))
    
        # Confectionary Shopping Histograms and KDEs
        trace1 = go.Histogram(
            x=self.df['Confectionary Shopping Value 2022'],
            opacity=0.5,
            name='Confectionary 2022',
            marker=dict(color='#9bcbeb'),
            histnorm='probability density'
        )
    
        trace2 = go.Histogram(
            x=self.df['Confectionary Shopping Value 2023'],
            opacity=0.5,
            name='Confectionary 2023',
            marker=dict(color='#0067a0'),
            histnorm='probability density'
        )
    
        kde_confectionary_2022 = gaussian_kde(self.df['Confectionary Shopping Value 2022'])
        x_confectionary_2022 = np.linspace(self.df['Confectionary Shopping Value 2022'].min(), self.df['Confectionary Shopping Value 2022'].max(), 1000)
        y_confectionary_2022 = kde_confectionary_2022(x_confectionary_2022)
    
        trace3 = go.Scatter(
            x=x_confectionary_2022,
            y=y_confectionary_2022,
            mode='lines',
            name='KDE Confectionary 2022',
            line=dict(color='#9bcbeb')
        )
    
        kde_confectionary_2023 = gaussian_kde(self.df['Confectionary Shopping Value 2023'])
        x_confectionary_2023 = np.linspace(self.df['Confectionary Shopping Value 2023'].min(), self.df['Confectionary Shopping Value 2023'].max(), 1000)
        y_confectionary_2023 = kde_confectionary_2023(x_confectionary_2023)
    
        trace4 = go.Scatter(
            x=x_confectionary_2023,
            y=y_confectionary_2023,
            mode='lines',
            name='KDE Confectionary 2023',
            line=dict(color='#0067a0')
        )
    
        # Coffee Shopping Histograms and KDEs
        trace5 = go.Histogram(
            x=self.df['Coffee Shopping Value 2022'],
            opacity=0.5,
            name='Coffee 2022',
            marker=dict(color='#9cdbd9'),
            histnorm='probability density'
        )
    
        trace6 = go.Histogram(
            x=self.df['Coffee Shopping Value 2023'],
            opacity=0.5,
            name='Coffee 2023',
            marker=dict(color='#007681'),
            histnorm='probability density'
        )
    
        kde_coffee_2022 = gaussian_kde(self.df['Coffee Shopping Value 2022'])
        x_coffee_2022 = np.linspace(self.df['Coffee Shopping Value 2022'].min(), self.df['Coffee Shopping Value 2022'].max(), 1000)
        y_coffee_2022 = kde_coffee_2022(x_coffee_2022)
    
        trace7 = go.Scatter(
            x=x_coffee_2022,
            y=y_coffee_2022,
            mode='lines',
            name='KDE Coffee 2022',
            line=dict(color='#9cdbd9')
        )
    
        kde_coffee_2023 = gaussian_kde(self.df['Coffee Shopping Value 2023'])
        x_coffee_2023 = np.linspace(self.df['Coffee Shopping Value 2023'].min(), self.df['Coffee Shopping Value 2023'].max(), 1000)
        y_coffee_2023 = kde_coffee_2023(x_coffee_2023)
    
        trace8 = go.Scatter(
            x=x_coffee_2023,
            y=y_coffee_2023,
            mode='lines',
            name='KDE Coffee 2023',
            line=dict(color='#007681')
        )
    
        # Add traces to subplots
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=1)
        fig.add_trace(trace3, row=1, col=1)
        fig.add_trace(trace4, row=1, col=1)
        fig.add_trace(trace5, row=1, col=2)
        fig.add_trace(trace6, row=1, col=2)
        fig.add_trace(trace7, row=1, col=2)
        fig.add_trace(trace8, row=1, col=2)
    
        # Update layout
        fig.update_layout(
            width=1200,
            height=600,
            plot_bgcolor='white',  # Background color
            paper_bgcolor='white',  # Paper color
            xaxis=dict(showgrid=False),  # Remove x-axis grid lines
            yaxis=dict(showgrid=False),  # Remove y-axis grid lines
            xaxis2=dict(showgrid=False),  # Remove x-axis grid lines for second subplot
            yaxis2=dict(showgrid=False)  # Remove y-axis grid lines for second subplot
        )
    
        # Update subplot titles
        fig.layout.annotations[0].update(x=0.06, y=1)  # Adjust position of the first subplot title
        fig.layout.annotations[1].update(x=0.58, y=1)  # Adjust position of the second subplot title
    
        # Show the plot
        pyo.iplot(fig)
