
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

class ChurnPrediction:
    def __init__(self, data, cluster, threshold=0.1):
        self.data = data[data['Cluster_Gower'] == cluster]
        self.threshold = threshold
        self.model = None
        self.features = [
            'Education', 'Income', 'Gender', 'Family status', 'Age',
            'Confectionary Shopping Value 2022', 'Confectionary Shopping Value 2023',
            'Coffee Shopping Value 2022', 'Coffee Shopping Value 2023',
            'coffee_spending_change', 'coffee_spending_change_pct'
        ]
        self.target = 'churn'
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def preprocess_data(self):
        self.data['coffee_spending_change'] = self.data['Coffee Shopping Value 2023'] - self.data['Coffee Shopping Value 2022']
        self.data['coffee_spending_change_pct'] = (
            self.data['Coffee Shopping Value 2023'] - self.data['Coffee Shopping Value 2022']
        ) / self.data['Coffee Shopping Value 2022']

        self.data['churn'] = self.data['coffee_spending_change_pct'] < -self.threshold

        X = self.data[self.features]
        y = self.data[self.target]

        X = pd.get_dummies(X, columns=['Gender', 'Family status'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        print(classification_report(self.y_test, y_pred))
        print('AUC-ROC:', roc_auc_score(self.y_test, y_pred_proba))

    def identify_at_risk_customers(self, probability_threshold=0.1):
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        X_test_copy = self.X_test.copy()  # Create a copy of X_test before adding the new column
        X_test_copy['churn_proba'] = y_pred_proba
        at_risk_customers = X_test_copy[X_test_copy['churn_proba'] > probability_threshold]
        at_risk_customers['Customer ID'] = self.data.loc[at_risk_customers.index, 'Customer ID']
        return at_risk_customers

    def get_highest_churn_customer(self):
        at_risk_customers = self.identify_at_risk_customers()
        highest_churn_customer = at_risk_customers.loc[at_risk_customers['churn_proba'].idxmax()]
        return highest_churn_customer
