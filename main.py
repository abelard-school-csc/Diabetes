import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import os

os.makedirs('images', exist_ok=True)

data = pd.read_csv('diabetes.csv')

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))

sns.histplot(data[data['Outcome'] == 1]['Glucose'], color='red', label='Diabetic', kde=True, stat='density')
sns.histplot(data[data['Outcome'] == 0]['Glucose'], color='blue', label='Non-Diabetic', kde=True, stat='density')
plt.legend()
plt.savefig('images/glucose_distribution.png')
plt.close()

t_stat, p_val = ttest_ind(data[data['Outcome'] == 1]['Glucose'], data[data['Outcome'] == 0]['Glucose'])
t_stat_bmi, p_val_bmi = ttest_ind(data[data['Outcome'] == 1]['BMI'], data[data['Outcome'] == 0]['BMI'])

correlations = data.corr()
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.savefig('images/correlation_heatmap.png')
plt.close()

x_data = data['BMI']
y_data = data['Glucose']
params, _ = curve_fit(sigmoid, x_data, y_data)
plt.scatter(x_data, y_data, label='Data')
plt.plot(np.sort(x_data), sigmoid(np.sort(x_data), *params), color='red', label='Sigmoid Fit')
plt.legend()
plt.savefig('images/sigmoid_fit.png')
plt.close()

X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend()
plt.savefig('images/roc_curve.png')
plt.close()
