import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import statsmodels.api as sm

# Load the data
df = pd.read_csv('C:\\Users\\Admin\\Desktop\\prog\\python\\Smarket.csv')

# Prepare the feature matrix and target vector
X = df[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
y = (df['Direction'] == 'Up').astype(int)  # Convert 'Direction' to binary format

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Print the coefficients
print('Coefficients:', model.coef_)

# Calculate and print the standard errors, test statistics, and p-values
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())

# Predict the probabilities and convert to class labels
probabilities = model.predict_proba(X)
labels = np.where(probabilities[:,1] > 0.5, 'Up', 'Down')

# Print the predicted probabilities and class labels for the first ten observations
print('Predicted probabilities:', probabilities[:10])
print('Class labels:', labels[:10])

# Convert 'labels' to binary format
labels_binary = (labels == 'Up').astype(int)

# Evaluate the model using a confusion matrix and compute the metrics
cm = confusion_matrix(y, labels_binary)
accuracy = accuracy_score(y, labels_binary)
precision = precision_score(y, labels_binary)
recall = recall_score(y, labels_binary)
f1 = f1_score(y, labels_binary)

print('Confusion matrix:', cm)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)




