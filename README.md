Simple Linear Regression Model for Solubility Prediction
This project demonstrates a simple machine learning model using Linear Regression to predict the solubility (logS) of molecules based on their descriptors such as MolLogP, MolWt, NumRotatableBonds, and AromaticProportion.

Project Overview
The dataset used in this project consists of molecular descriptors and their corresponding solubility values. The goal is to build a model that can predict the solubility of molecules based on these features.

The steps include:

Data Loading: Import the dataset.
Data Preparation: Separate the features (X) and target variable (y).
Data Splitting: Split the data into training and testing sets.
Model Building: Use a linear regression model to train the data.
Model Evaluation: Evaluate the model's performance using Mean Squared Error (MSE) and R-squared (R2).
Dataset
The dataset used is the delaney_solubility_with_descriptors.csv, which includes the following columns:

MolLogP: Logarithm of the octanol-water partition coefficient.
MolWt: Molecular weight.
NumRotatableBonds: Number of rotatable bonds in the molecule.
AromaticProportion: Proportion of aromatic atoms in the molecule.
logS: Solubility in logarithmic scale (target variable).
Sample Data:
MolLogP	MolWt	NumRotatableBonds	AromaticProportion	logS
2.59540	167.85	0.0	0.0	-2.180
2.37650	133.41	0.0	0.0	-2.000
2.59380	167.85	1.0	0.0	-1.740
Steps to Run
1. Install Dependencies
Ensure that you have the required libraries installed. You can install them using the following command:

bash
Copy code
pip install pandas scikit-learn
2. Data Loading
The dataset is loaded directly from a URL using pandas. The code snippet to load the data is:

python
Copy code
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")
df
3. Data Preparation
We separate the features (X) from the target variable (y):

python
Copy code
y = df["logS"]
x = df.drop("logS", axis=1)
4. Data Splitting
The data is then split into training and testing sets:

python
Copy code
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
5. Model Building and Training
We use LinearRegression from sklearn to build and train the model:

python
Copy code
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
6. Model Predictions
Make predictions on both the training and testing datasets:

python
Copy code
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
7. Model Evaluation
Evaluate the model using Mean Squared Error (MSE) and R-squared (R2):

python
Copy code
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print('LR MSE (Train): ', lr_train_mse)
print('LR MSE (Test): ', lr_test_mse)
print('LR R2 (Train): ', lr_train_r2)
print('LR R2 (Test): ', lr_test_r2)
Results:
Method	Training MSE	Training R2	Test MSE	Test R2
Linear Regression	1.007536	0.764505	1.020695	0.759951
Conclusion
The Linear Regression model performs well with an R2 value of approximately 0.76 on both the training and testing datasets. This suggests that the model can predict the solubility of molecules with reasonable accuracy based on the given descriptors.
