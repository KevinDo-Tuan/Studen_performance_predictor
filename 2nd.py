from sklearn.preprocessing import OneHotEncoder as od
import kagglehub as kg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.neural_network import MLPRegressor as mp

data = pd.read_csv(r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\neuralsorcerer\student-performance\versions\1\test.csv")
data.dropna(inplace=True)


encoder = od()
X_encoded = encoder.fit_transform(data)
print (X_encoded)