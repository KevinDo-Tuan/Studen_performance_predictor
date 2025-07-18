import kagglehub as kg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk


data = pd.read_csv(r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\neuralsorcerer\student-performance\versions\1\test.csv") # data 

data = data.dropna() # Remove rows with missing values

print (data)

