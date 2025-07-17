import kagglehub as kg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\neuralsorcerer\student-performance\versions\1\test.csv")
data = data.dropna("race")
data = data.head(500)

# Gender
gender = {
    "Female" :0,
    "Male" :1,

}
data["gender"] = data["gender"].map(gender)

