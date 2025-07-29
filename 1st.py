from xml.parsers.expat import model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.ensemble as mode
from sklearn.model_selection import train_test_split as trte
from sklearn.metrics import r2_score, mean_absolute_error

from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
import optuna as optu
# data 
data = pd.read_csv(r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\neuralsorcerer\student-performance\versions\1\train.csv")

data = data.dropna()# Remove rows with missing values

# Gender
gender = {
    "Female" :0,
    "Male" :1,

}
data["Gender"] = data["Gender"].map(gender)

#Race
race = {
    "Black": 0,
    "White": 1,
    "Hispanic": 2,
    "Two-or-more": 3,
    "Asian": 4,
    "Other": 5,}
data ["Race"] = data["Race"].map(race)

#SES_Quartile
#Parental education
edu= { 
    "HS": 0,
    "<HS": 1,
    "Bachelors+": 2,
    "SomeCollege": 3,
}
data["ParentalEducation"]= data["ParentalEducation"].map(edu)

#schooltype
schooltype = {
    "Public": 0,
    "Private": 1,
}
data["SchoolType"] = data["SchoolType"].map(schooltype)
#place living

place = {
    "Suburban": 0,
    "City": 1,
    "Town": 3,
    "Rural": 4,
}
data["Locale"] = data["Locale"].map(place)

X = data.drop("GPA", axis=1) 

Y = data["GPA"]

X_train,X_test, Y_train, Y_test = trte(X, Y, test_size=0.2, random_state=42)

def check_data():
    nan = data[data.isna().any(axis=1)]
    print("row with not a value:", nan)    
check_data() 

def train_model():
    model = MLPRegressor(hidden_layer_sizes=(14, ), max_iter=18, random_state=11)
    print("Training the neural network model...")
    model.fit(X_train, Y_train)
    print("data is trained")
    print("test data:", X_test)
    v = model.predict(X_test)
    print("prediction:", v)
    r2 = r2_score(Y_test, v)
    mae = mean_absolute_error(Y_test, v)
    print(f"R² score: {r2:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")

    # plot loss curve instead of tree
    plt.figure(figsize=(10, 5))
    plt.plot(model.loss_curve_)
    plt.title("MLPRegressor Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

train_model()










