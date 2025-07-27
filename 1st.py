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

def finding_parameters(trial):
    hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", 10, 20)
    max_iter = trial.suggest_int("max_iter", 10, 20)
    random_state = trial.suggest_int("random_state", 10, 20)

    model = MLPRegressor(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=max_iter, random_state=random_state)
    model.fit(X_train, Y_train)
    v = model.predict(X_test)
    r2 = r2_score(Y_test, v)
    return r2


print ("finding parameters...")
study = optu.create_study(direction="maximize")
study.optimize(finding_parameters, n_trials=30)

print("Best parameters found:", study.best_params)








