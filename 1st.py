import kagglehub as kg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.ensemble as mode
from sklearn.model_selection import train_test_split as trte
# data 
data = pd.read_csv(r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\neuralsorcerer\student-performance\versions\1\train.csv")
data1 =data.head(500)


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
    "Asian": 4,}
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

def chat_showdata():

    
    print("start with the dataset?")
    c = input("")
    if c.lower()[0] == "y": 
        print("Here is the dataset:")
        print(data)
        
        sns.pairplot(data1) 
        plt.show()
    
chat_showdata()

def train_model():
    
    model = mode.RandomForestRegressor()
    model.fit( X_train, Y_train)
    v = model.predict(X_test)
    print(v)

    
train_model()






