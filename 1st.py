import kagglehub as kg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.ensemble import HistGradientBoostingClassifier as mp
from sklearn.model_selection import train_test_split as trte
# data 
data = pd.read_csv(r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\neuralsorcerer\student-performance\versions\1\test.csv")
data1 =data.head(500)
data2 =data[500:1000]
data3 =data[1000:1500]
data4 =data[1500:2000]
data5 = data[2000:2500]
data6 = data[2500:3000]

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
    "Other": 4,}
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
    "public": 0,
    "private": 1,
}
data["SchoolType"] = data["SchoolType"].map(schooltype)
#place living

place = {
    "surburban": 0,
    "city": 1,
    "town": 3,
    "Rural": 4,
}
data["Locale"] = data["Locale"].map(place)

X = data.drop("GPA", axis=1) 
Y = data["GPA"]
X_train,X_test, Y_train, Y_test = trtey(X, Y, test_size=0.2, random_state=42)

def chat_showdata():

    
    print("start with the datset?")
    c = input("")
    if c.lower()[0] == "y": 
        print("Here is the dataset:")
        print(data)
        
        sns.pairplot(data1) 
        plt.show()
    
chat_showdata()
def train_model():
    
    model = mp
    pred = model.fit(X_train, Y_train)
    pred = model.predict(X_test)

    
train_model()






