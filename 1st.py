import kagglehub as kg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk

# data 
data = pd.read_csv(r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\neuralsorcerer\student-performance\versions\1\test.csv")
data1 =data.head(500)
data2 =data[500:1000]
data3 =data[1000:1500]
data4 =data[1500:2000]
data5 = data[2000:2500]
data6 = data[2500:3000]
data.dropna(inplace=True) # Remove rows with missing values


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

def chat_showdata():

    
    print("start with the datset?")
    c = input("")
    if c.lower()[0] == "y": 
        print("Here is the dataset:")
        print(data)
        sns.pairplot(data1) 
        sns.pairplot(data2)
        sns.pairplot(data3)
        sns.pairplot(data4)
        sns.pairplot(data5)
        sns.pairplot(data6)      # start changing the dataset to plot it on sns
        plt.show()
    
chat_showdata()
def train_model():
    X_train = data.drop("GPA")
    Y_train = data["GPA"]
    model = sk.random_forest.RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    
train_model()






