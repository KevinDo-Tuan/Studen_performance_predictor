import kagglehub as kg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# data 
data = pd.read_csv(r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\neuralsorcerer\student-performance\versions\1\test.csv")
data1 =data.head(500)
data2 =data[500:1000]
data3 =data[1000:1500]
data4 =data[1500:2000]
data5 = data[2000:2500]
data6 = data[2500:3000]

# Gender
gender = {
    "Female" :0,
    "Male" :1,

}
data["gender"] = data["gender"].map(gender)

#Race
race = {
    "Black": 0,
    "White": 1,
    "Hispanic": 2,
    "Two-or-more": 3,
    "Other": 4,}
data ["race"] = data["race"].map(race)
def chat_showdata():
    print ("can you describe your personality in 1 word so that I can understand you better, thank you?")
    a = input ("")

    if a is not float:
        print("Do you want to predict student performance?")
    else:
        print( "please enter a valid word")
    b = input("")
    
    transform = b.lower()
    
    if transform[0] == "y":
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






