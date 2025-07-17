import kagglehub as kg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv(r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\neuralsorcerer\student-performance\versions\1\test.csv")
data1 =data.head(500)

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
            
            sns.pairplot(data)  # start changing the dataset to plot it on sns
            plt.show()
    
chat_showdata()

def predicting():




