import kagglehub as kg
import pandas as pd
import seaborn as sns
print ("can you describe your personality in 1 word so that I can understand you better, thank you?")
g = input ("")

if g is not float:
    print("Do you want to predict early signs of diabetes?")
else:
    print( "please enter a valid word")

chat = input("")

kg.dataset_download("alexteboul/diabetes-health-indicators-dataset")

file = pd.read_csv(r"c:\Users\Do Pham Tuan\.cache\kagglehub\datasets\alexteboul\diabetes-health-indicators-dataset\versions\1\diabetes_012_health_indicators_BRFSS2015.csv")
transform = chat.lower()

for i in transform:
    if i == "y":
        print("great, let's go.")
        print ("do you first want to see the data?")
        input("")
        for i in input(""):
            if i == "y":
                print(file)
                sns.pairplot(file)
            else: print("ok, let's continue")

    else:
        print("ok, have a nice day though")


