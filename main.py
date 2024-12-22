# =================================Overview or project====================================================

# Load Traning and Test Data -> (pd.read_csv)

# Data Exploring
#     1.null values -> Data.isnull().sum()
#     2.Duplicate Value -> Data.Duplicated().sum()
#     3.Data Information -> Data.info()

# Cleaning the Data
#     1.Removing the un wanted Columns and rows  that will not use in predicting the   Final results
#     2. replacing the null value with then most occuring value of meadian value

# Preprocessing
#     1. Replacing catarogical data to Numerical data 
# Start Traning the Model by spliting train test data

# Selection best Regression Model

# Checking the accuracy Score

#================================================================================================================


# Importing needed packages
import pandas as pd

# Loading Training Data
Training_Path = r"C:\Users\z004fpbu\Desktop\jay\MyCodes\MyGIT_Code\Titanic_ML\Input\Titanic_train.csv"
Training_data = pd.read_csv(Training_Path)
print("Training Data:")
print(Training_data.head())

# Loading Test Data
test_data_path = r"C:\Users\z004fpbu\Desktop\jay\MyCodes\MyGIT_Code\Titanic_ML\Input\Titanic_Test.csv"
test_data = pd.read_csv(test_data_path)
print("\nTest Data:")
print(test_data.head())

#Exploring The data 
#1.Checking Null Values in Data
print(Training_data.isnull().sum())
#2. Checking Duplicated values in traning data 
print(Training_data.duplicated().sum())
#3. Checking traning Data info
print(Training_data.info())

#Data Cleaning 
#1.Removing the Unwanted Data Fields
def Clean(data):
    data=data.drop(["Ticket","Cabin","Name","Fare"],axis=1)
    #Filling the Value in age and Embearked data 
    #Embearked= as only 2 rows in Embearked column are blank so we can drop those 2 records
    #Age= as age is important for our result and have more missing values sum so in such case we will replace null values with mean on the age's
    data["Age"].fillna(data["Age"].median(),inplace=True)
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
    return data

Training_data= Clean(Training_data)
test_data= Clean(test_data)
print(Training_data.head())
print(test_data.isna().sum())

#Preporcessing  we use label encoding for handling caterogical value of sex,Embarked colums
from sklearn import preprocessing
le=preprocessing.LabelEncoder()

cols=["Sex","Embarked"]
for col in cols:
    Training_data[col]= le.fit_transform(Training_data[col])
    test_data[col]= le.fit_transform(test_data[col])
    print(le.classes_)
print(Training_data.head())

#Model Traning
from sklearn.model_selection import train_test_split
Y=Training_data["Survived"]
X=Training_data.drop(["Survived"],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#implementing Regression Model and fiting the Value on model
from sklearn.linear_model import LogisticRegression
Clf=LogisticRegression(random_state=0,max_iter=1000).fit(X_train,Y_train)
Y_pred=Clf.predict(X_test)

#checking the accuracy Score
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,Y_pred))

#Creating Final output file for Submition to kaggel 
Submission_Pred=Clf.predict(test_data)
df=pd.DataFrame({"PassengerId":test_data["PassengerId"],"Survived":Submission_Pred})
df.to_csv(r"C:\Users\z004fpbu\Desktop\jay\MyCodes\MyGIT_Code\Titanic_ML\Output\Titanic_Submission.csv",index=False)