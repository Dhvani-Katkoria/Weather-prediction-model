
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import math 
from sklearn.preprocessing import LabelEncoder#,LabelBinarizer

import time
import warnings
warnings.filterwarnings("ignore")


# In[143]:


data = pd.read_csv('/home/vidhikatkoria/Downloads/weather-dataset/weatherAUS.csv')

print("Total no.of points = {}".format(data.shape[0]))
data.head(5)


# In[144]:


data.drop_duplicates(inplace=True)


# In[145]:


data.isnull().any()


# In[146]:


data.isnull().sum() * 100 / len(data)  #calculate the percentage of missing data in each column 


# In[147]:


''' DECISION:
drop Evaporation, Sunshine, Cloud9am, Cloud3pm out of the data'''
data = data.drop( ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1 )


# In[148]:


sns.set(style="whitegrid")
sns.countplot(data.RainTomorrow)
plt.title("Target labels")
plt.show()


# In[149]:


#Separating the data based on its class label.
data_yes = data[data['RainTomorrow']=='Yes']
data_no = data[data['RainTomorrow']=='No']


# In[150]:


#Observing the mode for all columns when RainTomorrow = Yes  
mode_values_for_yes = data_yes.mode()
mode_values_for_yes


# In[151]:


#Observing the mode for all columns when RainTomorrow = No  
mode_values_for_no = data_no.mode()
mode_values_for_no


# In[157]:


#For Temparatures we cannot replace NaN values with 0, hence replacing NaN with its respective mode value
data_yes['MinTemp'].fillna(value=data_yes['MinTemp'].mode()[0],inplace=True )
data_no['MinTemp'].fillna(value=data_no['MinTemp'].mode()[0],inplace=True )

data_yes['MaxTemp'].fillna(value=data_yes['MaxTemp'].mode()[0],inplace=True )
data_no['MaxTemp'].fillna(value=data_no['MaxTemp'].mode()[0],inplace=True )


data_yes['Temp9am'].fillna(value=data_yes['Temp9am'].mode()[0],inplace=True )
data_no['Temp9am'].fillna(value=data_no['Temp9am'].mode()[0],inplace=True )

data_yes['Temp3pm'].fillna(value=data_yes['Temp3pm'].mode()[0],inplace=True )
data_no['Temp3pm'].fillna(value=data_no['Temp3pm'].mode()[0],inplace=True )


# For humidity also 
data_yes['Humidity9am'].fillna(value=data_yes['Humidity9am'].mode()[0],inplace=True )
data_no['Humidity9am'].fillna(value=data_no['Humidity9am'].mode()[0],inplace=True )



data_yes['Humidity3pm'].fillna(value=data_yes['Humidity3pm'].mode()[0],inplace=True )
data_no['Humidity3pm'].fillna(value=data_no['Humidity3pm'].mode()[0],inplace=True )

# For the rain fall feature we can replace NaN with 0.0 which says there is no rain fall
data_yes['Rainfall'].fillna(value=1.0,inplace=True)
data_no['Rainfall'].fillna(value=0.0,inplace=True)


data_yes['Pressure9am'].fillna(value=data_yes['Pressure9am'].median(),inplace=True )
data_no['Pressure9am'].fillna(value=data_no['Pressure9am'].median(),inplace=True )

data_yes['Pressure3pm'].fillna(value=data_yes['Pressure3pm'].median(),inplace=True )
data_no['Pressure3pm'].fillna(value=data_no['Pressure3pm'].median(),inplace=True )


data_yes['WindSpeed9am'].fillna(value=data_yes['WindSpeed9am'].median(),inplace=True )
data_no['WindSpeed9am'].fillna(value=data_no['WindSpeed9am'].median(),inplace=True )

data_yes['WindSpeed3pm'].fillna(value=data_yes['WindSpeed3pm'].median(),inplace=True )
data_no['WindSpeed3pm'].fillna(value=data_no['WindSpeed3pm'].median(),inplace=True )

#WindGustSpeed -- replacing with median
data_yes['WindGustSpeed'].fillna(value=data_yes['WindGustSpeed'].median(),inplace=True)
data_no['WindGustSpeed'].fillna(value=data_no['WindGustSpeed'].median(),inplace=True)


# In[158]:


# For RainToday feature we cannot fill any value, so better to remove the NaN values 
data_yes.dropna(inplace=True)
data_no.dropna(inplace=True)


# In[159]:


data_filled= data_yes.append(data_no, ignore_index=True)


# In[160]:


data_filled.isnull().any()


# In[161]:


print("Percentage of removed points= {}%".format(100.00-(len(data_filled)*100/len(data))))


# In[162]:


# sorting the data based on data (Time based splitting)
data_filled=data_filled.sort_values(by='Date')
data_filled.head()


# In[163]:


#Removing unwanted features, RISK_MM is same as target label hence removing with data and loaction  
data_filled = data_filled.drop(['Date', 'Location','RISK_MM'], axis=1)
data_filled.head()


# In[164]:


#It is time to look at a heatmap (using the library seaborn) to see the correlation among features
fig = sns.heatmap(data_filled.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #df_train.corr()-->correlation matrix


# In[165]:


#We can see that 'MaxTemp' and 'Temp3pm' seem to be highly correlated.
#We want to make sure, so we will look at the numerical values
data_filled.corr()


# In[166]:


#We can remove either 'MaxTemp' or 'Temp3pm' because the correlation value is 0.969 (close to 1)
#The same goes for the pairs ( 'MinTemp', 'Temp9am' ), ('Pressure9am', 'Pressure3pm')
#DECISION: Remove 'Temp3pm', 'Temp9am' and 'Pressure9am'
data_filled = data_filled.drop( ['Temp3pm', 'Temp9am', 'Pressure9am'] , axis=1 )


# In[167]:


#see the correlation among features in df_train again
fig = sns.heatmap(data_filled.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #df_train.corr()-->correlation matrix


# In[168]:


#Outliers we are checking only for numerical features
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_filled[['MinTemp','MaxTemp']])


# In[169]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_filled[['WindGustSpeed','WindSpeed9am','WindSpeed3pm']])


# In[170]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_filled[['Humidity9am','Humidity3pm']])


# In[171]:


data_filled= data_filled[data_filled['Humidity3pm']!=0.0]
data_filled= data_filled[data_filled['Humidity9am']!=0.0]


# In[172]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_filled[['Humidity9am','Humidity3pm']])


# In[173]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_final[['Pressure9am']])


# In[174]:


sns.set(style="whitegrid")
plt.figure(figsize=(5, 6))
sns.boxplot(data=data_final[['Rainfall']])


# In[175]:


WindGustDir_encode = LabelEncoder()
data_filled['WindGustDir']=WindGustDir_encode.fit_transform(data_filled['WindGustDir'])

WindDir9am_encode = LabelEncoder()
data_filled['WindDir9am']=WindDir9am_encode.fit_transform(data_filled['WindDir9am'])

WindDir3pm_encode = LabelEncoder()
data_filled['WindDir3pm']=WindDir3pm_encode.fit_transform(data_filled['WindDir3pm'])

RainToday_encode = LabelEncoder()
data_filled['RainToday']=RainToday_encode.fit_transform(data_filled['RainToday'])

RainTomorrow_encode = LabelEncoder()
data_filled['RainTomorrow']=RainTomorrow_encode.fit_transform(data_filled["RainTomorrow"])

data_filled.head()


# In[176]:


df_train, df_test = train_test_split( data_filled, test_size = 0.2, random_state=42 )
df_train = df_train.copy()
df_test = df_test.copy()


# In[177]:


X_train = df_train.drop('RainTomorrow', axis = 1)
X_test = df_test.drop('RainTomorrow', axis = 1)
y_train = df_train['RainTomorrow']
y_test = df_test['RainTomorrow']
X_train.head()


# In[178]:


''' NORMALIZATION '''
scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform( X_train ) 
X_train.loc[:,:] = scaled_values
scaled_values = scaler.transform( X_test ) #DO NOT USE fit METHOD BECAUSE IT'S BEEN MODIFIED ACCORDING TO X_train
X_test.loc[:,:] = scaled_values


# In[185]:


t0=time.time()
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
import  pickle
filename = 'LogisticsRegressionModel.sav'
pickle.dump(logreg,open(filename,'wb'))
from sklearn.metrics import accuracy_score
acc_log = accuracy_score(y_test,y_pred) *100
print(acc_log)
t_log = time.time()-t0
print(t_log)


# In[186]:


t0=time.time()
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
filename = 'GaussianNaiveBayesModel.sav'
pickle.dump(gaussian,open(filename,'wb'))
acc_gaussian = accuracy_score(y_test,y_pred) *100
print(acc_gaussian)
t_gaussian = time.time()-t0
print(t_gaussian)


# In[187]:


t0=time.time()
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
filename = 'DecisionTreeModel.sav'
pickle.dump(decision_tree,open(filename,'wb'))
acc_decision_tree = accuracy_score(y_test,y_pred) *100
print(acc_decision_tree)
t_decision_tree = time.time()-t0
print(t_decision_tree)


# In[188]:


t0=time.time()
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
filename = 'RandomForestModel.sav'
pickle.dump(random_forest,open(filename,'wb'))
acc_random_forest = accuracy_score(y_test,y_pred) *100
print(acc_random_forest)
t_random_forest = time.time()-t0
print(t_random_forest)


# In[189]:


models = pd.DataFrame(
    {
    'ML Algorithm': ['Naive Bayes', 'Logistic Regression', 'Random Forest', 'Decision Tree'],
    'Score': [acc_gaussian, acc_log, acc_random_forest, acc_decision_tree],
    'Time': [t_gaussian, t_log, t_random_forest, t_decision_tree]
    }
)
models.sort_values(by='Score', ascending=False)


# In[ ]:




