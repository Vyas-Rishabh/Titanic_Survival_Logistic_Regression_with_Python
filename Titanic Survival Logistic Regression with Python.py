#!/usr/bin/env python
# coding: utf-8

# # Titanic Disaster Survival Using Logistic Regression

# In[1]:


#import Libraries


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ### Load the Data 

# In[8]:


titanic_data = pd.read_csv("titanic_train.csv")


# In[10]:


titanic_data


# In[11]:


len(titanic_data)


# ### View the data using head function which return top rows

# In[12]:


titanic_data.head()


# In[13]:


titanic_data.index


# In[14]:


titanic_data.columns


# In[16]:


titanic_data.info()


# In[18]:


titanic_data.dtypes


# In[21]:


titanic_data.describe()


# ### Explaining Datasets

# survival : Survival 0 = No, 1 = Yes <br>
# pclass : Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd <br>
# sex : Sex <br>
# Age : Age in years <br>
# sibsp : Number of siblings / spouses aboard the Titanic 
# <br>parch # of parents / children aboard the Titanic <br>
# ticket : Ticket number fare Passenger fare cabin Cabin number <br>
# embarked : Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton <br>

# # Data Analysis

# ### import Seaborn for visually analysing the data

# ### Find out how many survived vs Died using countplot method of seaborn

# In[22]:


#countplot of survived vs not survived


# In[23]:


sns.countplot(x='Survived', data=titanic_data)


# ### Male vs Female Survival

# In[24]:


#Male vs Female Survival


# In[25]:


sns.countplot(x='Survived', data=titanic_data, hue='Sex')


# **See age group of passengeres travelled **<br>
# Note: We will use displot method to see the histogram. However some records does not have age hence the method will throw an error. In order to avoid that we will use dropna method to eliminate null values from graph

# In[26]:


#Check for null


# In[28]:


titanic_data.isna()


# In[29]:


#Check how many values are null


# In[30]:


titanic_data.isna().sum()


# In[31]:


#Visualize null values


# In[32]:


sns.heatmap(titanic_data.isna())


# In[33]:


#find the % of null values in age column


# In[39]:


(titanic_data['Age'].isna().sum()/len(titanic_data['Age']))*100


# In[37]:


#find the % of null values in cabin column


# In[40]:


(titanic_data['Cabin'].isna().sum()/len(titanic_data['Cabin']))*100


# In[41]:


#find the distribution for the age column


# In[42]:


sns.displot(x='Age', data=titanic_data)


# # Data Cleaning

# **Fill the missing values**</BR>
# we will find the missing values for age. In order to fill missing values we use fillna method.</BR>
# For now we will fill the missing age by taking average of all age.

# In[43]:


#fill age column


# In[52]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True) 


# In[ ]:


#verify null values


# In[51]:


titanic_data['Age'].isna().sum()


# **Alternatively we will visualise the null value using heatmap.**</BR>
# we will use heatmap method by passing only records which are null.

# In[53]:


#visualize null values


# In[54]:


sns.heatmap(titanic_data.isna())


# **we can see the cabin column has a number of null values, as such we can not use it for prediction. Hence we will drop it.**

# In[55]:


#drop cabin column


# In[56]:


titanic_data.drop('Cabin', axis=1, inplace=True)


# In[57]:


#see the content of data


# In[58]:


titanic_data.head()


# **Preaparing Data for Model**</BR>
# No we will require to convert all non-numerical columns to numeric. Please note this is required for feeding data into model. Lets see which columns are non numeric info describe method

# In[59]:


#Check for the non-numeric column


# In[60]:


titanic_data.info()


# In[61]:


titanic_data.dtypes


# **We can see, Name, Sex, Ticket and Embarked are non-numerical.It seems Name,Embarked and Ticket number are not useful for Machine Learning Prediction hence we will eventually drop it. For Now we would convert Sex Column to dummies numerical values****

# In[62]:


#convert sex column to numerical values


# In[65]:


gender=pd.get_dummies(titanic_data['Sex'],drop_first=True)


# In[66]:


titanic_data['Gender']=gender


# In[69]:


titanic_data.head()
#Here, in Gender male=1 & female=0


# In[70]:


#drop the column which are not require


# In[71]:


titanic_data.drop(['Name','Sex','Ticket','Embarked'], axis=1, inplace=True)


# In[72]:


titanic_data.head()


# In[73]:


#Seperate Dependent and Independent variables


# In[75]:


x=titanic_data[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=titanic_data['Survived']


# In[76]:


x


# In[77]:


y


# # Data Modeling

# **Building Model using Logistic Regression**</BR>
# </BR>
# **Build the model**

# In[79]:


#import train test split method


# In[80]:


from sklearn.model_selection import train_test_split


# In[81]:


#train test split


# In[83]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[84]:


#import logistic Regression


# In[85]:


from sklearn.linear_model import LogisticRegression


# In[86]:


#fit Logistic Regression


# In[92]:


lr = LogisticRegression()


# In[93]:


lr.fit(X_train,y_train)


# In[96]:


#predict


# In[99]:


predict=lr.predict(X_test)


# # Testing

# **See how our model is performing**

# In[100]:


#print confusion matrix


# In[101]:


from sklearn.metrics import confusion_matrix


# In[104]:


pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'], 
             index=['Actual No','Actual Yes'])


# In[105]:


#import classification report


# In[106]:


from sklearn.metrics import classification_report


# In[108]:


print(classification_report(y_test, predict))


# **Precision is fine considering Model Selected and Available Data. Accuracy can be increased by further using more features (which we dropped earlier) and/or by using other model**
# 
# Note:</BR>
# Precision : Precision is the ratio of correctly predicted positive observations to the total predicted positive observations
# Recall : Recall is the ratio of correctly predicted positive observations to the all observations in actual class F1 score - F1 Score is the weighted average of Precision and Recall.

# You can find this project on <a href="https://github.com/Vyas-Rishabh/Titanic_Survival_Logistic_Regression_with_Python"><B>GitHub</B></a>
