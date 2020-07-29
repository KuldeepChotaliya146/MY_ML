#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("loan_train.csv")
data.head(5)


# # missing values

# In[3]:


data.isnull().sum()


# In[4]:


sns.heatmap(data.isnull())


# In[5]:


sns.countplot(x='Gender',hue='Loan_Status',data=data)


# In[6]:


data['Gender'] = data['Gender'].fillna('Male')


# In[7]:


data.isnull().sum()


# In[8]:


sns.countplot(x='Loan_Status',hue='Married',data=data)


# In[9]:


data['Married'] = data['Married'].fillna('Married')


# In[10]:


sns.countplot(x='Loan_Status',hue='Dependents',data=data)


# In[11]:


data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])


# In[12]:


data.isnull().sum()


# In[13]:


sns.countplot(x='Loan_Status',hue='Self_Employed',data=data)


# In[14]:


data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])


# In[15]:


data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())
data.isnull().sum()


# In[16]:


data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean())


# In[17]:


sns.countplot(x='Loan_Status',hue='Credit_History',data=data)


# In[18]:


data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
data.isnull().sum()


# # feature engineering

# In[19]:


data.info()


# In[20]:


data['Dependents'] = data['Dependents'].replace('3+', '3')


# In[21]:


data


# In[22]:


from sklearn.preprocessing import LabelEncoder
categorical_column = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in categorical_column:
    data[i] = le.fit_transform(data[i])
data.head()


# In[23]:


data.drop('Loan_ID',axis=1,inplace=True)
data


# In[24]:


plt.figure(figsize=(12,9))
sns.heatmap(data.corr(),annot=True)


# # model training

# In[25]:


X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score 
model = LogisticRegression()
#model = RandomForestClassifier()
#model = DecisionTreeClassifier()
#model = SVC()
model.fit(X,y)


# In[27]:


acc = cross_val_score(model, X, y, cv=10, scoring='accuracy').mean()
acc


# In[ ]:




