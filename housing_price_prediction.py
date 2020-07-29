#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# In[2]:


housing = pd.read_csv("real_data.csv")


# In[3]:


housing


# In[4]:


housing.info()


# In[5]:


housing.describe()


# In[6]:


#housing.hist(figsize=(15,15))
#plt.show()


# In[7]:


from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(housing,test_size=0.2,random_state=0)


# In[8]:


train_data


# In[9]:


test_data


# In[10]:


#to split data in equal manner
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=0)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[11]:


strat_train_set


# In[12]:


housing = strat_train_set.copy()


# In[13]:


#for corelate data with each other


# In[14]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[15]:


from pandas.plotting import scatter_matrix
attributes=["MEDV","RM","CHAS","LSTAT"]
scatter_matrix(housing[attributes],figsize=(15,9))
plt.show()


# In[16]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)


# In[17]:


housing = strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# # to fill null values using imputer class

# In[18]:


median = housing["RM"].median()


# In[19]:


median


# In[20]:


housing["RM"].fillna(median)


# In[21]:


housing.shape


# In[22]:


housing.info()


# In[23]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[24]:


imputer.statistics_


# In[25]:


new_data = imputer.transform(housing)


# In[26]:


new_housing = pd.DataFrame(new_data,columns=housing.columns)

new_housing


# In[27]:


new_housing.describe()


# # creating a pipeline for filling missing data automatically

# In[28]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
my_pipeline = Pipeline([('imputer',SimpleImputer(strategy="median")),('std_scaler',StandardScaler())])


# In[29]:


converted_housing = my_pipeline.fit_transform(housing)


# In[30]:


converted_housing


# In[31]:


converted_housing.shape


# In[32]:


#mdoel/////


# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model  = RandomForestRegressor()
model.fit(converted_housing,housing_labels)


# In[34]:


some_data = housing.iloc[:5]


# In[35]:


some_labels= housing_labels.iloc[:5]


# In[36]:


prepared_data = my_pipeline.transform(some_data)


# In[37]:


model.predict(prepared_data)


# In[38]:


some_labels


# # evulating model

# In[39]:


#for DecisionTree 
housing_predictions = model.predict(converted_housing)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
rmse


# # for better evalution using cross validation

# In[40]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, converted_housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores


# In[41]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
print_scores(rmse_scores)


# In[42]:


#for decisiontree rmse 4.28
#for linearregression rmse 4.7
#for randomForest rmse 3.38


# # for saving model

# In[43]:


from joblib import dump,load
dump(model,'CK.joblib')


# In[44]:


model.score(converted_housing,housing_labels)*100


# # for testting

# In[45]:


x_test = strat_test_set.drop("MEDV",axis=1)
y_test = strat_test_set["MEDV"]
prepared_x = my_pipeline.transform(x_test)
final_pre = model.predict(prepared_x)
final_mse = mean_squared_error(y_test,final_pre)
final_rmse = np.sqrt(final_mse)


# In[46]:


final_rmse


# In[47]:


final_pre


# In[48]:


y_test


# In[ ]:




