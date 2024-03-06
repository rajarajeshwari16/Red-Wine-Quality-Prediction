#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMOPORTING ESSENTIAL LIBRARIES


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# IMPORTING AND UNDERSTANDING DATASET


# In[4]:


df = pd.read_csv("winequality-red.csv")
df


# In[5]:


# PRINTING OUT FIRST AND LAST 5 COLUMNS


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


# COLUMNS OF DATASET


# In[9]:


df.columns


# In[10]:


#SHAPE OF DATASET


# In[11]:


df.shape


# In[12]:


#DATATYPE OF EACH COLUMNS


# In[14]:


df.dtypes


# In[15]:


# checking whether there is a missing value 


# In[16]:


df.isna().sum()


# In[17]:


# SEPERATING INPUT AND OUTPUT LABELS 


# In[18]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]
y


# In[19]:


#GRAPH PLOTTING


# In[20]:


sns.regplot(x=df['fixed acidity'],y=y,color='m')


# In[21]:


sns.regplot(x=df['volatile acidity'],y=y,color='b')


# In[22]:


sns.regplot(x=df['citric acid'],y=y,color='r')


# In[23]:


sns.regplot(x=df['residual sugar'],y=y,color='hotpink')


# In[24]:


sns.regplot(x=df['chlorides'],y=y,color='black')


# In[25]:


sns.regplot(x=df['free sulfur dioxide'],y=y,color='yellow')


# In[26]:


sns.regplot(x=df['total sulfur dioxide'],y=y,color='black')


# In[27]:


sns.regplot(x=df['density'],y=y,color='orange')


# In[28]:


sns.regplot(x=df['pH'],y=y,color='grey')


# In[29]:


sns.regplot(x=df['sulphates'],y=y,color='gold')


# In[30]:


sns.regplot(x=df['alcohol'],y=y,color='blue')


# In[31]:


sns.regplot(x=df['quality'],y=y,color='grey')


# In[ ]:


# MODEL CREATION


# In[41]:


pip install scikit-learn


# In[45]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)


# In[46]:


#MODEL CREATION


# In[48]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred


# In[49]:


y_test


# In[50]:


# PRINTING DATAFRAME


# In[53]:


df1= pd.DataFrame({'Actual_value':y_test,'predicted_value':y_pred,'Difference':y_test-y_pred})
df1


# In[54]:


# PERFORMANCE EVALUATION


# In[55]:


# mean absolute error


# In[56]:


from sklearn.metrics import mean_absolute_error
print ('MAE is',mean_absolute_error(y_test,y_pred))


# In[57]:


# mean absolute percentage error


# In[58]:


from sklearn.metrics import mean_absolute_percentage_error
print ('MAE is',mean_absolute_percentage_error(y_test,y_pred))


# In[ ]:


# mean squared error


# In[60]:


from sklearn.metrics import mean_squared_error
value = mean_squared_error(y_test,y_pred)
print(value)


# # R2 score

# In[61]:


from sklearn.metrics import r2_score
print('R2 score is',r2_score(y_test,y_pred))


# In[ ]:




