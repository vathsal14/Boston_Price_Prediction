#!/usr/bin/env python
# coding: utf-8

# # Importing Required Libraries

# In[1]:


import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# # Importing Data

# In[2]:


from sklearn.datasets import fetch_openml

housing = fetch_openml(name="boston", version=1, as_frame=True)
data = housing.data
target = housing.target


# In[3]:


boston = {'data': data.values, 'target': target.values, 'feature_names': housing.feature_names}


# In[4]:


boston.keys()


# In[5]:





# # Dataset Preparation

# In[6]:


df = pd.DataFrame(boston['data'],columns=boston['feature_names'])


# In[7]:


df


# In[8]:


df['PRICE'] = boston['target']

for col in df.columns:
    if col != 'PRICE':
        df[col] = pd.to_numeric(df[col])



# In[9]:


df


# In[10]:





# In[11]:


df.describe()


# In[12]:


df.isnull().sum()


# In[13]:


df.dtypes


# # Exploratory Data Analysis

# In[14]:


plt.figure(figsize = (15,10))
sns.heatmap(df.corr(),annot = True)


# In[15]:


sns.pairplot(df)


# In[16]:


sns.regplot(x="CRIM",y="PRICE",data=df)
plt.show()


# In[17]:


sns.regplot(x="RM",y="PRICE",data=df)
plt.show()


# In[18]:


sns.regplot(x="INDUS",y="PRICE",data=df)
plt.show()


# In[19]:


sns.regplot(x="CHAS",y="PRICE",data=df)
plt.show()


# In[20]:


sns.regplot(x="ZN",y="PRICE",data=df)
plt.show()


# In[21]:


sns.regplot(x="NOX",y="PRICE",data=df)
plt.show()


# In[22]:


sns.regplot(x="AGE",y="PRICE",data=df)
plt.show()


# In[23]:


sns.regplot(x="DIS",y="PRICE",data=df)
plt.show()


# In[24]:


sns.regplot(x="RAD",y="PRICE",data=df)
plt.show()


# In[25]:


sns.regplot(x="TAX",y="PRICE",data=df)
plt.show()


# In[26]:


sns.regplot(x="PTRATIO",y="PRICE",data=df)
plt.show()


# In[27]:


sns.regplot(x="B",y="PRICE",data=df)
plt.show()


# In[28]:


sns.regplot(x="LSTAT",y="PRICE",data=df)
plt.show()


# # Data Preparation

# In[29]:


x = df.drop("PRICE",axis=1)

y = df['PRICE']


# In[30]:


x


# In[31]:


y


# In[32]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)


# In[33]:


print(x.shape)
print(x_train.shape)
print(x_test.shape)


# In[34]:


ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test  = ss.transform(x_test)


# In[35]:


x_train


# In[36]:


x_test


# # Model Training

# # Linear Regression

# In[39]:


from catboost import CatBoostRegressor

lr = CatBoostRegressor()

param_dist = {
    'depth': [4, 6, 8],
    'learning_rate': [0.1, 0.01, 0.001],
    'l2_leaf_reg': [1, 3, 5, 7],
    'bagging_temperature': [0.5, 1, 1.5],
    'random_strength': [0.5, 1, 1.5],
    'border_count': [32, 64, 128],
    'iterations': [100, 200, 300],
}

random_search = RandomizedSearchCV(lr, param_distributions=param_dist, n_iter=20, cv=5)


# In[40]:


random_search.fit(x_train,y_train)


# In[41]:


y_pred = random_search.predict(x_test)


# In[42]:


plt.scatter(y_test, y_pred, color ="green")


# In[43]:


r2_score(y_test,y_pred)


# # Model Validation

# In[44]:


tr = ss.transform(boston['data'][0].reshape(1,-1))


# In[45]:


random_search.predict(tr)


# # Pickling the Files

# In[46]:


import pickle as pkl

pkl.dump(random_search,open('housepred.pkl','wb'))

pkl.dump(ss,open('scaler.pkl','wb'))


# In[ ]:




