#!/usr/bin/env python
# coding: utf-8

# ## POLA RAVI TEJA
# 
# ### THE SPARKS FOUNDATION
# 
# ### Data Science and Business Analytics  - Task 6
# 
# Prediction using Decision Tree Classifier.
# 
# The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.
# 
# Dataset description: It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn import tree


# ## Loading and Preparing Data

# In[2]:


iris_df = pd.read_csv("Iris.csv")


# In[3]:


iris_df.info()


# In[4]:


iris_df


# In[5]:


iris_df = iris_df.drop("Id", axis=1)


# In[6]:


iris_df.head()


# In[7]:


sns.pairplot(iris_df)


# In[8]:


corr = iris_df.corr(method ='pearson')
sns.heatmap(corr, annot = True)


# ## Label Encoding

# In[9]:


LE = LabelEncoder()

iris_df_LE = iris_df 
iris_df_LE.Species = LE.fit_transform(iris_df_LE.Species)

for col in iris_df_LE.columns:
    print(col,':',len(iris_df_LE[col].unique()),'labels')


# ## Test and Train dataset Split

# In[10]:


data = iris_df.values
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 


# ## Decision Tree Classifier

# In[11]:


DTC = DecisionTreeClassifier()
DTC = DTC.fit(X_train,y_train)


# ## Predictions

# In[12]:


predictions = DTC.score(X,y)
print(predictions)


# In[13]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(DTC,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')


# In[ ]:




