#!/usr/bin/env python
# coding: utf-8

# In[129]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[130]:


data = pd.read_csv('College_Data')


# In[131]:


data.head()


# In[132]:


data['Unnamed: 0'].nunique()


# In[133]:


data['Private'].unique()


# In[134]:


data.info()


# In[135]:


sns.heatmap(data.isnull(), yticklabels=False, cbar= False, cmap='viridis')


# In[136]:


sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=data, hue='Private',
           palette='coolwarm',height=6,aspect=1,fit_reg=False)


# In[137]:


sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=data, hue='Private',
           palette='coolwarm',height=6,aspect=1,fit_reg=False)


# In[138]:


sns.set_style('whitegrid')
sns.lmplot('Accept','Enroll',data=data,hue='Private', fit_reg=False, palette='coolwarm' )


# In[139]:


data[data['Grad.Rate'] > 100]


# In[140]:


data['Grad.Rate']['Cazenovia College'] = 100


# In[141]:


data[data['Grad.Rate'] > 100]


# In[142]:


data.head()


# In[143]:


private = pd.get_dummies(data['Private'], drop_first=True)


# In[144]:


private


# In[145]:


data=pd.concat([data,private], axis=1)


# In[146]:


data.head()


# In[147]:


data.drop(['Unnamed: 0','Private'],axis=1, inplace=True)


# In[148]:


data.head()


# In[149]:


data.info()


# In[150]:


data.head()


# In[151]:


from sklearn.cluster import KMeans


# In[152]:


kmeans = KMeans(n_clusters=2)


# In[153]:


kmeans.fit(data)


# In[154]:


kmeans.labels_


# In[155]:


data.head()


# In[156]:


from sklearn.metrics import confusion_matrix,classification_report


# In[157]:


print(confusion_matrix(data['Yes'],kmeans.labels_))
print('\n')
print(classification_report(data['Yes'],kmeans.labels_))


# In[ ]:




