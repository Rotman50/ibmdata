#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system(' mamba install pandas==1.3.3-y')
get_ipython().system(' mamba install numpy=1.21.2-y')
get_ipython().system(' mamba install sklearn=0.20.1-y')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())


# In[5]:


path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'


# In[6]:


await download(path, "auto.csv")
path="auto.csv"


# In[7]:


df = pd.read_csv(path)
df.head()


# In[8]:


from sklearn.linear_model import LinearRegression


# In[10]:


lm = LinearRegression()
lm


# In[12]:


x = df[["highway-mpg"]]
y = df["price"]


# In[13]:


lm.fit(x, y)


# In[14]:


yhat = lm.predict(x)
y[0:5]


# In[15]:


lm.intercept_


# In[16]:


lm.coef_


# In[17]:


Price = 38423.31 - 821.73 x highway-mpg


# In[18]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)


# In[ ]:




