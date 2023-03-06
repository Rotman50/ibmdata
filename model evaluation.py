#!/usr/bin/env python
# coding: utf-8

# In[1]:


import piplite
import micropip
await piplite.install(['pandas'])
await piplite.install(['matplotlib'])
await piplite.install(['scipy'])
await piplite.install(['seaborn'])
await micropip.install(['ipywidgets'],keep_going=True)
await micropip.install(['tqdm'],keep_going=True)


# In[13]:


import pandas as pd
import numpy as np


# In[19]:


from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())


# In[20]:


import pandas as pd
import numpy as np


# In[21]:


path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'


# In[22]:


await download(path, "auto.csv")
path="auto.csv"


# In[23]:


df = pd.read_csv(path)


# In[24]:


df.to_csv('module_5_auto.csv')


# In[25]:


df=df._get_numeric_data()
df.head()


# In[26]:


from ipywidgets import interact, interactive, fixed, interact_manual


# In[27]:


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


# In[28]:


def pollyplotty(x_train, x_test, y_train, y_test, lr, poly_transform):
    width = 12
    height =10
    plt.figure(figsize=(width, height))
    
    
    
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


# <h3> Training Data</h3>

# In[29]:


y_data = df['price']


# In[30]:


x_data = df.drop('price', axis=1)


# <p> then we train_test_split the data</p>

# In[33]:


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.10, random_state = 1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# <p>Let's import LinearRegression from the module linear_model.</p>

# In[34]:


from sklearn.linear_model import LinearRegression


# In[35]:


lr = LinearRegression()


# In[42]:


lr.fit(x_train[['horsepower']], y_train)


# <p> calucate R^2 </p>

# In[46]:


lr.score(x_test[['horsepower']], y_test)


# In[47]:


lr.score(x_train[['horsepower']], y_train)


# <h3 id="ref2">Part 2: Overfitting, Underfitting and Model Selection</h3>

# In[55]:


lre = LinearRegression()
lre.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)


# In[50]:


y_hat = lre.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
y_hat[0:5]


# In[57]:


y_hat_test = lre.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
y_hat_test[0:5]


# In[51]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[53]:


Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, y_hat, "Actual Values (Train)", "Predicted Values (Train)", Title)


# In[58]:


Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_test, y_hat_test, "Actual Values (Train)", "Predicted Values (Train)", Title)


# In[ ]:




