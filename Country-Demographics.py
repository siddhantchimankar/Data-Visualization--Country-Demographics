#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
init_notebook_mode(connected = True)
cf.go_offline()

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import rcParams
rcParams['figure.figsize'] = 8,8
import warnings
warnings.filterwarnings('ignore')


# In[15]:


df = pd.read_csv('C:/Users/omprakash/Downloads/country/country.csv', decimal=",")


# In[16]:


df.columns = (["Country","Region","Population","Area","PopulationDensity","Coastline","Migration","InfantMortalityPer1000" ,"GDP","LiteracyRate","PhonesPer1000","FertileLand","Crops","Other","Climate","BirthRate","DeathRate","Agriculture","Industry","Service"])


# In[17]:


df.Country = df.Country.astype('category')
df.Region = df.Region.astype('category')
df.dtypes


# In[18]:


missing = df.isnull().sum()
missing
df.fillna(df.mean(),inplace=True)
df = df.drop('Other', 1)
df.region = df.Region.str.strip()
df.head(10)


# In[19]:


f, axes = plt.subplots(2,1,figsize=(15,10))


Region = df.Region.value_counts()
k1 = sns.barplot(data = df, y = Region.index, x = Region.values, orient = "h", palette='inferno_r')
plt.xlabel('Number of countries')
plt.ylabel('Region')
plt.title('Classification of Number of Countries by Region')
sns.set_style('whitegrid')
plt.show()

sns.set_style('whitegrid')
byRegion = df.groupby('Region')
byRegion.sum().Area
k1 = sns.barplot(data = df, y = Region.index, x = byRegion.sum().Area, orient = "h", palette='inferno_r')
plt.xlabel('Area')
plt.ylabel('Region')
plt.title('Classification of Regions by Total Area')
plt.show()





# In[42]:


sns.set_style('whitegrid')
sns.jointplot(data = df, x = 'Industry', y= 'InfantMortalityPer1000',               kind = 'hex', height = 9, space = 0, color="#4CB391")
plt.show()


# In[21]:


sns.distplot(df.DeathRate, bins = 10,hist = False, label = 'Death Rate' )
sns.distplot(df.BirthRate, bins = 10,hist = False, label = 'Birth Rate')
plt.show()


# In[22]:


sns.lmplot(data = df, x = 'LiteracyRate', y = 'BirthRate', fit_reg = False, hue = 'Region', height = 8, palette = 'inferno_r')
plt.show()


# In[23]:


sns.set_style('white')
sns.kdeplot(df.Agriculture, df.FertileLand, shade = True, cmap = 'Reds', shade_lowest = False)
sns.kdeplot(df.Agriculture, df.FertileLand, cmap = 'Reds')


plt.show()


# In[24]:


df['Progress'] = round(df['Industry']/df['Agriculture'])
df.replace([np.inf, -np.inf], np.nan,inplace=True)
sns.set_style('whitegrid')
sns.barplot(data = df, y = 'Region', x = 'Progress')
plt.xlim(0,20)
plt.show()


# In[25]:


sns.set_style('whitegrid')
byRegion = df.groupby('Region')
byRegion.sum().Area
k1 = sns.barplot(data = df, y = Region.index, x = byRegion.sum().Area, orient = "h", palette='inferno_r')
plt.xlabel('Area')
plt.ylabel('Region')
plt.title('Classification of Regions by Total Area')

plt.show()


# In[29]:


sns.clustermap(df.corr(), cmap = 'summer', linewidth = 2, linecolor = 'black')


# In[ ]:





# In[ ]:





# In[ ]:




