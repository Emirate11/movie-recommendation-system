#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
get_ipython().system(' pip install fuzzywuzzy')
from fuzzywuzzy import process


# In[4]:


movie_df = pd.read_csv("movies.csv")
rating_df = pd.read_csv("ratings.csv")
rating_df


# In[5]:


movie_df


# In[6]:


movie = movie_df[["movieId", "title"]]
movie


# In[7]:


rating = rating_df[["userId", "movieId", "rating"]]
rating


# In[10]:


movie_user_pivot = rating.pivot(index = "movieId", columns = "userId", values = "rating").fillna(0)
movie_user_pivot


# In[11]:


movie_user_matrix = csr_matrix(movie_user_pivot.values)
movie_user_matrix


# In[14]:


Nearestneighbor_model = NearestNeighbors(metric = "cosine", algorithm = "brute", n_neighbors = 30)
Nearestneighbor_model


# In[15]:


Nearestneighbor_model.fit(movie_user_matrix)


# In[29]:


def Justin_recommender(movie_name, data, model, n_recommendations):
    model.fit(data)
    idx = process.extractOne(movie_name, movie["title"])[2]
    print("Selected Movie: ", movie["title"][idx])
    print("Loading Recommendations....")
    print("Loading Recommendations.........")
    print("Loading Recommendations..............")
    vector_closeness, number_indices = model.kneighbors(data[idx], n_neighbors = n_recommendations)
    for search in number_indices:
        print(movie["title"][search].where(search != idx))
        
Justin_recommender("Intersteller", movie_user_matrix, Nearestneighbor_model, 30)
    


# In[ ]:




