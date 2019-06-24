#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv(r'F:\Tutorials\Python\Udemy - Machine Learning Practical 6 Real-World Applications\twitter sentiment\train_E6oV3lV.csv')


# In[3]:


dataset.head(5)


# In[4]:


import re
import nltk


# In[5]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[6]:


corpus=[]
for i in range(0,31962):
    review=dataset['tweet'][i]
    review=re.sub('@[\w]*',' ',review)
    review=re.sub('[^a-zA-z#]',' ',review)
    review=review.lower()
    review=review.split()
    review=[word for word in review if len(word)>3]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(corpus,dataset['label'], test_size = 0.2, random_state = 0)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
x_train_vector = cv.fit_transform(x_train).toarray()


# In[9]:


print(x_train_vector)


# In[10]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500,min_samples_leaf=250)
classifier.fit(x_train_vector,y_train)


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
x_test_vector = cv.fit_transform(x_test).toarray()


# In[12]:


y_pred=classifier.predict(x_test_vector)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# In[14]:


test=pd.read_csv(r'F:\Tutorials\Python\Udemy - Machine Learning Practical 6 Real-World Applications\twitter sentiment\test_tweets_anuFYb8.csv')


# In[15]:


corpus_test=[]
for i in range(0,17197):
    review=test['tweet'][i]
    review=re.sub('@[\w]*',' ',review)
    review=re.sub('[^a-zA-z#]',' ',review)
    review=review.lower()
    review=review.split()
    review=[word for word in review if len(word)>3]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_test.append(review)


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
corpus_test_vector = cv.fit_transform(corpus_test).toarray()


# In[17]:


predictions=classifier.predict(corpus_test_vector)


# In[18]:


print(predictions)


# In[19]:


result=pd.DataFrame({'id':test['id'],'label':predictions})


# In[20]:


print(result)


# In[21]:


result.to_csv('test_predictions.csv',index=False)


# In[ ]:




