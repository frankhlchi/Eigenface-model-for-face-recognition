#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import imageio
from matplotlib import pylab as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from numpy import linalg as LA
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_normal


# ### Eigenface for face recognition
# 
# (b) Load the training set into a matrix 
# 
# there are 540 training images in total, each has 50 × 50
# pixels that need to be concatenated into a 2500-dimensional vector. So the size of X should
# be 540×2500, where each row is a flattened face image. Pick a face image from X and display
# that image in grayscale. Do the same thing for the test set. The size of matrix Xtest for the test
# set should be 100×2500.

# In[2]:


#import the training set
train_labels, train_data = [], []
for line in open('./faces/train.txt'):
    im = imageio.imread(line.strip().split()[0])
    train_data.append(im.reshape(2500,))
    train_labels.append(line.strip().split()[1])

#import the testing set
test_labels, test_data = [], []
for line in open('./faces/test.txt'):
    im = imageio.imread(line.strip().split()[0])
    test_data.append(im.reshape(2500,))
    test_labels.append(line.strip().split()[1])
    
train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)
test_data, test_labels = np.array(test_data, dtype=float), np.array(test_labels, dtype=int)


# In[3]:


fig = plt.figure(figsize=(10,5))
ax1, ax2 = fig.subplots(1, 2, sharey=True)
ax1.imshow(train_data[10, :].reshape(50,50), cmap = cm.Greys_r)
ax1.set_title('random face in training set')
ax2.imshow(test_data[10, :].reshape(50,50), cmap = cm.Greys_r)
ax2.set_title('random face in testing set')


# In[4]:


print ('training set shape', train_data.shape, train_labels.shape)
print ('testing set shape', test_data.shape, test_labels.shape)


# (c) Average Face.
# 
# Compute the average face $ µ $ from the whole training set by summing up every
# column in $ X $ then dividing by the number of faces. Display the average face as a grayscale
# image.

# In[5]:


mean_face_train = train_data.mean(axis = 0)
mean_face_test = test_data.mean(axis = 0)

fig = plt.figure(figsize=(10,5))
ax1, ax2 = fig.subplots(1, 2, sharey=True)
ax1.imshow(mean_face_train.reshape(50,50), cmap = cm.Greys_r)
ax1.set_title('average face in training set')
ax2.imshow(mean_face_test.reshape(50,50), cmap = cm.Greys_r)
ax2.set_title('average face in testing set')


# (d) Mean Subtraction. 
# 
# Subtract average face $ µ $ from every column in $ X $ . That is, $xi := xi − µ $, where $ xi $ is the i-th column of $ X $. Pick a face image after mean subtraction from the new $X$ and display that image in grayscale. Do the same thing for the test set $ X $ test using the precomputed average face $µ$ in (c).

# In[6]:


mean_face_subtraction  =  train_data - mean_face_train
mean_face_subtraction_test  =  test_data - mean_face_test 

fig = plt.figure(figsize=(10,5))
ax1, ax2 = fig.subplots(1, 2, sharey=True)
ax1.imshow(mean_face_subtraction[1,:].reshape(50,50), cmap = cm.Greys_r)
ax1.set_title('some mean subtracted face in training set')
ax2.imshow(mean_face_subtraction_test[1,:].reshape(50,50), cmap = cm.Greys_r)
ax2.set_title('some mean subtracted face in testing set')


# (e) Eigenface. 
# 
# Perform Singular Value Decomposition (SVD) on training set. Display the first 10 eigenfaces as 10 images in grayscale.

# In[7]:


u, s, vh = np.linalg.svd(train_data, full_matrices=False)


# In[8]:


fig, ax = plt.subplots(2, 5, figsize=(15, 7), sharex='col', sharey='row')
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(vh[i*5 + j, :].reshape(50,50), cmap = cm.Greys_r)       
        if (i*5 + j + 1) == 1:
            ax[i, j].title.set_text(str('the 1st eigenface'))
        elif (i*5 + j + 1) == 2:
            ax[i, j].title.set_text(str('the 2nd eigenface'))
        else:
            ax[i, j].title.set_text(str('the %ith eigenface'%(i*5 + j + 1)))
       


# (f) Low-rank Approximation

# In[9]:


error_dic = {'rank':[], 'error':[]}
for rank in range(1,201):
    X_bar = np.dot(u, np.dot(np.diag(np.hstack([s[:rank], np.zeros(len(s) - rank)])), vh))
    #caluculate Frobenius Norm
    error = LA.norm(train_data -                   np.dot(u[:,:rank],                          np.dot(np.diag(s[:rank]), vh[:rank,:])))
    error_dic['rank'].append(rank)
    error_dic['error'].append(error)

error_df = pd.DataFrame(error_dic)

error_df


# In[10]:


plt.figure(figsize=(8,5))
error_df = pd.DataFrame(error_dic)
plt.plot(error_df['rank'],error_df['error'])
plt.xlabel('rank r')
plt.ylabel('rank-r approximation error')


# (g) Eigenface Feature

# In[11]:


# this is the F-function required in the Question 
def eigen_feature(X, v_matrix, r=10):
    F =  np.matmul(X,v_matrix[:r,:].T)
    return F


# In[12]:


u, s, vh = np.linalg.svd(train_data, full_matrices=False)
training_set_eigen_feature = eigen_feature(train_data, vh, r=10)

u_test, s_test, vh_test = np.linalg.svd(test_data, full_matrices=False)
test_set_eigen_feature = eigen_feature(test_data, vh, r=10)


# (h) Face Recognition

# In[13]:


# accuracy when r = 10
clf = LogisticRegression(solver='lbfgs').                    fit(training_set_eigen_feature, train_labels)
prediction = clf.predict(test_set_eigen_feature)

print ('accuracy', np.sum(test_labels == prediction)/len(test_labels))


# In[14]:


logit_error = {'rank':[], 'accuracy':[]}
u, s, vh = np.linalg.svd(train_data, full_matrices=False)
u_test, s_test, vh_test = np.linalg.svd(test_data, full_matrices=False)

for rank in range(1,201):
    training_set_eigen_feature = eigen_feature(train_data, vh, r=rank)
    test_set_eigen_feature = eigen_feature(test_data, vh, r=rank)
    
    clf = LogisticRegression(solver='lbfgs',multi_class='ovr').                    fit(training_set_eigen_feature, train_labels)
    prediction = clf.predict(test_set_eigen_feature)
    
    logit_error['rank'].append(rank)
    logit_error['accuracy'].append(np.sum(test_labels == prediction)/len(test_labels))


# In[15]:


plt.figure(figsize=(8,5))
logit_error_df = pd.DataFrame(logit_error)
plt.plot(logit_error_df['rank'],logit_error_df['accuracy'])
plt.xlabel('rank r')
plt.ylabel('accuracy of logistic regression with r eigen features')

