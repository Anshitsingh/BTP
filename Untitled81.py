
# coding: utf-8

# In[8]:

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
get_ipython().magic(u'matplotlib inline')
pylab.rcParams['figure.figsize'] = 16, 12


# In[47]:

image = Image.open('F:/LATEX/post midsem/oldbuilding.jpg')

# Image is (687 x 1025, RGB channels)
image = np.array(image)
original_shape = image.shape

# Flatten image.
X = np.reshape(image, [-1, 3])

plt.imshow(image)


# In[48]:

bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)
print(bandwidth)


# In[49]:

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)


# In[50]:

labels = ms.labels_
print(labels.shape)
cluster_centers = ms.cluster_centers_
print(cluster_centers.shape)

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)


# In[51]:

segmented_image = np.reshape(labels, original_shape[:2])  # Just take size, ignore RGB channels


# In[52]:

a=plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.axis('off')


# In[53]:

a.savefig("F:/LATEX/post midsem/o.pdf")


# In[ ]:



