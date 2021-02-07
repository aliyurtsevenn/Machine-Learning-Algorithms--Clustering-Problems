'''
In Unsupervised machine learning, you would have attributes and try to find out the labels without any
prior knowledge! You don't have labels training data here!

1. One of the most  basic and well known unsupervised algorithm
2. Only input vectors are used, without any labels!
3. The idea behind this is to group the similar data points together and discover underlying patterns!
4. To achieve such tast, you need to define a fixed number of cluster known as k in the dataset.
5. K refers to the number of clusters and number of centroids! Centroid is the center of the clusters
6. This algorithm defines k number of centroids, then allocates every data point to the nearest cluster
while keeping the centroids as smallest values!
7. Means here is to find the centroids by averaging the data!

The idea in this algorithm is that,

1. User define the number of the centroids!
2. Random number generater gives random coordinates for these much of centroids.
3. Mid coordinate between each centroid is taken,
4. 90 degree line can be drawn for the vectors of each 2 centroid. Here, you can classify, which points
are close to the which centroid!
5. Then, you take the average of the coordinates of these data points and change each of your centroid
coordinate as the average value!
6. Your repeat the same process until the classes are not changed!

Disadvantage!

We have to measure the distance of each datapoint between each centroid! When the number of
centroid increases the measurement numbers increase 2 fold. The measurement number is

Number of points x Number of centroids x Number of iterations(re-centering the centroids)! x Number of features!

It takes a lot of time if you have so many data points or features, but still, it is faster than some of the
unsupervised clustering algorithms!
'''

import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib.pyplot as plt
from kneed import KneeLocator
import pandas as pd
import pathlib
import os
from os.path import expanduser
from sklearn import linear_model,preprocessing
'''
Note that if we have digits which have very large values, we need to scale them between -1 to 1! 
We can achive this, using scale from sklearn.preprocessing!
'''

# digit_data=load_digits()
# print(digit_data.feature_names) # They are all rgb values!

'''
This data can be obtained from sklearn datasets library!
In real life, you have to deal with a real data file!

Here, You can use a real data instead. Above in the first 2 examples, you can get the attributes
this way, but I will upload the real data into attachment, so that you can upload another one
as excel, tab separated csv, comma separated csv and semicolon separated csv or colon separated csv files.
Your data must be in an appropriate formats!
'''
path_parameter_file=os.path.join(pathlib.Path(__file__).parent.absolute(),'parameters.txt')


if os.path.exists(path_parameter_file)==False:
    print('parameters.txt file should be in the same directory with the interactome_parameter.py file! There will be an error!')

data=pd.read_csv(path_parameter_file,delimiter='\t',header=None)
input_path= data[data.columns[0]][0]
output_path= data[data.columns[0]][1]
sample_name_situation= data[data.columns[0]][2]
attribute_madeup= data[data.columns[0]][3]
Categorical_feature= data[data.columns[0]][4].split(",")
max_iter= data[data.columns[0]][5]
init= data[data.columns[0]][6]
separation_csv=data[data.columns[0]][7]

# Let me look at which type you want to analyze and read your data accordingly!

filename, file_extension = os.path.splitext(input_path)

if file_extension==".xlsx":
    my_data= pd.read_excel(input_path)
elif file_extension==".txt" or ".csv":
    my_data=pd.read_csv(input_path,delimiter=separation_csv)

# Let me remove the missing values in your data!
s=len(my_data.columns)
my_data.dropna(inplace=True)
if s!=len(my_data.columns):
    my_data=my_data[my_data.columns[1:]]
# Filtering out the Name of the Samples to get only attributes!

if sample_name_situation=="Yes":
    sample_name="Names"
    my_data_frame=my_data.drop([sample_name],1)

if attribute_madeup=="No":
    converter = preprocessing.LabelEncoder()
    for j in Categorical_feature:
        my_data_frame[j] = pd.Series(converter.fit_transform(my_data_frame[j].tolist()))

#
# #Lets convert dataframe to numpy array
# Attributes= np.array(my_data_frame)
# #
# #Let's make your data between -1 and 1.
# Attributes=scale(Attributes)
#
# # # Shuffle the data!
# Attributes=shuffle(Attributes)
#
#
# # Shaping the data! according to the instance numbers!
# samples,features=Attributes.shape # This gives you have many features and samples you have
#
# # In order to find out how many clusters you have we can use elbow's rule!
# sse=[]
# for k in range(1,25):
#     clf=KMeans(n_clusters=k,init="k-means++",n_init=20,max_iter=500)
#     clf.fit(Attributes)
#     sse.append(clf.inertia_)
#
# kl = KneeLocator( range(1, 25), sse, curve="convex", direction="decreasing")
# k= kl.elbow
#
#
# '''
# At the end, you need to measure accuracy of your model! For this, you can use metric function of
# the sklearn library! There are many scores for the measuring the accuracy of unsupervised learning!
# These scores are found with a sophisticated mathematical calculations! We will write a function for
# accuracy scoring and call this function in the K-Means classifier model!
#
# Now, lets write the metric scoring function!
# '''
#
# '''
# Kmeans() parameters!
#
# n_clusters: Number of clusters
# init: Centroids are generated in random positions or in more equal distanced way! "random" or k-means++ (Play with this to see if accuracy change or not!) Result doesn't change a lot using K-Means++ you don't do the iteration many time!
# n_init: Amount of time for generating the first best centroid classifiers!
# max_iter: it is by default 300 iteration and around this, it takes the best classifier! But, increasing this can increase the accuracy of our classifier! If you have time, do this!
# '''
# K_means_clf=KMeans(n_clusters=k,init="k-means++",n_init=int(init),max_iter=int(max_iter))
#
# model= K_means_clf.fit(Attributes)
# #
# '''
#
# Note that, in these scores, the higher is the better, there is a mathematical terms and it is a little
# complicated. Link to read these scores is
#
# https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
# '''
# # The prediction is given below!
# predicted= model.labels_
# my_data["Labels"]=pd.Series(predicted)
# # Let's sort our overall data
# my_data.sort_values(by="Labels",ascending=True,inplace=True,ignore_index=True)
#
# # Let me export it into the output directory!
# my_data.to_excel(output_path)
