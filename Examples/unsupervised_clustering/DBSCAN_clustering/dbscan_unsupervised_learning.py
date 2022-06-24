'''

Advantages of DBSCAN
1. Can find randomly shaped  clusters
2. Find clusters which are completely surrounded by various clusters
3. Roboust towards outlier recognition
4. Need just 2 points which are very insensitive to the ordering of the points in databases.

Disadvantages of DBSCAN
1. Datasets which have chaning densities are misleading
2. Fails to find clusters if density varries and if the data set is too sparse (most cells of a varible is made up of NaN values)
3. Sampling affects the density measures


Note that high density of data means more accuracy.

Data density is very important and it shows how many items of a information set are examined.
Selecting these items, which stand for attributes in the machine learning is called as data sampling
'''

import sys
import subprocess

# Let me first install the suitable packages!! 
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'future'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'numpy>=1.19.4'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'pandas>=1.1.4'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'pandas>=0.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'pathlib>=1.0.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'matplotlib>=3.3.3'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'scipy>=1.5.4'])

# Let me import the packages!
from sklearn import metrics
import pandas as pd
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.preprocessing import scale, StandardScaler
import matplotlib.pyplot as plt
from  pylab import  rcParams
import numpy as np
from sklearn.utils import shuffle
import os
import pathlib
from sklearn import preprocessing
import tkinter as tk

from os.path import expanduser
import os

# Let me now take the parameters! 
path_parameter_file=os.path.join(pathlib.Path(__file__).parent.absolute(),'parameters.txt')

if os.path.exists(path_parameter_file)==False:
    print('parameters.txt file should be in the same directory with the interactome_parameter.py file! There will be an error!')

data=pd.read_csv(path_parameter_file,delimiter='\t',header=None)
input_path= data[data.columns[0]][0]
output_path= data[data.columns[0]][1]
sample_name_situation= data[data.columns[0]][2]
attribute_madeup= data[data.columns[0]][3]
Categorical_feature= data[data.columns[0]][4].split(",")
separation_csv=data[data.columns[0]][5]

try:
    epsilion_value=float(data[data.columns[0]][6])
except:
    print("Epsilon value should be a value")


try:
    min_cluster_num=int(data[data.columns[0]][7])
except:
    print("Minimum cluster size should be positive integer!")

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

#Lets convert dataframe to numpy array
Attributes= np.array(my_data_frame)
#
#Let's make your data between -1 and 1.
Attributes= scale(Attributes)

# # Shuffle the data!
Attributes=shuffle(Attributes)

# Shaping the data! according to the instance numbers!
samples,features=Attributes.shape # This gives you have many features and samples you have

metric_list=["cityblock","cosine","manhattan","l1","l2","euclidean","braycurtis",
             "canberra","chebyshev","correlation","dice","hamming","jaccard","kulsinski",
             "rogerstanimoto","russellrao","sokalmichener",
             "sokalsneath","sqeuclidean","yule"]

coe_list=[]
for j in  metric_list:
    dbscan_model = DBSCAN(eps=epsilion_value, min_samples=min_cluster_num, metric=j)
    model = dbscan_model.fit(Attributes)
    try:
        coef=metrics.silhouette_score(Attributes,model.labels_)
        coe_list.append(coef)
    except Exception as e:
        coe_list.append(0)
max_coef=max(coe_list)

ind_max=coe_list.index(max_coef)

metr_highest=metric_list[ind_max]

# Create the model
dbscan_model= DBSCAN(eps=epsilion_value, min_samples=min_cluster_num, metric=metr_highest)
model= dbscan_model.fit(Attributes)
# Let me printout the results

my_data_frame["labels"]=pd.Series(model.labels_)
my_data_frame.loc[my_data_frame["labels"]==-1,"Outlier or not"]="Outlier"
my_data_frame.loc[my_data_frame["labels"]!=-1,"Outlier or not"]="Not-Outlier"

my_data_frame.sort_values(by="labels",ascending=True)
my_data_frame.to_excel(output_path)

# Let me calculate Silhouette score and give a result in tkinter box.

try:
    coef=metrics.silhouette_score(Attributes,model.labels_)
    root = tk.Tk()
    root.title("Results")
    T = tk.Text(root, height=15, width=60,font=("Helvetica", 14))
    T.pack()
    T.insert(tk.END,"RESULT\n\n")
    T.insert(tk.END, "Silhouette score is calculated as {}.\n\n".format(coef))
    T.insert(tk.END, "Number of clusters including outliers is calculated as {}\n\n".format(len(list(set(model.labels_)))))
    T.insert(tk.END, "The best metric used for DBSCAN is found as {}\n\n".format(metr_highest))
    T.insert(tk.END, "Users can find the labels in the output file under the output directory!\n\n\n\n\n")
    T.insert(tk.END, "                                                                              Made by Ali Yurtseven")
    tk.mainloop()
except:
    print("There is no more than 1 clusters. Each sample is defined as outliers, so accuracy score "
          "could not calculated!")

