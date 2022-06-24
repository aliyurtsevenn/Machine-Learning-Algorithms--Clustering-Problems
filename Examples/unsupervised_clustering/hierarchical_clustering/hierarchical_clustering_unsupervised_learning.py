import sys
import subprocess

# Let me first install the required packages! 
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'numpy>=1.19.4'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'pandas>=1.1.4'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'sklearn>=0.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'kneed>=0.7.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'pathlib>=1.0.1'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'matplotlib>=3.3.3'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'scipy>=1.5.4'])

# Let me install the packages
import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import scipy
import os
from os.path import expanduser
from sklearn import linear_model,preprocessing
import sklearn
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage,leaves_list
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from pylab import rcParams
import sklearn.metrics as sm
'''
Hierarchical clustering

- These methods predict subgroups within the data, by finding distance between each data point and
its nearest and linking the most nearby neighbors.
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
separation_csv=data[data.columns[0]][5]

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
Attributes=scale(Attributes)

# # Shuffle the data!
Attributes=shuffle(Attributes)


# Shaping the data! according to the instance numbers!
samples,features=Attributes.shape # This gives you have many features and samples you have

Z=linkage(Attributes,"ward")

dendrogram(Z,p=12,truncate_mode="lastp",leaf_rotation=45,leaf_font_size=12,show_contracted=True)
plt.title("Truncated Hierarchy Clustering Dendogram")
plt.xlabel("Cluster Size",fontsize=14)
plt.ylabel("Distance",fontsize=14)

plt.axhline(y=500)
plt.axhline(y=150)
plt.subplots_adjust(bottom=0.2)
plt.show()

try:
    while True:
        give_cluster_number = int(input(
            "Give me cluster number bigger than 1. You should enter the cluster number according to the dendogram you obtained. You can draw the horizontal line from the lonhest heights of the 2 childs in the tree and"
            "see how many edges are averlaping the line: "))
        if give_cluster_number<2:
            print("You should enter an integer bigger than 1")
        break
except Exception as p:
    print("ERROR! You need to enter an integer number bigger than 1 as number of cluster!")

# Let me now fit our model from the given cluster number with various data!

my_linkages=["ward","complete","average","single"]
affinity=["euclidean","manhattan","l1","l2","cosine"] # Since there should be reshaping in our attributes, we excepted precompute method!

my_coeffs=[]
applied=[]
for j in affinity:
    for z in my_linkages:
        sub = []
        if z=="ward" and j=="euclidean":
            Hier_model = AgglomerativeClustering(n_clusters=give_cluster_number, affinity=j, linkage=z)
            Hier_model.fit(Attributes)
            coeff = sm.silhouette_score(Attributes, Hier_model.labels_)
            my_coeffs.append(coeff)
            sub.append(j)
            sub.append(z)
            applied.append(sub)
        elif z!="ward":
            Hier_model = AgglomerativeClustering(n_clusters=give_cluster_number, affinity=j, linkage=z)
            Hier_model.fit(Attributes)
            coeff = sm.silhouette_score(Attributes, Hier_model.labels_)
            sub.append(j)
            sub.append(z)
            applied.append(sub)
            my_coeffs.append(coeff)

print("--PARAMETERS APPLIED FOR THE MODEL--\n\n")

ind=[]
for i in range(len(my_coeffs)):
    ind.append(i)
data_high=pd.DataFrame({"coef":my_coeffs,"ind":ind})
data_high.sort_values(by="coef",inplace=True,ascending=True)
all_ind=data_high["ind"].tolist()

inted=[]
for j in all_ind:
    s=int(j)
    inted.append(s)

indexes=[]
for j,k in enumerate(my_coeffs):
    if k==max(my_coeffs):
        indexes.append(j)

appropriate_labels=[]
for i in indexes:
    name= applied[i]
    appropriate_labels.append(name)

for j in range(0,len(appropriate_labels)):
    model = AgglomerativeClustering(n_clusters=give_cluster_number, affinity=appropriate_labels[j][0], linkage=appropriate_labels[j][1])
    model.fit(Attributes)
    labels=model.labels_
    my_data["predictions"]=pd.Series(labels)

    uni= my_data["predictions"].unique()
    respond="Yes"
    for i in uni:
        k=len(my_data[my_data["predictions"]==i])
        if k<10:
            respond="No"
    if respond=="Yes":
        link=appropriate_labels[j][1]
        aff=appropriate_labels[j][0]
        break
if respond=="No":
    for j in inted:
        model = AgglomerativeClustering(n_clusters=give_cluster_number, affinity=applied[j][0],
                                        linkage=applied[j][1])
        model.fit(Attributes)
        labels = model.labels_
        my_data["predictions"] = pd.Series(labels)

        uni = my_data["predictions"].unique()
        respond = "Yes"
        for i in uni:
            k = len(my_data[my_data["predictions"] == i])
            if k < 10:
                respond = "No"
        if respond == "Yes":
            link = applied[j][1]
            aff = applied[j][0]

            break

# Let's write the good clustering method here with silhoutte score!

model = AgglomerativeClustering(n_clusters=give_cluster_number, affinity=aff, linkage=link)
model.fit(Attributes)
labels=model.labels_
coeff = sm.silhouette_score(Attributes, labels)
print("Silhouette score of the model is: {}".format(max(my_coeffs)))
print("Considering the highest Silhouette score and cluster sizes, affinity method and linkage selected orderly as {} and {} ".format(aff,link))

my_data["predictions"]=pd.Series(labels)


my_data.sort_values(by="predictions",ascending=True,inplace=True,ignore_index=True)


# Let me export it into the output directory!
my_data.to_excel(output_path)
