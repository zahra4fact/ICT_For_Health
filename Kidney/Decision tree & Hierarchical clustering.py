from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.cluster import KMeans
from math import floor
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

#determine feat names and target
feat_names = ['age','bp','sg','al','su','rbc','pc','pcc',
              'ba','bgr','bu','sc','sod','pot','hemo','pcv',
              'wc','rc','htn','dm','cad','appet','pe','ane','class']

target_names = ['ckd','notckd']

# Reading and cleaning data
df = pd.read_csv('chronic_kidney_disease.csv',
    ",",
    header=None,
    na_values=['?','\t?'],
    names=feat_names,
    skiprows=29)

print(df.info())


# Mapping the values
df.replace(to_replace=['normal','present','yes','good','ckd'], value=0, inplace=True)
df.replace(to_replace=['abnormal','notpresent','no','poor','notckd'], value=1, inplace=True)

# describe method
#print df.describe().T
#print "\n"


#  Managing NaN values
# Option 1 # Removing all the rows with NaN values
#df = df.dropna()

#Option 2 # substitute it with a number that is not already present in the datase
for i in range(0,25):
    # specific gravity
    if df.columns[i] == 'sg':
        for j in range(0, len(df[pd.isnull(df[df.columns[i]])].index.values)):
            index = df[pd.isnull(df[df.columns[i]])].index.values[0]
            df.at[index, df.columns[i]] = random.choice([1.005,1.010,1.015,1.020,1.025])
    # albumin and sugar
    elif df.columns[i] in ['al','su']:
        for j in range(0, len(df[pd.isnull(df[df.columns[i]])].index.values)):
            index = df[pd.isnull(df[df.columns[i]])].index.values[0]
            df.at[index, df.columns[i]] = random.randint(1, 5)
    # categorical df
    elif df.columns[i] in ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','class']:
        for j in range(0, len(df[pd.isnull(df[df.columns[i]])].index.values)):
            index = df[pd.isnull(df[df.columns[i]])].index.values[0]
            df.at[index, df.columns[i]] = random.randint(0, 1)
    else:
        for j in range(0, len(df[pd.isnull(df[df.columns[i]])].index.values)):
            index = df[pd.isnull(df[df.columns[i]])].index.values[0]
            df.at[index, df.columns[i]] = random.uniform(
              df[df.columns[i]].mean() - df[df.columns[i]].std(),
              df[df.columns[i]].mean() + df[df.columns[i]].std())
             

# Defining data and the target
target = df['class']
data = df.drop('class', axis=1)

# Decison tree
clf = tree.DecisionTreeClassifier("entropy")
clf = clf.fit(data, target)
dot_data = tree.export_graphviz(
    clf,
    out_file="Tree.dot",
    feature_names= [feat_names[i] for i in range(0,24)],
    class_names=target_names,
    filled=True,
    rounded=True,
    special_characters=True)
#graph = graphviz.Source(dot_data) 
# Plot predictions
predict = clf.predict(data)
#
print ("Healthy -> " + str(np.count_nonzero(predict == 1)))
print ("Disease -> " + str(np.count_nonzero(predict == 0)))
#

labels = 'Disease', 'Healthy'
sizes = [np.count_nonzero(predict == 0), np.count_nonzero(predict == 1)]
colors = ['red', 'yellow']
#
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
#
plt.axis('equal')
plt.show()

#Plot the feature importances
feature_importances = clf.feature_importances_
indexes = np.argsort(feature_importances)[::-1]
plt.figure()
plt.title("Feature importances")
plt.bar(range(data.shape[1]), feature_importances[indexes],
       color="g", align="center")
plt.xticks(range(data.shape[1]), indexes)
plt.xlim([-1, data.shape[1]])
plt.show()



##################################
####### Lab 4 - Clustering ####### agglomerative clustering
##################################

#Z = linkage(data,'single') # Contains the Hierarchical clustering information
#Z = linkage(data,'complete') # Contains the Hierarchical clustering information
#Z = linkage(data,'average') # Contains the Hierarchical clustering information
#Z = linkage(data,'weighted') # Contains the Hierarchical clustering information
#Z = linkage(data,'centroid') # Contains the Hierarchical clustering information
#Z = linkage(data,'median') # Contains the Hierarchical clustering information
Z = linkage(data ,'ward') # Contains the Hierarchical clustering information

plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z)
plt.show()
