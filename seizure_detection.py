import numpy as np
import time
import glob


import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier
from wildboar.tree import ShapeletTreeClassifier
from numpy import genfromtxt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score
from wildboar._utils import print_tree



source_dir = "/home/lisa/Dokumente/SU_CD/DAMI2/EEG_seizure_detection"


label = np.genfromtxt("/home/lisa/Dokumente/SU_CD/DAMI2/EEG_seizure_detection/EEG_Data/labels.txt", delimiter="\n", dtype=int)


#Map to multivariate time series
data = []
file_list = sorted(glob.glob(source_dir + '/EEG_Data' + '/Data' + '/*.txt'))


for file_path in file_list:
  data.append(
     np.genfromtxt(file_path, delimiter="\n", dtype=int))

np.stack(data)

data = np.stack(data)
data = np.vsplit(data, 25)
data = np.stack(data)


#Change data type & reshape data
data = np.array(data, dtype=np.float64)

x_samples, x_dimensions, x_timesteps = data.shape

data = data.reshape(x_samples, x_dimensions * x_timesteps)

# Random shuffling of samples
random_state = np.random.RandomState(123)
order = np.arange(x_samples)
random_state.shuffle(order)

data = data[order, :]
label = label[order]

tree = ShapeletTreeClassifier(
    random_state=10,
    n_shapelets=50,
    min_shapelet_size=0,
    max_shapelet_size=1,
    force_dim=x_dimensions,
    metric="euclidean",
   # metric="scaled_euclidean",
    #metric="scaled_dtw",
    #metric_params={"r": 3},
)


bag = BaggingClassifier(
    base_estimator=tree,
    bootstrap=True,
    n_jobs=16,
    n_estimators=100,
    random_state=100,
)



pred = []
true = []
prob = []

kf = KFold(n_splits=4)
kf.get_n_splits(data)
KFold(n_splits=4, random_state=None, shuffle=False)

for train_index, test_index in kf.split(data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    bag.fit(X_train, y_train)
    prediction = bag.predict(X_test)
    prob.extend(bag.predict_proba(X_test))
    pred.extend(prediction)
    true.extend(y_test)


    print("Real Class:", y_test)
    print("Predicted class:", prediction)
    print("Probabilities", bag.predict_proba(X_test))
    print(pred)
    print(true)
    print(prob)

#Print tree
#    for tree in bag.estimators_:
         #   print_tree(tree.root_node_)
         #   print(tree.root_node_.shapelet.array)



print("AUC score", roc_auc_score(true, pred))

# AUC, ROC
fpr, tpr, _ = roc_curve(true, pred)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

