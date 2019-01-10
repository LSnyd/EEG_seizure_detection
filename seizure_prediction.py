import data_wrangling
import evaluation
import cross_validation

from sklearn.ensemble import BaggingClassifier
from wildboar.tree import ShapeletTreeClassifier

from wildboar._utils import print_tree


#Get wrangled data with labels and dimensions
data, label, x_dimensions = data_wrangling.wrangled_data()



#Build shapelet trees
tree = ShapeletTreeClassifier(
    random_state=10,
    n_shapelets=50,
    min_shapelet_size=0,
    max_shapelet_size=1,
    force_dim=x_dimensions,
    metric="euclidean",
    #metric="scaled_euclidean",
    #metric="scaled_dtw",
    #metric_params={"r": 3},
)

#Build gRSF
bag = BaggingClassifier(
    base_estimator=tree,
    bootstrap=True,
    n_jobs=16,
    n_estimators=100,
    random_state=100,
)

#Results from cross validation
true, pred = cross_validation.kfold(data,label, bag)



#Print trees for debugging
#    for tree in bag.estimators_:
         #   print_tree(tree.root_node_)
         #   print(tree.root_node_.shapelet.array)

#Evaluation
evaluation.auc(true, pred)
evaluation.rocplot(true, pred)
