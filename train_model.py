# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn import tree
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")

#SVM Method
# recognizer = RandomForestClassifier(n_estimators=100)

# Bagging
recognizer = BaggingClassifier(tree.DecisionTreeClassifier(max_depth = 23))


#GridSearch 5%

# params = {"C": [1e3, 5e3, 1e4, 5e4, 1e5],
# 	"gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01,0.1]}
# recognizer = GridSearchCV(SVC(kernel="rbf", gamma="auto",class_weight='balanced',
# 	probability=True), params, cv=3, n_jobs=3)



recognizer.fit(data["embeddings"], labels)

print("[INFO] hyperparameters: {}".format(recognizer))
# print("[INFO] best hyperparameters: {}".format(recognizer.best_params_))

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
# f.write(pickle.dumps(recognizer.best_estimator_))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()