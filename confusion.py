import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pathlib import Path
import os
from imutils import paths
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
labels=[]

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", type=str,
	help="folder to create confusion")
args = vars(ap.parse_args())

for name in os.listdir("./temp") :
    if os.path.isdir("./temp/"+name):
        labels.append(name)
labels.append("Unknown")
# for i in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]:
for i in [0.06,0.07]:
    y_true=open("answer_co_0deg_hog.txt","rb")
    y_pred=open(args["folder"]+"./ouput_prob"+str(i)+"/output_"+str(i)+".txt","rb")
    y_true=json.load(y_true)
    y_pred=json.load(y_pred)
    y_true=y_true['names']
    y_pred=y_pred['names']

    a=confusion_matrix(y_true, y_pred, labels=labels)
    from sklearn.metrics import accuracy_score
    print(i,accuracy_score(y_true, y_pred))

    cm=confusion_matrix(y_true,y_pred)
    print(cm)
    print('\n')
    print(classification_report(y_true, y_pred))

df_cm = pd.DataFrame(a, index = [i for i in labels],
                  columns = [i for i in labels])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()


