import os
os.getcwd()

# import general_module which locate at my parent directory's child
import sys
import numpy as np
sys.path.append("..")
import pickle
from general_module.evaluation import *
from general_module.training import *

from sklearn.model_selection import train_test_split
def split(dataset):
    return train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset.ptype)


essays = extract("../dataset/merged/essays_otrain.pickle")
mbti = extract("../dataset/merged/mbti_otrain.pickle")

essays_psychofeature_trainset,essays_psychofeature_validationset = split(essays)
mbti_psychofeature_trainset,mbti_psychofeature_validationset = split(mbti)

pickle.dump(essays_psychofeature_trainset, open("../dataset/merged/essays_train.pickle", 'wb'))
pickle.dump(essays_psychofeature_validationset, open("../dataset/merged/essays_validation.pickle", 'wb'))
pickle.dump(mbti_psychofeature_trainset, open("../dataset/merged/mbti_train.pickle", 'wb'))
pickle.dump(mbti_psychofeature_validationset, open("../dataset/merged/mbti_validation.pickle", 'wb'))


