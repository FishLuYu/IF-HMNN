import tensorflow as tf
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras.layers import concatenate
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import feature_column
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import DenseFeatures,Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/graphviz/bin/'
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import sklearn
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix,classification_report,roc_auc_score


path_Multi_Imu_Mag=r'data.csv'

trn=pd.read_csv(path_Multi_Imu_Mag,header=0)
# trn=trn[(int)(trn.shape[0]/2)+1000:-1]
trn.head(5)

# trn=trn.drop(labels='timestamp' , axis = 1)
# trn=trn.drop(labels='LineNo' , axis = 1)
# trn=trn.drop(labels='TimeUS' , axis = 1)
# trn=trn.drop(labels='LineNo.1' , axis = 1)
# trn=trn.drop(labels='timestamp' , axis = 1)
# trn=trn.drop(labels='TimeUS.1' , axis = 1)
trn = trn.loc[:, (trn != trn.iloc[0]).any()]
trn=trn.sample(frac=1.0)

X_trn = trn.drop(['labels'] , axis = 1).values
Y_trn = trn['labels'].values

scaler = StandardScaler()

X_trn = scaler.fit_transform(X_trn)
X_trn.shape

train_data,test_data,train_label,test_label=train_test_split(X_trn,Y_trn,test_size=0.2)

# Create Base Classifiers
def create_dnn_model():
#     weights_class={0:2,1:2,2:3,3:2,4:2}
    model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Input(shape=(X_trn.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     model1.fit(train_data,train_label,epochs=200,validation_split=0.2,class_weight=weights_class)    
    return model

def create_cnn_model():
#     weights_class={0:2,1:2,2:3,3:2,4:2}
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((X_trn.shape[1],1)),
        tf.keras.layers.Conv1D(filters=6,kernel_size=4,activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     model1.fit(train_data,train_label,epochs=200,validation_split=0.2,class_weight=weights_class)    
    return model

# create the ensemble model
model1 = KerasClassifier(build_fn=create_dnn_model, epochs=200)
model1._estimator_type = "classifier"
model2 = KerasClassifier(build_fn=create_cnn_model, epochs=200)
model2._estimator_type = "classifier"
# model3 = KerasClassifier(build_fn=create_dnn_model, epochs=200)
# model3._estimator_type = "classifier"
# model4 = KerasClassifier(build_fn=create_cnn_model, epochs=200)
# model4._estimator_type = "classifier"

# the weight of each class, if not given, all classes are supposed to have the same weight.
weights_class={0:0.6,
               1:0.1,
               2:0.1,
               3:0.1,
               4:0.1,
}

history1=model1.fit(train_data,train_label,epochs=200,validation_split=0.2,class_weight=weights_class)#
history2=model2.fit(train_data,train_label,epochs=200,validation_split=0.2,class_weight=weights_class)#

test_acc1=model1.score(test_data,test_label)
test_acc2=model2.score(test_data,test_label)
scoress=[test_acc1,test_acc2]
score=[(scoress[0])/sum(scoress),(scoress[1])/sum(scoress)]

models = [('model1', model1), ('model2', model2)]
ensemble = VotingClassifier(estimators=models,weights=score, voting='soft')  # weights=score

ensemble.fit(train_data, train_label)

ensemble.score(test_data, test_label)

# 评估投票分类器
test_predictions = ensemble.predict(test_data)
accuracy = accuracy_score(test_label, test_predictions)
print('Accuracy:', accuracy)

print("ACC", accuracy_score(test_label, test_predictions))    #准确率
# print("REC", recall_score(test_label, test_predictions),average='weighted')      #召回率
# print("F-score", f1_score(test_label, test_predictions),average='weighted')      #F1-score
cm=confusion_matrix(test_label, test_predictions) 

test_predictions = ensemble.predict(test_data)
# test_prediction=np.argmax(test_predictions,axis=1)
accuracy = accuracy_score(test_label,test_predictions)
print('Test Accuracy:', accuracy)

classificationReport=classification_report(test_label,test_predictions)
print('Classification Report:\n', classificationReport)

cm=confusion_matrix(test_label,test_predictions)

precision = precision_score(test_label, test_predictions, average='macro')
print('precision:', precision)

f1=f1_score(test_label,test_predictions,average='micro')
print("F1:",f1)

recall = recall_score(test_label, test_predictions, average='macro')
print('recall:', recall)

labels=[0,1,2]
test_label=label_binarize(test_label,classes=labels)
test_predictions=label_binarize(test_predictions,classes=labels)
roc_auc=roc_auc_score(test_label, test_predictions,multi_class='ovo')
print('Roc_auc:', roc_auc)