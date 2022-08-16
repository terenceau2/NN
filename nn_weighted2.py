

import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras import Input
import tensorflow as tf
#from utils import word2sent,preprocess


import matplotlib.pyplot as plt
import keras
import csv

"""
import pickle
import operator
import re
import string
import matplotlib.pyplot as plt

from plot_keras_history import plot_history
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix

from keras import layers
from keras import optimizers

from keras.models import Model

from tensorflow.keras import Input

from tensorflow_addons.layers import CRF
from tensorflow_addons import losses
from tensorflow_addons import metrics
from tf_crf_layer.loss import crf_loss
from tf_crf_layer.metrics import crf_accuracy
"""

def preprocess(filename,delimiter=',',format='conll'):
    """
    if filename.endswith('.csv'):
        with open(filename, 'rt') as f:
            data = csv.reader(f,delimiter=delimiter)
            data=list(data)

    else:
       """
    with open(filename) as myfile:
            data = myfile.readlines()
            data = [i.rstrip('\n') for i in data]

    if format=='edgar':
                data = [i.rsplit(delimiter,1) for i in data]
    elif format=='conll' or format=='others':
                data = [i.split(delimiter) for i in data]



    if format=='conll':
            for i in data:
                if i != [''] and i!=[]:
                    del i[1]
                    del i[1]  # delete the middle 2 columns from the data
    for i in range(0, len(data)):
                if data[i] == [''] or data[i]==[]:
                    data[i] = ["", "O"]


    return data

def word2sent(wordlist):
    df = pd.DataFrame(data=wordlist)
    sentences = []
    buffer = []
    for i in range(0, df.shape[0]):
        if df.iloc[i, 0] != '' and df.iloc[i,0]!=None:
            a = tuple(df.loc[i].to_list())
            buffer.append(a)
        else:
            sentences.append(buffer)
            buffer = []

    if buffer!=[]:
        sentences.append(buffer)
    sentences=[x for x in sentences if len(x)!=0]

    return sentences



def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def pred2label(pred,idx2tag1):
    out = []
    for pred_i in pred:
                        #pred is 3d array, number of layers= no. of sentences in test set
                        #each layer is the prediction matrix of size (max_sentence_length x no. of tags)


        out_i = []
        for p in pred_i:  #loop for each row
            p_i = np.argmax(p)
            out_i.append(idx2tag1[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out





def createtest(testdata,max_len,n_tags,word2idx1,tag2idx1):

    testsentences=word2sent(testdata)

    maxlengthtest = max(len(x) for x in testsentences)
    maxlist = max((x) for x in testsentences)

    X_test=[]
    for i in range(0,len(testsentences)):
        s=testsentences[i]
        sent=[]
        for w in s:
            if w[0] in word2idx1:
                a=word2idx1[w[0]]
                sent.append(a)

            elif w[0] not in word2idx1:   #this is to deal with unseen vocab in the train set.
                a=word2idx1['-UNKNOWN-']
                sent.append(a)

        X_test.append(sent)

    #X_test = pad_sequences(maxlen=max_len, sequences=X_test, padding="post", value=0, truncating='post')
    y_test = [[tag2idx1[w[1]] for w in s] for s in testsentences]
    y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=tag2idx1["O"], truncating='post')
    y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]

    return X_test,y_test

#------------------------------------------------------------------------------------------------------------------------------------------------------------


#testdata=preprocess('eng.testb',delimiter=' ', format='conll')
testdata=preprocess('/Users/terenceau1/PycharmProjects/pythonProject/data/synthetic_big_test.csv',delimiter=',', format='edgar')

testsentences=word2sent(testdata)
testlengths=[len(x) for x in testsentences]






#---------------------------------------------------------------------------------------------------------
#traindata=preprocess('eng.train',delimiter=' ', format='conll')
traindata=preprocess('/Users/terenceau1/PycharmProjects/pythonProject/data/synthetic_big_train.csv',delimiter=',', format='edgar')

trainsentences=word2sent(traindata)

#validdata=preprocess('eng.testa',delimiter=' ', format='conll')
validdata=preprocess('/Users/terenceau1/PycharmProjects/pythonProject/data/synthetic_big_valid.csv',delimiter=',', format='edgar')

validsentences=word2sent(validdata)




df=pd.DataFrame(data=traindata)
df = df.rename(columns={0: 'Word', 1: 'Tag'})
maxlength = max(len(x) for x in trainsentences)

words_train = list(set(df["Word"].values))
words_train.sort()
words=words_train
words.remove('')

n_words = len(words)

    #for tags, explicitly list out all the possible tags
tags=['O','I']

n_tags = len(tags)

word2idx = {w: i  for i, w in enumerate(words)}  #the 0 is for the pad token
idx2word= {i: w for w, i in word2idx.items()}

tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: w for w, i in tag2idx.items()}



#X = [[word[0] for word in sentence] for sentence in trainsentences]
#y = [[word[1] for word in sentence] for sentence in trainsentences]



X_train = [[word2idx[w[0]] for w in s] for s in trainsentences]
#X_train = pad_sequences(sequences=X_train, padding="post", value=0)
#X_train1 = [to_categorical(i, num_classes=n_words) for i in X_train]


y_train = [[tag2idx[w[1]] for w in s] for s in trainsentences]
#y_train = pad_sequences(sequences=y_train, padding="post", value=tag2idx["O"])
y_train = [to_categorical(i, num_classes=n_tags) for i in y_train]

#another way of 1-hot encoding it
#TAG_COUNT = len(tag2idx)
#y = [ np.eye(TAG_COUNT)[sentence] for sentence in y_train]




X_test,y_test=createtest(testdata,maxlength,n_tags,word2idx,tag2idx)
X_valid,y_valid=createtest(validdata,maxlength,n_tags,word2idx,tag2idx)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

#-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_-----_

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(maxlength,)))
model.add(tf.keras.layers.Embedding(input_dim=n_words + 1, output_dim=32,input_length=maxlength, mask_zero=True))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(2))

"""
WORD_COUNT = len(word2idx)
DENSE_EMBEDDING = 50
LSTM_UNITS = 50
LSTM_DROPOUT = 0.1
DENSE_UNITS = 100
BATCH_SIZE = 256
MAX_EPOCHS = 5

input_layer = layers.Input(shape=(10,))
model = layers.Embedding(WORD_COUNT, DENSE_EMBEDDING, embeddings_initializer="uniform", input_length=10)(input_layer)
model = layers.Bidirectional(layers.LSTM(LSTM_UNITS, recurrent_dropout=LSTM_DROPOUT, return_sequences=True))(model)
model = layers.TimeDistributed(layers.Dense(DENSE_UNITS, activation="relu"))(model)
crf_layer = CRF(units=2)
_, output_layer, _, _ = crf_layer(model)

ner_model = Model(input_layer, output_layer)

loss = crf_loss
acc_metric = crf_accuracy

ner_model.compile(optimizer=opt, loss=loss, metrics=[acc_metric])

ner_model.summary()
history = ner_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_split=0.1, verbose=2)
"""


opt = tf.keras.optimizers.Adam(learning_rate=0.001)
def custom_loss(y_pred,y_true):
    loss=tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred,pos_weight=1)
    return loss
model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
#model.compile(optimizer=opt, loss=custom_loss)

model.summary()



"""
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_3 (Embedding)     (None, 10, 32)            224       
                                                                 
 dense_7 (Dense)             (None, 10, 50)            1650      
                                                                 
 dense_8 (Dense)             (None, 10, 50)            2550      
                                                                 
 dense_9 (Dense)             (None, 10, 2)             102       
                                                                 
=================================================================
Total params: 4,526
Trainable params: 4,526
Non-trainable params: 0
_________________________________________________________________

"""


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(X_train, np.array(y_train), batch_size=32, epochs=50,
                         verbose=1,validation_data=(np.array(X_valid), np.array(y_valid)),callbacks=callback)

#-----------------------------------------------------------------------------
test_pred = model.predict(X_test, verbose=1)
labels_pred = pred2label(test_pred,idx2tag)

labels_true = pred2label(y_test,idx2tag)



relevant=0
retrieved=0
accuracy=0
tp=0
fp=0
fn=0
for i in range(0,len(labels_pred)):
    ground_truth=labels_true[i]
    pred=labels_pred[i]

    for j in range(0,len(ground_truth)):
        if ground_truth[j]=='I':
            relevant+=1
        if pred[j]=='I':
            retrieved+=1
        if ground_truth[j]==pred[j]:
            accuracy+=1
            if ground_truth[j]=='I':
                tp+=1
        if ground_truth[j]=='I' and pred[j]=='O':
            fn+=1
        if ground_truth[j]=='O' and pred[j]=='I':
            fp+=1



from collections import Counter

#accuracy=accuracy/10000
print(tp,fp,fn,accuracy)
recall=tp/relevant
precision=tp/retrieved
fscore = 2 * precision * recall / (precision + recall)
print(precision,recall,fscore)

count=Counter(x for xs in labels_pred for x in xs)
print(count)

#=========================================================================================================================================


#for the baseline one
#1100 5730 2573 1697
#0.16105417276720352 0.29948271167982576 0.20946396267733028
#Counter({'I': 6830, 'O': 3170})

