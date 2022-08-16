

import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras import Input
import tensorflow as tf
from utils import endofphrase,startofphrase,list_files,word2sent,preprocess,performance_micro
import matplotlib.pyplot as plt
import keras
import csv





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

    X_test = pad_sequences(maxlen=max_len, sequences=X_test, padding="post", value=0, truncating='post')
    y_test = [[tag2idx1[w[1]] for w in s] for s in testsentences]
    y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=tag2idx1["O"], truncating='post')
    y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]

    return X_test,y_test

#------------------------------------------------------------------------------------------------------------------------------------------------------------


#testdata=preprocess('eng.testb',delimiter=' ', format='conll')
testdata=preprocess('edgar_test_4.csv',delimiter=' ', format='edgar')

testsentences=word2sent(testdata)
testlengths=[len(x) for x in testsentences]


"""
#do this when we train on conll and test on edgar
for j in range(0,len(testdata)):
    truetag=testdata[j][1]
    if 'LOCATION' in truetag:
            testdata[j]=[testdata[j][0],truetag.replace('LOCATION','LOC')]
    if 'PERSON' in truetag:
            testdata[j]=[testdata[j][0],truetag.replace('PERSON','PER')]
    if 'BUSINESS' in truetag:
            testdata[j]=[testdata[j][0],truetag.replace('BUSINESS','ORG')]
    if 'COURT' in truetag:
            testdata[j]=[testdata[j][0],truetag.replace('COURT','ORG')]
    if 'GOVERNMENT' in truetag:
            testdata[j]=[testdata[j][0],truetag.replace('GOVERNMENT','ORG')]
    if 'MISCELLANEOUS' in truetag:
            testdata[j]=[testdata[j][0],truetag.replace('MISCELLANEOUS','MISC')]
    if 'LEGISLATION/ACT' in truetag:
            testdata[j]=[testdata[j][0],truetag.replace('LEGISLATION/ACT','MISC')]
    #y_test.append(truelist)
"""





#---------------------------------------------------------------------------------------------------------
#traindata=preprocess('eng.train',delimiter=' ', format='conll')
traindata=preprocess('data/edgar_train_4.csv',delimiter=' ', format='edgar')

trainsentences=word2sent(traindata)

#validdata=preprocess('eng.testa',delimiter=' ', format='conll')
validdata=preprocess('edgar_valid_4.csv',delimiter=' ', format='edgar')

validsentences=word2sent(validdata)

#maxlist = max(sentences, key = len)
#ind=sentences.index(maxlist)

df=pd.DataFrame(data=traindata)
df = df.rename(columns={0: 'Word', 1: 'Tag'})
maxlength = max(len(x) for x in trainsentences)

words_train = list(set(df["Word"].values))
words=words_train


words.insert(0,'-PAD-')
words.append('-UNKNOWN-')

n_words = len(words)

    #for tags, explicitly list out all the possible tags
tags=['O','I-PER','B-PER','I-LOC','B-LOC','I-ORG','B-ORG','I-MISC','B-MISC']
#tags=['O','I-PERSON','B-PERSON','I-LOCATION','B-LOCATION','I-BUSINESS','B-BUSINESS','I-MISCELLANEOUS','B-MISCELLANEOUS','I-COURT', 'B-COURT','I-GOVERNMENT','B-GOVERNMENT','I-LEGISLATION/ACT','B-LEGISLATION/ACT']

#tags=swapPositions(tags,0,tags.index('O'))

n_tags = len(tags)

word2idx = {w: i  for i, w in enumerate(words)}  #the 0 is for the pad token
idx2word= {i: w for w, i in word2idx.items()}

tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: w for w, i in tag2idx.items()}




X_train = [[word2idx[w[0]] for w in s] for s in trainsentences]
X_train = pad_sequences(sequences=X_train, padding="post", value=0)
y_train = [[tag2idx[w[1]] for w in s] for s in trainsentences]
y_train = pad_sequences(sequences=y_train, padding="post", value=tag2idx["O"])
y_train = [to_categorical(i, num_classes=n_tags) for i in y_train]



    # each sentence is padded to become the "max sentence length" in the train set. for conll this is 113. If there is a sentence that is longer than this in the test set,
    #the approach used here, is to cut off the tail of the long test sentence. (truncating=post)
#-----------------------------------------------------------------------------------------------------------------


X_test,y_test=createtest(testdata,maxlength,n_tags,word2idx,tag2idx)
X_valid,y_valid=createtest(validdata,maxlength,n_tags,word2idx,tag2idx)

    #the word "-UNKNOWN-" has word2idx=23626
    #this is to find the position unseen vocab in train set. we see that eng.testb has 5655 words being unseen in eng.train
    #the first tuple shows the row of the i^th unseen word.
    #the second tuple shows the column of the i^th unseen word.

    #result=np.where(X_test==23626)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=n_words + 1, output_dim=32,input_length=maxlength, mask_zero=True))
model.add(Dense(128,activation='softmax'))
model.add(Dense(9))

    #model = tf.keras.models.Sequential()

    #model.add(Embedding(input_dim=n_words + 1, output_dim=32, input_length=max_len, mask_zero=True)(input))  # 20-dim embedding

    #model=Dense(128,activation='relu')


   # model = Bidirectional(LSTM(units=50, return_sequences=True,recurrent_dropout=0.1))(model)  # biLSTM
    #model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    #crf = CRF(n_tags)  # CRF layer
    #out = crf(model)  # output

    #model = Model(input, out)
opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)



model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy())

#model.compile(optimizer=opt, loss=custom_loss)

model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(X_train, np.array(y_train), batch_size=32, epochs=50,
                         verbose=1,validation_data=(np.array(X_valid), np.array(y_valid)),callbacks=callback)





#hist = pd.DataFrame(history.history)



    #-------------------------------------------------------------------------------------------------------------------------------------------------------------


test_pred = model.predict(X_test, verbose=1)
labels_pred = pred2label(test_pred,idx2tag)

labels_true = pred2label(y_test,idx2tag)

for i in range(0,len(labels_true)):
    length=testlengths[i]
    predseq=labels_pred[i]
    trueseq=labels_true[i]
    predseq=predseq[:length]
    trueseq=trueseq[:length]
    labels_pred[i]=predseq
    labels_true[i]=trueseq



tagged_seq=labels_pred


"""

y_test=[]
for i in testsentences:
    lab=[x[1] for x in i]
    y_test.append(lab)

"""


relevant=0
retrieved=0
tp=0
fp=0
fn=0
for i in range(0,len(tagged_seq)):
    results1=performance(tagged_seq[i],labels_true[i])
    relevant+=results1[4]
    retrieved+=results1[5]
    tp+=results1[6]

    results2=performance_micro(tagged_seq[i],labels_true[i])
    fp+=results2[0]
    fn+=results2[1]


print(tp,fp,fn)
recall=tp/relevant
precision=tp/retrieved
fscore = 2 * precision * recall / (precision + recall)
print(precision,recall,fscore)



"""
with open("/Users/terenceau1/PycharmProjects/pythonProject/nn/conll_train_conll_pred/pred.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(tagged_seq)

with open("/Users/terenceau1/PycharmProjects/pythonProject/nn/conll_train_conll_pred/pred.csv", 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    tagged_seq = list(csv_reader)

"""



