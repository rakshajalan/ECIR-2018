from __future__ import print_function

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Dense, Dropout
import csv
import keras
import re
import csv
import keras
import re
import re
    
from nltk.corpus import stopwords

import string    
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()
def clean_str(string1):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    str_stop = ""
    string1 = re.sub(r'^https?:\/\/.*[\r\n]*', '',string1)
    string1 = re.sub(r"\\", " ", string1)    
    string1 = re.sub(r"\'", " ", string1)    
    string1 = re.sub(r"\"", " ", string1)   
    string1 = re.sub(r'(\W)\1+', r'\1', string1)
    word_list=string1.split(" ")
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    for kj in filtered_words:
         new=lmtzr.lemmatize(str(kj)) 
         str_stop=str_stop +" "+new
         str_stop.encode('utf-8')
    return str_stop.strip().lower()
    
MAX_SEQUENCE_LENGTH = 1600
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')
embeddings_index = {}
f = open('/home/jalan.raksha/2017_research/glove.840B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

uniq_label=["SOCL","PREG","GOAL","TRMT","DEMO","FAML","DISE"]
with open("/home/jalan.raksha/2017_research/ICHI2016_DataSet/ICHI2016-TrainData.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    c=0
    label_train=[]
    text_train=[]
    question=[]
    for line in tsvreader:
      if c>0:
       label_train.append(line[0])  
       string1= str(line[1])+" "+str(line[2])
       string1=clean_str(string1)
       string1=re.sub(r'[^\x00-\x7F]+',' ',string1) #to remove non ascii characters
       text_train.append(string1)  
       texts.append(string1) #stored both title and question together
      c=c+1 

with open("/home/jalan.raksha/2017_research/ICHI2016_DataSet/ICHI2016-TestData_label.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    
    
    text_test=[]
    label_test=[]
    question=[]
    for line in tsvreader:
      
       label_test.append(line[0])  
       string1= str(line[1])+" "+str(line[2])
       string1=clean_str(string1)
       string1=re.sub(r'[^\x00-\x7F]+',' ',string1) #to remove non ascii characters
       texts.append(string1)
       text_test.append(string1)
          #stored both title and question together

      


print('Found %s texts.' % len(texts))


print (len(label_train))

  
Y_train=[]  
for d in range(0,len(label_train)):
    if label_train[d]==uniq_label[0]:
        Y_train.append(0)
    elif label_train[d]==uniq_label[1]:
        Y_train.append(1)
    elif label_train[d]==uniq_label[2]:
        Y_train.append(2)
    elif label_train[d]==uniq_label[3]:
        Y_train.append(3)
    elif label_train[d]==uniq_label[4]:
        Y_train.append(4)
    elif label_train[d]==uniq_label[5]:
        Y_train.append(5)
    elif label_train[d]==uniq_label[6]:
        Y_train.append(6)


  
Y_test=[]  
for d in range(0,len(label_test)):
    if label_test[d]==uniq_label[0]:
        Y_test.append(0)
    elif label_test[d]==uniq_label[1]:
        Y_test.append(1)
    elif label_test[d]==uniq_label[2]:
        Y_test.append(2)
    elif label_test[d]==uniq_label[3]:
        Y_test.append(3)
    elif label_test[d]==uniq_label[4]:
        Y_test.append(4)
    elif label_test[d]==uniq_label[5]:
        Y_test.append(5)
    elif label_test[d]==uniq_label[6]:
        Y_test.append(6)
texts = [s.encode('utf-8') for s in texts]
# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


labels_train= to_categorical(np.asarray(Y_train))
labels_test= to_categorical(np.asarray(Y_test))
train_index=len(label_train)
print ("train_index",train_index)

x_train = data[:train_index]
y_train = labels_train
x_val = data[train_index:]
y_val = labels_test

print ("x_val",len(x_val))
print ("y_val",len(y_val))

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,))


print('Defining model.')
num_filters=64
embedded_sequences = embedding_layer(sequence_input)






x=Bidirectional(LSTM(100))(embedded_sequences)


#x = Conv1D(filters=64,kernel_size=3,padding="valid",activation="relu",strides=1)(embedded_sequences)
#x =MaxPooling1D(pool_size=2)(x)
#x = Dropout(0.2)(x)

#x = Conv1D(filters=num_filters,kernel_size=4,padding="valid",activation="relu",strides=1)(x)
#x =MaxPooling1D(pool_size=2)(x)
#x= Flatten()(x)

#x = Conv1D(filters=num_filters,kernel_size=5,padding="valid",activation="relu",strides=1)(x)
#x =MaxPooling1D(pool_size=2)(x)
#x= Flatten()(x)
x = Dropout(0.5)(x)

x = Dense(100, activation='relu')(x)
preds = Dense(7, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
model.fit(x_train, y_train,batch_size=16,epochs=10,verbose=1,validation_data=(x_val, y_val))
score = model.evaluate(x_val, y_val,batch_size=16,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print ("new_-bilstm")