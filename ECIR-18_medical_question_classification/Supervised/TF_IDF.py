from gpu_config import *
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import model_from_json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from keras import backend as K
import os
from collections import Counter
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import os
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,GRU
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
from keras.layers import Conv1D, MaxPooling1D, Embedding,MaxPooling2D
from keras.models import Model
from keras.layers import Dense, Dropout
import csv
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Merge

import re
from keras import initializers
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from collections import defaultdict
import re
from numpy  import array

import sys
import os

os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from itertools import chain


    
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


from nltk import tokenize
from nltk.tokenize import sent_tokenize

texts=[]
self_texts=[]

with open("/home/jalan.raksha/2017_research/ICHI2016_DataSet/ICHI2016-TestData_label.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")   
    label_test=[]
    question=[]
    for line in tsvreader:
       label_test.append(line[0])  
       string1= str(line[1])+" "+str(line[2])
       string1=clean_str(string1)
       string1=re.sub(r'[^\x00-\x7F]+',' ',string1) #to remove non ascii characters

       string1=clean_str(string1.decode('utf-8'))
       
       #sent_tokenize_list = sent_tokenize(string1)   
       #texts.append(string1) #para
       #text_all.append(sent_tokenize_list)
       self_texts.append(string1)  




with open("/home/jalan.raksha/2017_research/ICHI2016_DataSet/ICHI2016-TrainData.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    c=0
    label_train=[]
    text_all=[]
    question=[]
    for line in tsvreader:
      if c>0:
       label_train.append(line[0])  
       string1= str(line[1])+" "+str(line[2])
       
       string1=re.sub(r'[^\x00-\x7F]+',' ',string1) #to remove non ascii characters
       string1=clean_str(string1)

       #sent_tokenize_list = sent_tokenize(string1)
       #for jj in range(len(sent_tokenize_list)):
       # sent_tokenize_list[jj] = re.sub(r'[^\w\s]','', sent_tokenize_list[jj])
       #text_all.append(sent_tokenize_list)  
       #texts.append(string1) #para 
       self_texts.append(string1)   
      c=c+1 




uniq_label=["SOCL","PREG","GOAL","TRMT","DEMO","FAML","DISE"]
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

y_train_copy=Y_train
  
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
y_train= to_categorical(np.asarray(Y_train))


      
MAX_SENT_LENGTH = 100
MAX_SENTS = 20
MAX_NB_WORDS = 46000
EMBEDDING_DIM = 300


self_texts = [s.encode('utf-8') for s in self_texts]

print "labels_assigned",Counter(self_labels)

y_val= to_categorical(np.asarray(Y_test))






train_len=len(label_train)

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', max_features=30000,ngram_range=(1,1), min_df = 0, stop_words = 'english')

tfidf_matrix =  tf.fit_transform(self_texts).toarray() #numpy array storing tf-idf repr of each sentence

import pickle
with open('/home/jalan.raksha/2017_research/self_train_check/tf_idf_check.pickle', 'wb') as handle:
    pickle.dump(tf, handle, protocol=pickle.HIGHEST_PROTOCOL)

feature_names = tf.get_feature_names() 
input_list1 = array(tfidf_matrix[3000:11000])
test_list1 = array(tfidf_matrix[:3000])



input_list1= preprocessing.normalize(input_list1, norm='l1') #normalize data
test_list1 = preprocessing.normalize(test_list1, norm='l1')

model2 = Sequential()
model2.add(Dense(128,batch_input_shape=(None,30000)))
model2.add(Dropout(0.02))
model2.add(Dense(7, activation='softmax'))

from sklearn.metrics import confusion_matrix
model2.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
model2.fit(input_list1,y_train,batch_size=32,epochs=20,verbose=1,validation_data=(test_list1, y_val))
predictions_1=model2.predict(x_val,batch_size=32,verbose=1)
print "with  20 epochs tfidf"




