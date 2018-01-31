from gpu_config import *
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from keras import backend as K
import os

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









import operator
f=open("/home/jalan.raksha/2017_research/pubmed_soa/train_doc_soa_score_dict.txt","r")
import ast
import numpy as np
data=f.readlines()
data_dict={}

data_socl={}
data_preg={}
data_goal={}
data_trmt={}
data_demo={}
data_faml={}
data_dise={}
dict_common={}

for l in data:
  data_dict=ast.literal_eval(l) 
  word=data_dict['word']
  value=data_dict['score']
  if dict_common.has_key(word):
    l_prev=dict_common[word]
  else:
      l_prev=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]

  if value >= 0:
    if data_dict['label']=='SOCL':
        data_socl[word]=value
        l_prev[0]=value
        
    if data_dict['label']=='PREG':
        data_preg[word]=value
        l_prev[1]=value
    if data_dict['label']=='GOAL':
        data_goal[word]=value
        l_prev[2]=value
    if data_dict['label']=='TRMT':
        data_trmt[word]=value   
        l_prev[3]=value 

    if data_dict['label']=='DEMO':
        data_demo[word]=value
        l_prev[4]=value

    if data_dict['label']=='FAML':
        data_faml[word]=value
        l_prev[5]=value
    if data_dict['label']=='DISE':
        data_dise[word]=value
        l_prev[6]=value
  dict_common[word]=l_prev        

complete_word=[]
complete_word_vec=[]
f1=open("/home/jalan.raksha/2017_research/ICHI2016_DataSet/train_text_cui.txt","r")
data=f1.readlines()
for line in data:
  #sem_type_dict={"anab":0,"anst":0,"antb":0,"bdsy":0,"biof":0,"blor":0,"bodm":0,"bpoc":0,"bsoj":0,"clna":0,"clnd":0,"diap":0,"drdd":0,"dsgn":0,"famg":0,"food":0,"hlta":0,"hops":0,"horm":0,"inpo":0,"irda":0,"lbpr":0,"ortf":0,"patf":0,"podg":0,"sosy":0,"virs":0}   
  flag=0
  #sem_2={"trmnt":0,"disease":0,"famg":0} 
  tokens=line.strip().split(";")
  word_list=[]
  #print ("000---------------------------")
  #print  (tokens)
  add_vector=[0,0,0,0,0,0,0]

  for k in tokens[:-1]:
   
    q=k.split(":")   
    t=q[0].split("|")
    cui=t[0]
    word=t[1].lower()
    word_list.append(word)
    import operator
    #print "--",dict_common[word]
    if dict_common.has_key(word):
      word_vector_temp=dict_common[word]
      #print word_vector
      max_index1, max_value1 = max(enumerate(word_vector_temp), key=operator.itemgetter(1))
      
      max_value2= sorted(word_vector_temp)[-2]
      if max_value2==0:
         max_value2=0.001
      th=max_value1/float(max_value2)
      
      if th >= 0.5:
        add_vector[max_index1]= add_vector[max_index1]+1
  complete_word_vec.append(add_vector)


#print complete_word_vec[:20]
complete_word_vec=np.array(complete_word_vec)


test_word_vec=[]
f1=open("/home/jalan.raksha/2017_research/ICHI2016_DataSet/test_text_cui.txt","r")
data=f1.readlines()
for line in data:
  #sem_type_dict={"anab":0,"anst":0,"antb":0,"bdsy":0,"biof":0,"blor":0,"bodm":0,"bpoc":0,"bsoj":0,"clna":0,"clnd":0,"diap":0,"drdd":0,"dsgn":0,"famg":0,"food":0,"hlta":0,"hops":0,"horm":0,"inpo":0,"irda":0,"lbpr":0,"ortf":0,"patf":0,"podg":0,"sosy":0,"virs":0}   
  flag=0
  #sem_2={"trmnt":0,"disease":0,"famg":0} 
  tokens=line.strip().split("],")
  word_list=[]
  #print ("000---------------------------")
  #print  (tokens)
  add_vector=[0,0,0,0,0,0,0]
    
  for k in tokens[:-1]:
    
    q=k.split(":")   
    t=q[0].split("|")
    cui=t[0]
    word=t[1].lower()
    word_list.append(word)
    
    #print word
    if dict_common.has_key(word):
      word_vector_=dict_common[word]
      #print word_vector_
      max_index1, max_value1 = max(enumerate(word_vector_), key=operator.itemgetter(1))
      max_value2= sorted(word_vector_)[-2]
      if max_value2==0:
           max_value2=0.001
      th=max_value1/float(max_value2)
      if th >= 0.5:
        add_vector[max_index1]= add_vector[max_index1]+1
  #complete_word.append(word_list)
  test_word_vec.append(add_vector)
print "test_word_vec"
print test_word_vec[:50]
test_word_vec=np.array(test_word_vec)



MAX_SENT_LENGTH = 100
MAX_SENTS = 20
MAX_NB_WORDS = 46000
EMBEDDING_DIM = 300


    
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

       sent_tokenize_list = sent_tokenize(string1)
       #for jj in range(len(sent_tokenize_list)):
        # sent_tokenize_list[jj] = re.sub(r'[^\w\s]','', sent_tokenize_list[jj])
       text_all.append(sent_tokenize_list)  
       texts.append(string1) #para 
      
      c=c+1 

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
       
       sent_tokenize_list = sent_tokenize(string1)
      
       
       
       texts.append(string1) #para
       text_all.append(sent_tokenize_list)
          #stored both title and question together


texts = [s.encode('utf-8') for s in texts]
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

data= np.zeros((len(text_all), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')


word_index = tokenizer.word_index
for i, sentences in enumerate(text_all):
    for j, sent in enumerate(sentences):
        if j< MAX_SENTS:
            wordTokens = text_to_word_sequence(sent.encode('utf-8'))
            k=0
            for _, word in enumerate(wordTokens):
                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                    data[i,j,k] = tokenizer.word_index[word]
                    k=k+1                   
                    
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


word_index = tokenizer.word_index
print'Total %s unique tokens.' % len(word_index)

labels_train= to_categorical(np.asarray(Y_train))
labels_test= to_categorical(np.asarray(Y_test))
train_index=len(label_train)
print "train_index",train_index
x_train = data[:train_index]
y_train = labels_train
x_val = data[train_index:]
y_val = labels_test
text_x=texts[train_index:]
print "x_val",len(x_val)


embeddings_index = {}
f = open('/home/jalan.raksha/2017_research/glove.840B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(128))(embedded_sequences)
sentEncoder = Model(sentence_input, l_lstm)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(128))(review_encoder)
preds = Dense(7, activation='softmax')(l_lstm_sent)
model1 = Model(review_input, preds)

model1.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
model1.fit(x_train, y_train,batch_size=32,epochs=20,verbose=1,validation_data=(x_val, y_val))
predictions_1=model1.predict(x_val,batch_size=32,verbose=1)

#-----------

train_len=len(label_train)


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
tfidf_matrix =  tf.fit_transform(texts[:8000]).toarray() #numpy array storing tf-idf repr of each sentence
tfidf =  tf.transform(texts[8000:])


input_list1 = array(tfidf_matrix)


test_list1 = array(tfidf)


input_list1= preprocessing.normalize(input_list1, norm='l1') #normalize data
output_list1 = array(Y_train) 
ratio = 'auto' #class imbalance technique smote
test_data = preprocessing.normalize(test_list1, norm='l1')
output_list2 = array(Y_test)
 
model2 = Sequential()
model2.add(Dense(128,batch_input_shape=(None,15393)))
model2.add(Dropout(0.02))
model2.add(Dense(7, activation='softmax'))


model2.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
model2.fit(input_list1, y_train,batch_size=32,epochs=10,verbose=1,validation_data=(test_data, y_val))
predictions_2=model2.predict(test_data,batch_size=32,verbose=1)

#--------
predictions_3=test_word_vec
final_class_pred=[]

for g in range(0,3000):

          final_predict=np.zeros(shape=(1,7))
          pred_1 = predictions_1[g]
          pred_2= predictions_2[g]
          pred_3 = predictions_3[g]
          sum_3=0
          for ki in pred_3:
            sum_3=sum_3+ki
          print "pred_3",pred_3
          print "sum_3",sum_3
          for ji in range(0,7): 
            pred_3[ji]=pred_3[ji]/float(sum_3)
          final_predict=[(0.6*x +1.5*y+0.4*z) for x,y,z in zip(pred_1,pred_2,pred_3)]
          
          class_assigned= np.argmax(final_predict)
          final_class_pred.append(class_assigned)   
          #final_predict[ind]=final_predict[ind]+1
print "final _class _pred=top5",final_class_pred[:5]

from sklearn.metrics import accuracy_score
y_true = Y_test
l=accuracy_score(y_true,final_class_pred)
print "accuracy:",l
print ("joint_with soa_wt_vote 0.4,2,0.4")

