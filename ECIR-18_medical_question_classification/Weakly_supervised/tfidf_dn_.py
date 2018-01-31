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


"""
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
complete_cat_vec=[]
f1=open("/home/jalan.raksha/2017_research/ICHI2016_DataSet/train_text_cui.txt","r")
data=f1.readlines()
for line in data:

  #sem_type_dict={"anab":0,"anst":0,"antb":0,"bdsy":0,"biof":0,"blor":0,"bodm":0,"bpoc":0,"bsoj":0,"clna":0,"clnd":0,"diap":0,"drdd":0,"dsgn":0,"famg":0,"food":0,"hlta":0,"hops":0,"horm":0,"inpo":0,"irda":0,"lbpr":0,"ortf":0,"patf":0,"podg":0,"sosy":0,"virs":0}   
  flag=0
  
  tokens=line.strip().split(";")
  word_list=[]
  #print ("000---------------------------")
  #print  (tokens)
  add_vector=[0,0,0,0,0,0,0]
  trmnt_vector=[0,0,0,0,0,0,0]
  disease_vector=[0,0,0,0,0,0,0]
  famg_vector=[0,0,0,0,0,0,0]
  anatomy_vector=[0,0,0,0,0,0,0]
  activity_vector=[0,0,0,0,0,0,0]  
  symptoms_vector=[0,0,0,0,0,0,0]  


  for k in tokens[:-1]:
   
    q=k.split(":")   
    t=q[0].split("|")
    cui=t[0]

    word=t[1].lower()
    
    sem_type=q[1]
    sem_type=re.sub("[\[\]]","",sem_type)
    if dict_common.has_key(word):
      word_vector_temp=dict_common[word]
      #print word_vector_temp
      max_index1, max_value1 = max(enumerate(word_vector_temp), key=operator.itemgetter(1))
      max_value2= sorted(word_vector_temp)[-2]

      if max_value2==0:
         max_value2=0.001
      th=max_value1/float(max_value2)
      if th >= 0.5:
        
        if sem_type in ["anab","acti","anst","antb","bdsy","biof","blor","bodm","bpoc","bsoj","clna","clnd","diap","dysn","drdd","famg","food","hlta","hops","horm","inpo","irda","lbpr","ortf","patf","podg","sosy","virs"]:
            if sem_type in ["antb","clnd","diap","drdd","irda","topp"]:
              trmnt_vector[max_index1]= trmnt_vector[max_index1]+1

            elif sem_type =="famg":
              famg_vector[max_index1]=famg_vector[max_index1]+1

            elif sem_type in ["anab","dsyn","inpo"]:
              disease_vector[max_index1]= disease_vector[max_index1]+1

            elif sem_type in ["sosy"]:
              symptoms_vector[max_index1]= symptoms_vector[max_index1]+1

            elif sem_type in ["acti"]:
              activity_vector[max_index1]= activity_vector[max_index1]+1

            elif sem_type in ["bdsy","biof","blor","bpoc","bdoj"]:
               anatomy_vector[max_index1]= anatomy_vector[max_index1]+1
        add_vector[max_index1]= add_vector[max_index1]+1     #  flag=1

        #append all list 
  cat_vector=list(chain(trmnt_vector,famg_vector,disease_vector,symptoms_vector,activity_vector,anatomy_vector))
  #complete_cat_vec.append(cat_vector)
  complete_word_vec.append(add_vector)
complete_word_vec=np.array(complete_word_vec)

complete_cat_vec=np.array(complete_cat_vec)
test_word_vec=[]
test_cat_vec=[]

f1=open("/home/jalan.raksha/2017_research/ICHI2016_DataSet/test_text_cui.txt","r")
data=f1.readlines()
for line in data:

  #sem_type_dict={"anab":0,"anst":0,"antb":0,"bdsy":0,"biof":0,"blor":0,"bodm":0,"bpoc":0,"bsoj":0,"clna":0,"clnd":0,"diap":0,"drdd":0,"dsgn":0,"famg":0,"food":0,"hlta":0,"hops":0,"horm":0,"inpo":0,"irda":0,"lbpr":0,"ortf":0,"patf":0,"podg":0,"sosy":0,"virs":0}   
  flag=0
  sem_2={"trmnt":0,"disease":0,"famg":0,"anatomy":0,"activity":0,"symptoms":0} 
   
  tokens=line.strip().split("],")
  word_list=[]
  #print ("000---------------------------")
  #print  (tokens)
  add_vector=[0,0,0,0,0,0,0]
  trmnt_vector=[0,0,0,0,0,0,0]
  disease_vector=[0,0,0,0,0,0,0]
  famg_vector=[0,0,0,0,0,0,0]
  anatomy_vector=[0,0,0,0,0,0,0]
  activity_vector=[0,0,0,0,0,0,0]  
  symptoms_vector=[0,0,0,0,0,0,0]  
  for k in tokens[:-1]:
    q=k.split(":")   
    t=q[0].split("|")
    cui=t[0]

    word=t[1].lower()
    sem_type=q[1]
    sem_type=re.sub("[\[\]]","",sem_type)
    if dict_common.has_key(word):
      word_vector_temp=dict_common[word]
      #print word_vector
      max_index1, max_value1 = max(enumerate(word_vector_temp), key=operator.itemgetter(1))
      max_value2= sorted(word_vector_temp)[-2]
      if max_value2==0:
         max_value2=0.001
      th=max_value1/float(max_value2)
      if th >= 0.5:
        
        if sem_type in ["anab","acti","anst","antb","bdsy","biof","blor","bodm","bpoc","bsoj","clna","clnd","diap","drdd","dsyn","famg","food","hlta","hops","horm","inpo","irda","lbpr","ortf","patf","podg","sosy","virs"]:
            if sem_type in ["antb","clnd","diap","drdd","irda","topp"]:
              trmnt_vector[max_index1]= trmnt_vector[max_index1]+1

            elif sem_type =="famg":
              famg_vector[max_index1]=famg_vector[max_index1]+1

            elif sem_type in ["anab","dsyn","inpo"]:
              disease_vector[max_index1]= disease_vector[max_index1]+1

            elif sem_type in ["sosy"]:
              symptoms_vector[max_index1]= symptoms_vector[max_index1]+1

            elif sem_type in ["acti"]:
              activity_vector[max_index1]= activity_vector[max_index1]+1

            elif sem_type in ["bdsy","biof","blor","bpoc","bdoj"]:
               anatomy_vector[max_index1]= anatomy_vector[max_index1]+1
        add_vector[max_index1]= add_vector[max_index1]+1       #flag=1

        #append all list 
  cat_vector=list(chain(trmnt_vector,famg_vector,disease_vector,symptoms_vector,activity_vector,anatomy_vector))
  #cat_vector=[(trmnt_vector+famg_vector+disease_vector+symptoms_vector+activity_vector+anatomy_vector),zip(trmnt_vector,famg_vector,disease_vector,symptoms_vector,activity_vector,anatomy_vector)]
  #test_cat_vec.append(cat_vector)
  test_word_vec.append(add_vector)
test_word_vec=np.array(test_word_vec)

"""


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

f_data=open("/home/jalan.raksha/2017_research/self_train_check/self_data1.txt","r")
data1=f_data.readlines()
for line1 in data1: 
       string1=clean_str(line1)
       string1=re.sub(r'[^\x00-\x7F]+',' ',string1) #to remove non ascii characters

       string1=clean_str(string1.decode('utf-8'))
       self_texts.append(string1) #para
      

self_labels=y_train_copy
f_val=open("/home/jalan.raksha/2017_research/self_train_check/self_val1.txt","r")
data1=f_val.readlines()
for line1 in data1: 
       p=line1.split("-->")
       self_labels.append(int(p[0]))
      
MAX_SENT_LENGTH = 100
MAX_SENTS = 20
MAX_NB_WORDS = 46000
EMBEDDING_DIM = 300


self_texts = [s.encode('utf-8') for s in self_texts]

print "labels_assigned",Counter(self_labels)

y_val= to_categorical(np.asarray(Y_test))
y_semi = to_categorical(np.asarray(self_labels))





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
semi_list1 = array(tfidf_matrix[3000:])


input_list1= preprocessing.normalize(input_list1, norm='l1') #normalize data


test_list1 = preprocessing.normalize(test_list1, norm='l1')

semi_list1= preprocessing.normalize(semi_list1, norm='l1') #normalize data


 
model2 = Sequential()
model2.add(Dense(128,batch_input_shape=(None,30000)))
model2.add(Dropout(0.02))
model2.add(Dense(7, activation='softmax'))

from sklearn.metrics import confusion_matrix
model2.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=["accuracy"])

model2.fit(semi_list1, y_semi,batch_size=32,epochs=10,verbose=1,validation_data=(test_list1, y_val))

print "Model2 i s done"



model_json = model2.to_json()
with open("/home/jalan.raksha/2017_research/self_train_check/tfidf_check_model.json", "w") as json_file:
    json_file.write(model_json)

# load json and create model
json_file = open('/home/jalan.raksha/2017_research/self_train_check/tfidf_check_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


model2_new = Sequential()
model2_new.add(Dense(128,batch_input_shape=(None,30000)))
model2_new.add(Dropout(0.02))
model2_new.add(Dense(7, activation='softmax'))


weights1=[]
for layer in loaded_model.layers:
   weights1.append(layer.get_weights())
print "weights len", len(weights1)


for k in range(len(weights1)):
  model2_new.layers[k].set_weights(weights1[k])


model2_new.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
model2_new.fit(input_list1,y_train,batch_size=32,epochs=20,verbose=1,validation_data=(test_list1, y_val))
predictions_1=model2_new.predict(x_val,batch_size=32,verbose=1)
print "with  10 epochs tfidf"
#weakly label tfidf


# #--------
# predictions_3=test_word_vec
# final_class_pred=[]

# for g in range(0,3000):

#           final_predict=np.zeros(shape=(1,7))
#           pred_1 = predictions_1[g]
#           pred_2= predictions_2[g]
#           pred_3 = predictions_3[g]
#           sum_3=0
#           for ki in pred_3:
#             sum_3=sum_3+pred_3[ki]
#           D
#           for ji in range(0,7): 
#             pred_3[ji]=pred_3[ji]/float(sum_3)
#           final_predict=[(0.6*pred_1+pred_2+0.5*pred_3), zip(pred_1,pred_2,pred_3)]
          
#           class_assigned= np.argmax(final_predict)
#           final_class_pred.append(class_assigned)   
#           #final_predict[ind]=final_predict[ind]+1
# print "final _class _pred=top5",final_class_pred[:5]

# from sklearn.metrics import accuracy_score
# y_true = y_val
# l=accuracy_score(y_true,final_class_pred)
# print "accuracy:",l
# print ("joint_with soa_wt_vote_only all 3")



from sklearn.metrics import confusion_matrix





"""
model3 = Sequential()
model3.add(Dense(7,batch_input_shape=(None,7)))

merged = Merge([model1,model2,model3], mode='concat')
final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(7, activation='softmax'))

from sklearn.metrics import confusion_matrix
#final_model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=["accuracy"])

#final_model.fit([x_train,input_list1,complete_word_vec], y_train,batch_size=32,epochs=12,verbose=1,validation_data=([x_val,test_data,test_word_vec], y_val))

print "weakly lebelled merge using add_vector
"""
