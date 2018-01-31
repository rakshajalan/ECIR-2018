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
from keras.layers import Layer, Input, merge, Dense, LSTM, Bidirectional, GRU, SimpleRNN
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model
from keras import regularizers, constraints
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

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


MAX_SENT_LENGTH = 100
MAX_SENTS = 20
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 300


    
from nltk.corpus import stopwords

import string    
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()



class AttentionWithContext(Layer):
    """
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = glorot_uniform()

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = K.dot(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]
    
    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (input_shape[0], input_shape[-1])



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
with open('/home/jalan.raksha/2017_research/transfer/bilstm_tokenizer_HAN.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

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
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(128))(l_lstm)
l_att = AttentionWithContext()(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(128))(l_lstm_sent)
l_att_sent = AttentionWithContext()(l_dense_sent)

preds = Dense(7, activation='softmax')(l_att_sent)
model1= Model(review_input, preds)





 #load json and create model
json_file = open('/home/jalan.raksha/2017_research/transfer/weak_label_HAN_conf.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

model1_new = Model(review_input, preds)

weights1=[]
for layer in loaded_model.layers:
   weights1.append(layer.get_weights())
print "weights len", len(weights1)


for k in range(len(weights1)):
  model1_new.layers[k].set_weights(weights1[k])


train_len=len(label_train)


#model1_new.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
#model1_new.fit(x_train, y_train,batch_size=32,epochs=20,verbose=1,validation_data=([x_val], y_val))
#score = final_model.evaluate([x_val],y_val,batch_size=32,verbose=0)
#predictions_deep=model1_new.predict(x_val,batch_size=32,verbose=1)
#print('Test loss:', score[0]
#print('Test accuracy:', score[1])
#-----------------------------------------------------





import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',max_features=30000,ngram_range=(1,1), min_df = 0, stop_words = 'english')
tf=pickle.load("/home/jalan.raksha/2017_research/transfer/feature_conf.pkl", "rb")

tfidf_matrix =  tf.transform(texts) #numpy array storing tf-idf repr of each sentence




from sklearn.feature_extraction.text import TfidfVectorizer
#tf = TfidfVectorizer(analyzer='word',max_features=30000,ngram_range=(1,1), min_df = 0, stop_words = 'english')
tfidf_matrix =  tf.transform(texts).toarray() #numpy array storing tf-idf repr of each sentence

input_list1 = array(tfidf_matrix[:train_len])
# for j in range(len(input_list1)):
#    np.append(input_list1[j],feature_names[:train_len])


test_list1 = array(tfidf_matrix[train_len:])
# for j in range(len(test_list1)):
#    np.append(test_list1[j],feature_names[train_len:])


input_list1= preprocessing.normalize(input_list1, norm='l1') #normalize data
 
ratio = 'auto' #class imbalance technique smote
# cc =  SMOTE(ratio=ratio, kind='regular')
# osx,osy = cc.fit_sample(input_list1,output_list1.ravel())
test_data = preprocessing.normalize(test_list1, norm='l1')

 

tfidf_matrix=[]

BATCH_SIZE=32
pred=[]


      
y_train= to_categorical(np.asarray(Y_train))      
y_val= to_categorical(np.asarray(Y_test))






model2_new = Sequential()
model2_new.add(Dense(128,batch_input_shape=(None,30000)))
model2_new.add(Dropout(0.02))
#model2_new.add(model.add(Dense(7, activation='softmax')))


 #load json and create model
json_file = open('/home/jalan.raksha/2017_research/transfer/weakly_label_tfidf_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model2 = model_from_json(loaded_model_json)

weights=[]
for layer in loaded_model2.layers:
   weights.append(layer.get_weights())
print "weights len", len(weights)


for k in range(0,len(weights)):
  model2_new.layers[k].set_weights(weights[k])





#model2.add(Dense(7, activation='softmax'))
print("model fitting - Hierachical LSTM")
merged = Merge([model1_new,model2_new], mode='concat')
final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(64))
#final_model.add(Dropout(0.1))
#final_model.add(Dense(64))
final_model.add(Dense(7, activation='softmax'))





final_model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
final_model.fit([x_train,input_list1], y_train,batch_size=32,epochs=20,verbose=1,validation_data=([x_val,test_data], y_val))
score = final_model.evaluate([x_val,test_data],y_val,batch_size=32,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print ("new glove____joint model_edited with dropout_new joint_tanh,64")






print ("new glove____joint model_edited with dropout_new joint_tanh,64")




