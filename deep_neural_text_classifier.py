import re
from nltk import stem
import keras
from keras.models import Sequential
from keras.engine import Input
from keras.layers import Embedding, merge, Activation
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing import sequence
from keras.layers import LSTM, Conv1D, Flatten,Dense, Dropout, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization
from keras.optimizers import adam, rmsprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
#from nltk.corpus import stopwords


###########################################################################
# Preprocessing Functions
###########################################################################

stemmer = stem.PorterStemmer()
#stop_words = set(stopwords.words('english'))

def text_preprocess(text_data):
    
    res = []
    
    for k in range(len(text_data)):
        
        cur_string = re.sub(r'[^\w\s]', ' ', text_data[k])
        cur_string = re.sub(' +', ' ', cur_string)
        cur_list = [x.lower() for x in cur_string.split(' ') if not x.isdigit()]
        res.append(' '.join(cur_list))
        
    return np.asanyarray(res)
    
##############################################################################
# Load and Transform Data
##############################################################################

from sklearn.model_selection import train_test_split

df = pd.read_csv('data_fp')
matrix = df[['Narrative', 'Umb_Cat']].as_matrix() 
text = np.asanyarray(matrix[:,0], dtype='str')
labs = np.zeros((text.shape[0],len(cats)), dtype='i4')

for k in range(len(labs)):
    
    labs[k, cats_id[matrix[k,1]]] = 1
    
X_train_r, X_test_r, y_train, y_test = train_test_split(text, labs, test_size=300000)
tokenizer = Tokenizer(num_words=30000)

tokenizer.fit_on_texts(text_preprocess(X_train_r))

X_train_n_s = tokenizer.texts_to_sequences(text_preprocess(X_train_r))
X_train_n = pad_sequences(X_train_n_s, maxlen=30)

X_test_n_s = tokenizer.texts_to_sequences(text_preprocess(X_test_r))
X_test_n = pad_sequences(X_test_n_s, maxlen=30)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


print('Shape of training data tensors:', X_train_n.shape)
print('Shape of training label tensor:', y_train.shape)
print('Shape of test data tensors:', X_test_n.shape)
print('Shape of test label tensor:', y_test.shape)



######################################################################################
# Load the Pretrained Glove Embedding 
######################################################################################
import os

embeddings_index = {}
f = open(os.path.join('/home/james/anaconda3/data/glove_6B/', 'glove.6B.200d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 200))

for word, i in word_index.items():
    
    embedding_vector = embeddings_index.get(word)
    
    if embedding_vector is not None:
        
        embedding_matrix[i] = embedding_vector
        
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            200,
                            weights=[embedding_matrix],
                            input_length=30,
                            trainable=True)
                            
 ###################################################################################
 # Build the Model 
 ###################################################################################
 
 model = Sequential()
model.add(embedding_layer)
# Block 1
model.add(Conv1D(128,
                 3,
                 padding='valid',
                 activation='elu',
                 strides=1))
model.add(BatchNormalization())
# Block 2
model.add(Conv1D(128,
                 3,
                 padding='valid',
                 activation='elu',
                 strides=1))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
# Block 3
model.add(Conv1D(128,
                 3,
                 padding='valid',
                 activation='elu',
                 strides=1))
model.add(BatchNormalization())
# Block 4
model.add(Conv1D(128,
                 3,
                 padding='valid',
                 activation='elu',
                 strides=1))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
# Recurrent Block
model.add(LSTM(256))
model.add(BatchNormalization())
# Dense Block
model.add(Dense(1028))
model.add(Activation('elu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(len(cats)))
model.add(Activation('softmax'))


sgd = adam(lr=5e-5, epsilon=1e-7)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
              
              
model.fit(X_train_n, y_train, validation_data=[X_test_n, y_test],
          batch_size=512, epochs=10)
