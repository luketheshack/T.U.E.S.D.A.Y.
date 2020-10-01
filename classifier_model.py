from classifier_data import *
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer

# random seed for reproducability
np.random.seed(42069)

# split test and train data inputs and outputs
train_size = int(len(data) * .9)
print(train_size)
train_post = data['utterance'][:train_size]
train_cat = data['intent'][:train_size]
test_post = data['utterance'][train_size:]
test_cat = data['intent'][train_size:]

# parameters
num_labels = 20
batch_size = 64
vocab_size = 10000

# tokenize vocabulary to assign each word a number
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(train_post)
x_train = tokenizer.texts_to_matrix(train_post, mode = 'tfidf')
x_test = tokenizer.texts_to_matrix(test_post, mode = 'tfidf')

# one hot encoding for the outputs
encoder = LabelBinarizer()
encoder.fit(train_post)
y_train = encoder.transform(train_post)
y_test = encoder.transform(test_post)

# build the actual model
model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0,3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(17572))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', 
                optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, 
                    epochs = 35, verbose = 1, validation_split=0.1)

score = model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 1)
print('Test Accuracy: ', score[1])

text_labels = encoder.classes_

# gives live demo test of model
for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction[0])]
    print('Actual label: ', test_cat.iloc[i])
    print('Predicted label: ', predicted_label)

# saves model
model.model.save('classifier_v1.h5')

# save tokenizer/vocab
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# current accuracy: 96.41%
