## tests the classifier model we built previously

from classifier_data import getPOS, get_program, get_system, lemmatize_string
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import pickle
import pandas as pd
import numpy as np

model = load_model('classifier_v1.h5')
tokenizer = Tokenizer()
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

s = 'I can\'t remember my sharepoint password, can you help me get it back?'

s = s.lower()

system = get_system(s)
program = get_program(s)
print('System: ', system)
print('Program: ', program)
s = lemmatize_string(s)
s = [s]
print('Lemmatized: ', s)

data_series = pd.Series(s)
data_tokenized = tokenizer.texts_to_matrix(s, 'tfidf')

for data_t in data_tokenized:   
    prediction = model.predict(np.array([data_t]))
    print(prediction[0])
    print(prediction)
