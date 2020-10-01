# classification tool for text to find out the issue

import pandas as pd
import nltk
from collections import Counter 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize 

data = pd.read_csv("classification_data.csv")

likely_programs = ['sharepoint', 'sharefile', 'veeva', 'medidata', 'adp', 
                    'zenqms', 'actitime', '365', 'word', 'excel', 'rackspace',
                    'rack space', 'share file', 'share point', 'teams', 'outlook',
                    'file explorer', 'vpn', 'chrome', 'RSA']

likely_systems = ['printer', 'brother', 'epson', 'computer', 'phone', 'network', 'wifi', 
                    'wi-fi', 'wi fi', 'hp', 'email', 'windows']


def getPOS(word):
    likely = wordnet.synsets(word)
    pos = Counter()
    pos["n"] = len(  [ item for item in likely if item.pos()=="n"]  )
    pos["v"] = len(  [ item for item in likely if item.pos()=="v"]  )
    pos["a"] = len(  [ item for item in likely if item.pos()=="a"]  )
    pos["r"] = len(  [ item for item in likely if item.pos()=="r"]  )
    most_likely_part_of_speech = pos.most_common(1)[0][0]
    return most_likely_part_of_speech


def lemmatize_string(string):    
    lemmatizer = WordNetLemmatizer()
    tokenized_string = word_tokenize(string)
    words = [lemmatizer.lemmatize(word, getPOS(word)) for word in tokenized_string]
    words = ' '.join(words)
    return words


def get_program(string):

    global likely_programs
    s = string.split(' ')
    problematic_programs = []

    for word in s:
        if word in likely_programs:
            problematic_programs.append(word)

    return problematic_programs  

def get_system(string):

    global likely_systems
    s = string.split(' ')
    problematic_systems = []
    for word in s:
        if word in likely_systems:
            problematic_systems.append(word)

    return problematic_systems

for line in data.index:
    if data.loc[line, 'category'] != 'ACCOUNT':
        data.loc[line, 'intent'] = 'other'
for line in data.index:
    s = data.iloc[line, 0]
    data.iloc[line, 0] = lemmatize_string(s).lower()
