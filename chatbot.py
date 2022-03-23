import pandas as pd
import numpy as np
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import re
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import load_model
import tensorflow




def trainIntentModel():
    # Load the dataset and prepare it to the train the model

    # Importing dataset and splitting into words and labels
    dataset = pd.read_csv('datasets/intent.csv', names=["Query", "Intent"])

    X = dataset["Query"]
    y = dataset["Intent"]

    unique_intent_list = list(set(y))

    print("Intent Dataset successfully loaded!")
    
    # Clean and prepare the intents corpus
    queryCorpus = []
    ps = PorterStemmer()

    for query in X:
        query = re.sub('[^a-zA-Z]', ' ', query)

        # Tokenize sentence
        query = query.split(' ')

        # Lemmatizing
        tokenized_query = [ps.stem(word.lower()) for word in query]

        # Recreate the sentence from tokens
        tokenized_query = ' '.join(tokenized_query)

        # Add to corpus
        queryCorpus.append(tokenized_query)
        
    print(queryCorpus)
    print("Corpus created")
    
    countVectorizer= CountVectorizer(max_features=800)
    corpus = countVectorizer.fit_transform(queryCorpus).toarray()
    print(corpus.shape)
    print("Bag of words created!")
    
    # Save the CountVectorizer
    pk.dump(countVectorizer, open("saved_state/IntentCountVectorizer.sav", 'wb'))
    print("Intent CountVectorizer saved!")
    
    # Encode the intent classes
    labelencoder_intent = LabelEncoder()
    y = labelencoder_intent.fit_transform(y)
    y = np_utils.to_categorical(y)
    print("Encoded the intent classes!")
    print(y)
    
    # Return a dictionary, mapping labels to their integer values
    res = {}
    for cl in labelencoder_intent.classes_:
        res.update({cl:labelencoder_intent.transform([cl])[0]})

    intent_label_map = res
    print(intent_label_map)
    print("Intent Label mapping obtained!")
    
    # Initialising the Aritifcial Neural Network
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 96, kernel_initializer = 'uniform', activation = 'relu', input_dim = 133))

    # Adding the second hidden layer
 