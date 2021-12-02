import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd              
import numpy as np 
# import tensorflow as tf

data = pd.read_csv("bible.csv")
import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer() 

vocabulary_size = 0
word2location = {} # Bag of words
data=data.sample(frac=1)

# Construct the Bag of words
def prepare_vocabulary(data):
    index = 0
    counter = 0
    for sentance in data['text']:
        counter +=1
        if counter<4804*0.7:
            words = nltk.word_tokenize(sentance) 
            for word in words:
                stemed_word = ps.stem(word) # make all letters lower

                if stemed_word not in word2location:
                    
                    word2location[stemed_word] = index
                    index += 1
                    
    return index # the size of our bag of words

def convert2vec(sentance):
    #sentance = sentance.lower()
    res_vec = np.zeros(vocabulary_size) # initialize the result by fill with zeros
    words = nltk.word_tokenize(sentance) # split the verse into words
    for word in words:
        stemed_word = ps.stem(word) # make the word in lower case
        if stemed_word in word2location: 
            res_vec[word2location[stemed_word]]+=1 # calculate the relative amount of the
            # number of occurrences of the word out of all its occurrences in our entire data set
    return res_vec

books = ['Genesis', '1 Samuel','Psalms'] # our data set

