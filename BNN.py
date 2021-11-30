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
word2location = {}
data=data.sample(frac=1)
# Bag of words

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
                    
    return index


