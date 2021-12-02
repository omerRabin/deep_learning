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

def prepare_vocabulary(data):
    index = 0
    for sentance in data['text']:
        #sentance = sentance.lower()
        words = nltk.word_tokenize(sentance)
        for word in words:
            stemed_word = ps.stem(word)
            if stemed_word not in word2location:
                word2location[stemed_word] = index
                index += 1
    return index

def convert2vec(sentance):
    #sentance = sentance.lower()
    res_vec = np.zeros(vocabulary_size)
    words = nltk.word_tokenize(sentance)
    for word in words:
        stemed_word = ps.stem(word)
        if stemed_word in word2location:
            res_vec[word2location[stemed_word]]+=1
    return res_vec

books = ['Genesis', '1 Samuel', 'Psalms']

def encode(line):
    res_vec = np.zeros(3)
    idx = books.index(data.iloc[line]['book'])
    res_vec[idx] = 1
    return res_vec

vocabulary_size = prepare_vocabulary(data)
print("the size of the vocabulary is: ", vocabulary_size)
import random
tf.compat.v1.disable_v2_behavior()

data_x = np.array([convert2vec(data.iloc[i]['text']) for i in range(len(data['text']))])
data_y = np.array([encode(i) for i in range(len(data['text']))])

features = vocabulary_size
categories = 3 #number of books
epsilon = 1e-12
(hidden1_size) = (50)
x = tf.compat.v1.placeholder(tf.float32,[None,features])
y_ = tf.compat.v1.placeholder(tf.float32,[None,categories])

W1 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([features, hidden1_size], stddev = 0.1))
b1 = tf.compat.v1.Variable(tf.constant(0.1, shape = [hidden1_size]))
z1 = tf.nn.relu(tf.matmul(x,W1)+b1)

W2 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([hidden1_size, categories], stddev = 0.1))
b2 = tf.compat.v1.Variable(tf.constant(0.1, shape = [categories]))
z2 = tf.nn.relu(tf.matmul(z1,W2)+b2)

y = tf.nn.softmax(tf.matmul(z1,W2)+b2)

cross_entropy = tf.reduce_mean(-tf.compat.v1.reduce_sum(y_*tf.compat.v1.log(y), reduction_indices = [1]))
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
iteration_axis = np.array([])
accuracy_axis = np.array([])
rand = []

for i in range (500):
    for _ in range(1000):
        for r in range (200):
            ra = random.randrange(0, 4804)
            rand.append(ra)
        batch_xs = np.array([convert2vec(data.iloc[j]['text']) for j in rand])
        batch_ys = np.array([encode(j) for j in rand])
        rand = []
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    iteration_axis = np.append(iteration_axis, [i], axis=0)
    accuracy_axis = np.append(accuracy_axis, [sess.run(accuracy, feed_dict={x: data_x, y_: data_y})], axis=0)
    print("Iteration: ",i, " Accuracy:", sess.run(accuracy, feed_dict={x: data_x, y_: data_y}))
    
