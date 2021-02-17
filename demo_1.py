## Baekkwan Park
## any questions contact: baekkwan.park@gmail.com

## You can find what we discuss in the workshop.
## This is a .py file, but you can also use .ipynb file in this folder.

import os
import pandas as pd # importing pandas for data read and management.

# 1. Reading Data
# reading data with pandas
# Sample_Data.csv is in this folder.
DATA = DATA = pd.read_csv('Sample_Data.csv')

# reading texts and annotations
texts = DATA['texts']
labels = DATA['labels']

# 2. Tokening Text Data
# Building a dictionary to store the number of counts for each word in the data.
from collections import defaultdict # useful dictionary modules from collections.

# use a simple function for word counting
def term_freq(tokens):

    # making a dictionary
    term_counts = defaultdict(int)

    # reading each token and adding count to the dictionary
    for token in tokens:
        term_counts[token.lower()] += 1

    return term_counts

import re # importing regex (regular expression) module

text_counts = []
token_names = []

# reading text data one by one and tokenize words and counting them.
for text in texts:

    # making tokens
    # using re to identify words.
    tokens = re.findall(re.compile(r'\b\w\w+\b'), text)

    # Counting each term frquency
    text_count = term_freq(tokens)
    text_counts.append(text_count)

    # appending each token names
    token_names.extend(text_count.keys())

# identifying token names
token_names_list = list(set(token_names))

# 3. Putting the word counts into a matrix form (document term matrix).

matrix_base = []

# from each word count from previous steps
for text_count in text_counts:

    matrix_rows = []

    # for each unique word and matching the correspoding word counts
    # and adding it to the matrix rows
    for token_name in token_names_list:
        matrix_rows.append(text_count.get(token_name, 0))

    # aggregating all the matrix rows into the matrix base
    matrix_base.append(matrix_rows)

# making a sparse matrix
from scipy.sparse import csr_matrix # import scipy sparse matrix tool

# converting to sparse matrix
matrix = csr_matrix(matrix_base)

# If you want to see what you have just made--that is, document term matrix--in a nice dataframe,
# try the following lines.

# getting the indices for the matrix
text_names = [i for i, v in enumerate(matrix)]

# Displaying data structure with pandas dataframe
dtm_df = pd.DataFrame(data=matrix.toarray(), index= text_names, columns = token_names_list)
print (dtm_df) # this won't look pretty, but it would look better if you run this in jupyter notebook.


# 4. Modeling
# 4.1. have your data aready in the right format.
# We bring what we have just made in the previous steps
# Defining Text Data X

# use numpy
import numpy as np

# matrix_base itslef is a list. converting to numpy array.
X = np.array(matrix_base)

# if you use other machine learning libraries such as scikit-learn, you would not have to do this,
# but, we are going over what's inside those machine learning algorithms. We need to specify many things ourselves.

# for example, we need to specify bias terms.
bias_term = np.zeros((X.shape[0], 1))
bias_term.fill(1) # simply assinging the value of 1.

# Now, our text data matrix is X2 with the bias term.
X2 = np.append(bias_term, X, axis=1)

#4.2. You need to split the data X2 into training and testing set. (we do not do anything extra in this workshop).
# We use train_test_split modules from scikit-learn.
from sklearn.model_selection import train_test_split

# Randomly splitting the data X2 into train and test (30 percent), the annotation labels into train and test (30 percent).
train_X, test_X, train_Y, test_Y = train_test_split(X2, labels, test_size=0.30, random_state = 20)

#4.3 Training
# We do optimization manually to show step by step.
np.random.seed(2) # setting a random seed for weight initialization

theta = np.zeros((train_X.shape[1], 1)) # initial theta (weights)

for i in range(200): # 200 iteration for gradient descent optimization

    m = train_X.shape[0] # total number of data

    z = np.dot(train_X, theta) # dot product between training data and our weights (theta)

    # sigmoid function
    h = 1 / (1 + np.exp(-z))

    # reformatting the annotation for matrix calculation
    y = np.array(train_Y).reshape(-1,1)

    # Loss Function
    # In the workshop, we do not talk about regularization, but for comparison to scikit-learn results.
    # It's included in this loss function.
    J = (-1/m) * ((np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1-h))) + (0.1 * np.sum(theta)))

    # updating weights
    # learning rate is set manually here.
    theta = theta - (0.0001 / m) * np.dot((train_X.T), (h - y))

print ('loss: ', J) # see your loss
print ('thetas: ', theta) # see your updated theta that will be used for prediction.

#4.4 Testing
# Just to avoid confusion, we name test inner product z_t.
z_t = np.dot(test_X, theta)

# calculating the predicted probability using sigmoid function.
pred_probs= 1 / (1 + np.exp(-z_t))

# We set the probability threshold at 0.5 for classification, otherwise set at 0.
pred_labels = np.where(pred_probs > 0.5, 1, 0)
pred_labels = [i[0] for i in pred_labels] # format better for comparing to the true labels.

# Calculating model accuracy--that is, how mwny test items the trained model predicted correctly.

correct_pred = 0 # counter set at 0.

# comparing for each test data point
for pred, truth in zip(pred_labels, test_Y): # remember, test_Y is our true labels for testing data.

    # when they are the same count
    if pred == truth:
        correct_pred += 1

# calculate accuracy
accuracy = correct_pred / len(test_Y)
print ('model accuracy: ', accuracy) # expecting 0.933.

# Creating a confusion matrix with scikit-learn modules.
from sklearn.metrics import confusion_matrix

# confusion matrix
print (confusion_matrix(test_Y.tolist(), pred_labels))
# expected outcome
# array([[26,  0],
#        [ 4, 30]])

# *** I would not recommend calculating the following metrics manually.
# But, what we are doing here is to understand the processes step by step.
# Thus, you need to understand the concept of precision, recall, and F-1 scores.

# Calculating precision
# # of true positives / total # of predicited positive (i.e. true positives + false positives)
print (26/30)# precision for class 0
print (30/30)# precision for class 1

# Calculating recall
# # of true positives/ total # of actual positives
print (26/26) # recall for class 0
print (30/34)# recall for class 1

# Calculating F-1 score
# 2*((precision*recall)/(precision+recall))
print (2*((26/30)*(26/26))/((26/30) + (26/26))) # F1 for class 0
print (2*((30/30)*(30/34))/((30/30) + (30/34)))# F1 for class 1


## Part 2. Easy Way !!
# You can do all of the above steps in a very quickly without typing lines of code.
# Here, we talk about how to use mahcine learning library such as scikit-learn, which is very powerful and conveinent.

# 1. reading data
import pandas as pd # you already did this, but this is a demo. So, do it again.

DATA = pd.read_csv('Sample_Data.csv')
texts = DATA['texts']
labels = DATA['labels']

# 2. building a document term matrix.
from sklearn.feature_extraction.text import CountVectorizer # use CounterVectorizer module. EASY!!!!
Vectorizer = CountVectorizer()
X = Vectorizer.fit_transform(texts) # seriously, this is it!

# 3. Data split
# this is the same as above.
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.30, random_state = 20)

# 4. Train: We use LogisticRegression module from scikit-learn.
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression() # call it
classifier.fit(X_train, y_train)  #  and use it. This is it!.

# 5. prediction
y_pred = classifier.predict(X_test)

# 6. Evauation metrics
# As I said above, scikit learn has all these useful modules.
# This is what I meant: I would not recommend doing these manually.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('confusion matrix: ', '\n', confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print('accuracy: ', accuracy_score(y_test, y_pred))

print ("END!!!")
# END








#
