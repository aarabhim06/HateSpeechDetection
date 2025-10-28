#Hate Speech Detection code

import pandas as pd
import numpy as np

dataset = pd.read_csv("dataset.csv")
dataset["labels"] = dataset["class"].map({0:"Hate speech", 1:"Offensive language", 2:"No hate or offensive lang"})
print("--------------------MAIN DATASET--------------------")
print(dataset)

data = dataset[["tweet","labels"]]
print("--------------------DATASET TO CLEAN TWEET--------------------")
print(data)

import re
import nltk #Natural Language Tool Kit
import string

#Importing stop words and stemming the words
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))

#Import stemming
stemmer = nltk.SnowballStemmer("english")

#data cleaning
def clean_data(text):
    text = str(text).lower()
    text = re.sub('https?://\S+|www\.S+', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' %re.escape(string.punctuation), '', text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*', '', text)
    #Stopwords removal
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    #Stemming text
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

data.loc[:,"tweet"] = (data.loc[:,"tweet"]).apply(clean_data)
print("--------------------AFTER CLEANING TWEET--------------------")
print(data)

X = np.array(data["tweet"])
Y = np.array(data["labels"])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

cv = CountVectorizer()
X = cv.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)

#Building ML model

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)

Y_pred = dt.predict(X_test)

#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm,annot = True, fmt=".1f", cmap ="YlGnBu")
plt.show()

from sklearn.metrics import accuracy_score
print("Accuracy is :",accuracy_score(Y_test, Y_pred))
print("----------------------------------------END----------------------------------------")
