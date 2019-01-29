from docutils.nodes import inline
from tabulate import tabulate

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from  sklearn.model_selection import cross_val_score
from  sklearn.model_selection import StratifiedShuffleSplit
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
TRAIN_SET_PATH = "class-train-no-stop.txt"

GLOVE_6B_100D_PATH = "glove.6B.100d.txt"

encoding="utf-8"

X, y = [], []
with open(TRAIN_SET_PATH, "r") as infile:
    for line in infile:
        label, text = line.split("\t")
        # texts are already tokenized, just split on space
        # in a real case we would use e.g. spaCy for tokenization
        # and maybe remove stopwords etc.
        X.append(text.split())
        y.append(label)
X, y = np.array(X), np.array(y)
print ("total examples %s" % len(y))

import numpy as np
with open(GLOVE_6B_100D_PATH, "rb") as lines:
    wvec = {line.split()[0].decode(encoding): np.array(line.split()[1:],dtype=np.float32)
               for line in lines}


# reading glove files, this may take a while
# we're reading line by line and only saving vectors
# that correspond to words from our training set
# if you wan't to play around with the vectors and have
# enough RAM - remove the 'if' line and load everything
import struct
glove_small = {}
all_words = set(w for words in X for w in words)
with open(GLOVE_6B_100D_PATH, "rb") as infile:
      for line in infile:
            parts = line.split()
            word = parts[0].decode(encoding)
            if (word in all_words):
                  nums = np.array(parts[1:], dtype=np.float32)
                  glove_small[word] = nums


# train word2vec on all the texts - both training and test set
# we're not using test labels, just texts so this is fine
model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
print(len(all_words))

# start with the classics - naive bayes of the multinomial and bernoulli varieties
# with either pure counts or tfidf features
'''
mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
'''
# SVM - which is supposed to be more or less state of the art
# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])


class MeanEmbeddingVectorizer(object):
      def __init__(self, word2vec):
            self.word2vec = word2vec
            if len(word2vec) > 0:
                  self.dim = len(word2vec[next(iter(glove_small))])
            else:
                  self.dim = 0

      def fit(self, X, y):
            return self

      def transform(self, X):
            return np.array([
                  np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                          or [np.zeros(self.dim)], axis=0)
                  for words in X
            ])


# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
      def __init__(self, word2vec):
            self.word2vec = word2vec
            self.word2weight = None
            if len(word2vec) > 0:
                  self.dim = len(word2vec[next(iter(glove_small))])
            else:
                  self.dim = 0

      def fit(self, X, y):
            tfidf = TfidfVectorizer(analyzer=lambda x: x)
            tfidf.fit(X)
            # if a word was never seen - it must be at least as infrequent
            # as any of the known words - so the default idf is the max of
            # known idf's
            max_idf = max(tfidf.idf_)
            self.word2weight = defaultdict(
                  lambda: max_idf,
                  [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

            return self

      def transform(self, X):
            return np.array([
                  np.mean([self.word2vec[w] * self.word2weight[w]
                           for w in words if w in self.word2vec] or
                          [np.zeros(self.dim)], axis=0)
                  for words in X
            ])


# Extra Trees classifier is almost universally great, let's stack it with our embeddings# Extra
etree_glove_small = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)),
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_small_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)),
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])


etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])

all_models = [
    # ("mult_nb", mult_nb),
    # ("mult_nb_tfidf", mult_nb_tfidf),
    # ("bern_nb", bern_nb),
    # ("bern_nb_tfidf", bern_nb_tfidf),
    ("svc", svc),
    ("svc_tfidf", svc_tfidf),
    # ("w2v", etree_w2v),
    # ("w2v_tfidf", etree_w2v_tfidf),
    # ("glove_small", etree_glove_small),
    # ("glove_small_tfidf", etree_glove_small_tfidf),


]


unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])

#creating the modell with the training data
svc_tfidf.fit(X, y)
print(svc_tfidf.predict([["how", "is", "the",  "weather", "in",  "israel"], ["how", "is", "the", "rain", "situation"]]))
print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))

# labels_index = {}  # dictionary mapping label name to numeric id
# sample_texts = ['hi', 'how are you', 'how cold is it', 'how is weather this week', 'what time is sunrise']
# unseen = Tokenizer.texts_to_sequences(sample_)
# unseen_sequence = pad_sequences(unseen)
# probs = model.predict(unseen_sequence)texts
# print( labels_index)
# print(probs.argmax(axis=-1))