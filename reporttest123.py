# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 18:37:25 2019

@author: Dan Vu
"""
import os
import pandas as pd #import export dataframe
import re #regex to remove
import numpy as np
import matplotlib.pyplot as plt #plot

import seaborn as sns
import unicodedata
import pickle
import sys

import nltk #natural language
from nltk.corpus import stopwords #corpus = corpora
from nltk.stem.porter import PorterStemmer #stemmizing
from nltk.stem.snowball import FrenchStemmer #import the French stemming library
from nltk import FreqDist
from nltk import tokenize
from collections import Counter


import timeit


 
path = "C:/Users/Dan Vu/Desktop/Soutenance/"
os.chdir(path)
        
dataset = pd.read_excel('dataall.xlsx',encoding='latin-1')
dataset.shape
dataset.head()
dataset.info()
dataset.describe()


#PRE PROCESSING
start = timeit.default_timer()

#DEF function PRE PROCESSING

def get_tokens(data):
    '''get the nltk tokens from a text per sentences'''
    tokens = nltk.word_tokenize(data) 
    return tokens
    #tokens = [nltk.word_tokenize(t) for t in nltk.sent_tokenize(data)]

def get_sentences(data):
    '''get the nltk tokens from a text per sentences'''
    sentences = nltk.sent_tokenize(data)
    return sentences

#remove punctuation from data
from string import punctuation
def strip_punctuation(data):
    return ' '.join(c for c in data if c not in punctuation)

#remove number from data
from string import digits
def strip_digit(data):
    return ''.join(i for i in data if i not in digits)


def get_stopswordsfr(type="veronis"):
    '''returns the veronis stopwords in unicode, or if any other value is passed, it returns the default nltk french stopwords'''
    if type=="veronis":
        #VERONIS STOPWORDS
        raw_stopword_listfr = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "and", "an", "am", "afin", "ah", "ai", "aie", "aient", "aies", "ait", "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune", "audit", "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres", "autrui", "aux", "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons", "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "celà", "cent", "cents", "cependant", "certain", "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci", "cinq", "cinquante", "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit", "cinquante-neuf", "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois", "cl", "cm", "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de", "depuis", "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant", "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel", "durant", "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin", "entre", "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f", "fait", "fi", "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem", "heu", "hg", "hl", "hm", "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i", "ici", "il", "ils", "j", "j'", "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'", "jusqu'au", "jusqu'aux", "jusqu'à", "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre", "l'on", "l'un", "l'une", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "lez", "lors", "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma", "maint", "mainte", "maintes", "maints", "mais", "malgré", "me", "mes", "mg", "mgr", "mil", "mille", "milliards", "millions", "ml", "mm", "mm²", "moi", "moins", "mon", "moyennant", "mt", "m²", "m³", "même", "mêmes", "n", "n'avait", "n'y", "ne", "neuf", "ni", "non", "nonante", "nonobstant", "nos", "notre", "nous", "nul", "nulle", "nº", "néanmoins", "o", "octante", "oh", "on", "ont", "onze", "or", "ou", "outre", "où", "p", "par", "par-delà", "parbleu", "parce", "parmi", "pas", "passé", "pendant", "personne", "peu", "plus", "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi", "pourtant", "pourvu", "près", "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles", "qu'il", "qu'ils", "qu'on", "quand", "quant", "quarante", "quarante-cinq", "quarante-deux", "quarante-et-un", "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept", "quarante-six", "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq", "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf", "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf", "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze", "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize", "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que", "quel", "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques", "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'", "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se", "seize", "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six", "soi", "soient", "sois", "soit", "soixante", "soixante-cinq", "soixante-deux", "soixante-dix", "soixante-dix-huit", "soixante-dix-neuf", "soixante-dix-sept", "soixante-douze", "soixante-et-onze", "soixante-et-un", "soixante-et-une", "soixante-huit", "soixante-neuf", "soixante-quatorze", "soixante-quatre", "soixante-quinze", "soixante-seize", "soixante-sept", "soixante-six", "soixante-treize", "soixante-trois", "sommes", "son", "sont", "sous", "soyez", "soyons", "suis", "suite", "sur", "sus", "t", "t'", "ta", "tacatac", "tandis", "te", "tel", "telle", "telles", "tels", "tes", "toi", "ton", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "trente-cinq", "trente-deux", "trente-et-un", "trente-huit", "trente-neuf", "trente-quatre", "trente-sept", "trente-six", "trente-trois", "trois", "très", "tu", "u", "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux", "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois", "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à", "ç'", "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée", "étées", "étés", "êtes", "être", "ô"]
    else:
        #get French stopwords from the nltk kit
        raw_stopword_listfr = stopwords.words('french') #create a list of all French stopwords
    stopword_listfr = [word for word in raw_stopword_listfr] #make to decode the French stopwords as unicode objects rather than ascii
    return stopword_listfr
SWF = get_stopswordsfr(type="veronis")

def filter_stopwords(data,stopword_listfr):
    '''normalizes the words by turning them all lowercase and then filters out the stopwords'''
    words=[w.lower() for w in data] #normalize the words in the text, making them all lowercase    
    #filtering stopwords
    filtered_words = [] #declare an empty list to hold our filtered words
    for word in words: #iterate over all words from the text
        if word not in stopword_listfr and stopwords.words('english') and word.isalpha() and len(word) > 1: #only add words that are not in the French stopwords list, are alphabetic, and are more than 1 character
            filtered_words.append(word) #add word to filter_words list if it meets the above conditions
    #filtered_words.sort() #sort filtered_words list
    return filtered_words

def stem_wordsfr(words):
    '''stems the word list using the French Stemmer'''
    #stemming words
    stemmed_wordsfr = [] #declare an empty list to hold our stemmed words
    stemmerfr = FrenchStemmer() #create a stemmer object in the FrenchStemmer class
    for word in words:
        stemmed_wordfr=stemmerfr.stem(word) #stem the word
        stemmed_wordsfr.append(stemmed_wordfr) #add it to our stemmed word list
    #stemmed_wordsfr.sort() #sort the stemmed_words
    return stemmed_wordsfr

def stem_wordseng(words):
    '''stems the word list using the Porter Stemmer for english'''
    #stemming words
    stemmed_wordseng = [] 
    stemmereng = PorterStemmer() 
    for word in words:
        stemmed_wordeng=stemmereng.stem(word) 
        stemmed_wordseng.append(stemmed_wordeng) 
    #stemmed_wordseng.sort() 
    return stemmed_wordseng                



corpus =[]
for i in range(0, len(dataset)):
    review = get_tokens(str(dataset['Com'][i]))
    review = strip_punctuation(review)
    review = strip_digit(review)
    # Replace 2+ dots with space
    review = re.sub(r'\.{2,}', ' ', review)
    # Strip space, " and ' from data
    review = review.strip(' "\'')
    # Replace multiple spaces with a single space
    review = re.sub(r'\s+', ' ', review)
    review = review.strip('\'"?!,.():;')
    review = review.split()
    review = filter_stopwords(review,SWF)
    #review = stem_wordsfr(review)
    #review = stem_wordseng(review)
    review = ' '.join(review)
    corpus.append(review)


stop = timeit.default_timer()
print('Time: ', stop - start) 


'''
from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(corpus).toarray()
from sklearn_pandas import DataFrameMapper

mapper = DataFrameMapper([(corpus.columns, StandardScaler())])
scaled_features = corpus.fit_transform(corpus.copy(), 4)
scaled_features_df = pd.DataFrame(scaled_features, index=corpus.index, columns=corpus.columns)
'''


'''Features'''
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize

'''Classifiers'''
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

'''Metrics/Evaluation'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from scipy import interp
from itertools import cycle

from sklearn.feature_extraction.text import CountVectorizer
cv = TfidfVectorizer(ngram_range=(1, 3), min_df = 2, max_df = .95, max_features = 1000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#modelll
model_dict = {'Dummy' : DummyClassifier(random_state=3),
              'Stochastic Gradient Descent' : SGDClassifier(random_state=3, loss='log'),
              'Logistic Regression' : LogisticRegression(random_state=3),
              'Random Forest': RandomForestClassifier(random_state=3),
              'Decsision Tree': DecisionTreeClassifier(random_state=3),
              'AdaBoost': AdaBoostClassifier(random_state=3),
              'Gaussian Naive Bayes': GaussianNB(),
              'K Nearest Neighbor': KNeighborsClassifier()}

#Train test split with stratified sampling for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = .3, 
                                                    shuffle = True, 
                                                    stratify = y, 
                                                    random_state = 3)

#Function to get the scores for each model in a df
def model_score_df(model_dict):   
    model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], []
    for k,v in model_dict.items():   
        model_name.append(k)
        v.fit(X_train, y_train)
        y_pred = v.predict(X_test)
        ac_score_list.append(accuracy_score(y_test, y_pred))
        p_score_list.append(precision_score(y_test, y_pred, average='macro'))
        r_score_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
        model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
        model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
        model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)
    return model_comparison_df

results= model_score_df(model_dict)


#ADVANCED PROCESSING
    
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Logistic Regression Case
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

for c in [0.01, 0.05, 0.25, 0.5, 1]: #hyperparameter tuning - Grid search - cross validation
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    lr.predict(X_test)
    print ("Accuracy for C=%s: %s"%(c, accuracy_score(y_test, lr.predict(X_test))))



final_model = LogisticRegression(C=0.05)
final_model.fit(X, y)
print ("Final Accuracy: %s" % accuracy_score(y, final_model.predict(X)))

feature_to_coef = { word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])}

for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:10]:
    print (best_positive)
for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:10]:
    print (best_negative)
    
    
    
# ngrams case 
# Creating the Bag of Words model with ngrams

from sklearn.feature_extraction.text import CountVectorizer
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), max_features = 1000)
X = ngram_vectorizer.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s"%(c, accuracy_score(y_test, lr.predict(X_test))))

final_ngram = LogisticRegression(C=0.25)
final_ngram.fit(X, y)
print ("Final Accuracy: %s" % accuracy_score(y, final_ngram.predict(X)))

feature_to_coef = { word: coef for word, coef in zip(cv.get_feature_names(), final_ngram.coef_[0])}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:20]:
    print (best_positive)
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:20]:
    print (best_negative)


#word counts case
wc_vectorizer = CountVectorizer(binary=False, max_features = 1000)
X = wc_vectorizer.fit_transform(corpus).toarray() 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s"%(c, accuracy_score(y_test, lr.predict(X_test))))

final_wc = LogisticRegression(C=0.05)
final_wc.fit(X, y)
print ("Final Accuracy: %s" % accuracy_score(y, final_wc.predict(X)))









#TF-IDF case BESTTTTTTTTTTTT

start = timeit.default_timer()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features = 1500, max_df = 0.7, min_df=20, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s"%(c, accuracy_score(y_test, lr.predict(X_test))))
    
final_tdif = LogisticRegression(C=1)
final_tdif.fit(X, y)
print ("Final Accuracy: %s" % accuracy_score(y, final_tdif.predict(X)))


feature_to_coef = { word: coef for word, coef in zip(tfidf_vectorizer.get_feature_names(), final_tdif.coef_[0])}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:10]:
    print (best_positive)
for best_negative in sorted(
feature_to_coef.items(), 
    key=lambda x: x[1])[:10]:
    print (best_negative)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, final_tdif.predict(X_test)))
print(classification_report(y_test, final_tdif.predict(X_test)))
print(accuracy_score(y_test, final_tdif.predict(X_test))) 


with open('sentiment_classifier', 'wb') as picklefile:
    pickle.dump(final_tdif,picklefile)

stop = timeit.default_timer()
print('Time: ', stop - start) 




with open('sentiment_classifier', 'rb') as training_model:
    model = pickle.load(training_model)



y_pred2 = model.predict(X_test)

print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2)) 




start = timeit.default_timer()
      
test = pd.read_excel('bonapp.xlsx',encoding='latin-1')

corpus1 =[]
for i in range(0, len(test)):
    review = get_tokens(str(test['Sentences'][i]))
    review = strip_punctuation(review)
    review = strip_digit(review)
    # Replace 2+ dots with space
    review = re.sub(r'\.{2,}', ' ', review)
    # Strip space, " and ' from data
    review = review.strip(' "\'')
    # Replace multiple spaces with a single space
    review = re.sub(r'\s+', ' ', review)
    review = review.strip('\'"?!,.():;')
    review = review.split()
    review = filter_stopwords(review,SWF)
    #review = stem_wordsfr(review)
    #review = stem_wordseng(review)
    review = ' '.join(review)
    corpus1.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features = 1000)   
Xnew = tfidf_vectorizer.fit_transform(corpus).toarray()
ynew = dataset.iloc[:, 1].values
labelencoder_y = LabelEncoder()
ynew = labelencoder_y.fit_transform(y)

y_pred = model.predict(Xnew)

stop = timeit.default_timer()
print('Time: ', stop - start) 

print(confusion_matrix(ynew, y_pred))
print(classification_report(ynew, y_pred))
print(accuracy_score(ynew, y_pred)) 



start = timeit.default_timer()
def lexic(text):
    lex_score = []
    for review in text: # assuming sentences == text
        score = sum([feature_to_coef[word] for word in review.split() if word in feature_to_coef])
        lex_score.append(score)
    return lex_score
final = pd.DataFrame({'Sentences': test['Sentences'],
                        'Score': lexic(corpus1)})
final['Neg/pos'] = pd.np.where(final.Score > 0.1, 'P','N')

print(confusion_matrix(final['Neg/pos'], test['Neg/Pos']))
print(classification_report(final['Neg/pos'], test['Neg/Pos']))
print(accuracy_score(final['Neg/pos'], test['Neg/Pos'])) 
stop = timeit.default_timer()
print('Time: ', stop - start) 





def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()







#Algorithms case - Support Vector Machines
from sklearn.svm import LinearSVC
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3),max_features = 1000)
X = ngram_vectorizer.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


for c in [0.01, 0.05, 0.25, 0.5, 1]:
    svm = LinearSVC(C=c)
    svm.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_test, svm.predict(X_test))))

final_svm_ngram = LinearSVC(C=0.05)
final_svm_ngram.fit(X, y)
print ("Final Accuracy: %s" % accuracy_score(y, final_svm_ngram.predict(X)))

feature_to_coef = { word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:10]:
    print (best_positive)
for best_negative in sorted(
feature_to_coef.items(), 
    key=lambda x: x[1])[:10]:
    print (best_negative)


#Export dictionnary July.2013-Juin.2019
df = pd.DataFrame(data=feature_to_coef, index=[0])
df = (df.T) #transpose
print (df)
df.to_excel('features.xlsx')



#to have a weight of phrase
coef = pd.DataFrame(data=feature_to_coef, index=[0])
coeft=np.array(coef)
Xnew = X*coeft
np.inner(X,coef).shape   

#Export dictionnary 
features = tfidf_vectorizer.get_feature_names()
feedback = pd.DataFrame(data=dataset['feedback'])
df = pd.DataFrame(data=Xnew, columns = features)
df.insert(loc = 0, column = 'feedback', value = feedback )
#calcul total weight
df['sum'] = df.sum(axis=1)

select_list = ['feedback','sum']
data = df[select_list]
data.to_excel('sent2b.xlsx')

Predictdata = lr.predict(X)
Predictpct = pd.DataFrame(lr.predict_proba(X))
Predictpct.to_excel('pctb.xlsx')








df = (df.T) #transpose
print (df)
df.to_excel('foodvocab.xlsx')


print ("Accuracy for C=%s: %s"%(c, accuracy_score(y_test, lr.predict(X))))

print(lr.predict(X))

text = "personnel professionnel repas bord correct services proposés films musique fille fêtait ans retour février beaucoup attention personnel bord remercions carte anniversaire cadeau gâteau air france excellent"
a = 0
for t in text.split() :
    try:
        a = a + feature_to_coef[t]
        print(t,feature_to_coef[t])    
    except:
        pass
print(a)
    