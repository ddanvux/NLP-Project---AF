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



path = "C:/Users/m429673/Desktop/Travail/Project Python Verbatims/"
os.chdir(path)
        
datasettrain = pd.read_csv('VBT.csv',encoding='latin-1', sep =';')
datasettest = pd.read_csv('IFEdata.csv', encoding='latin-1', sep =';')
#dataset.shape
#dataset.head()
#dataset.info()
#dataset.describe()

#PRE PROCESSING



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
        raw_stopword_listfr = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "afin", "ah", "ai", "aie", "aient", "aies", "ait", "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune", "audit", "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres", "autrui", "aux", "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons", "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "celà", "cent", "cents", "cependant", "certain", "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci", "cinq", "cinquante", "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit", "cinquante-neuf", "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois", "cl", "cm", "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de", "depuis", "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant", "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel", "durant", "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin", "entre", "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f", "fait", "fi", "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem", "heu", "hg", "hl", "hm", "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i", "ici", "il", "ils", "j", "j'", "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'", "jusqu'au", "jusqu'aux", "jusqu'à", "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre", "l'on", "l'un", "l'une", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "lez", "lors", "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma", "maint", "mainte", "maintes", "maints", "mais", "malgré", "me", "mes", "mg", "mgr", "mil", "mille", "milliards", "millions", "ml", "mm", "mm²", "moi", "moins", "mon", "moyennant", "mt", "m²", "m³", "même", "mêmes", "n", "n'avait", "n'y", "ne", "neuf", "ni", "non", "nonante", "nonobstant", "nos", "notre", "nous", "nul", "nulle", "nº", "néanmoins", "o", "octante", "oh", "on", "ont", "onze", "or", "ou", "outre", "où", "p", "par", "par-delà", "parbleu", "parce", "parmi", "pas", "passé", "pendant", "personne", "peu", "plus", "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi", "pourtant", "pourvu", "près", "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles", "qu'il", "qu'ils", "qu'on", "quand", "quant", "quarante", "quarante-cinq", "quarante-deux", "quarante-et-un", "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept", "quarante-six", "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq", "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf", "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf", "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze", "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize", "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que", "quel", "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques", "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'", "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se", "seize", "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six", "soi", "soient", "sois", "soit", "soixante", "soixante-cinq", "soixante-deux", "soixante-dix", "soixante-dix-huit", "soixante-dix-neuf", "soixante-dix-sept", "soixante-douze", "soixante-et-onze", "soixante-et-un", "soixante-et-une", "soixante-huit", "soixante-neuf", "soixante-quatorze", "soixante-quatre", "soixante-quinze", "soixante-seize", "soixante-sept", "soixante-six", "soixante-treize", "soixante-trois", "sommes", "son", "sont", "sous", "soyez", "soyons", "suis", "suite", "sur", "sus", "t", "t'", "ta", "tacatac", "tandis", "te", "tel", "telle", "telles", "tels", "tes", "toi", "ton", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "trente-cinq", "trente-deux", "trente-et-un", "trente-huit", "trente-neuf", "trente-quatre", "trente-sept", "trente-six", "trente-trois", "trois", "très", "tu", "u", "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux", "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois", "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à", "ç'", "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée", "étées", "étés", "êtes", "être", "ô"]
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


corpustrain =[]
for i in range(0, len(datasettrain)):
    review = get_tokens(str(datasettrain['feedback'][i]))
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
    corpustrain.append(review)

corpustest =[]
for i in range(0, len(datasettest)):
    review = get_tokens(str(datasettest['General comments'][i]))
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
    corpustest.append(review)
#ADVANCED PROCESSING
    
# Creating the Bag of Words model 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 6000)
X = cv.fit_transform(corpustrain).toarray()
y = datasettrain.iloc[:, 2].values

from sklearn.preprocessing import LabelEncoder
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



final_model = LogisticRegression(C=1)
final_model.fit(X, y)
print ("Final Accuracy: %s" % accuracy_score(y, final_model.predict(X)))

feature_to_coef = { word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])}

new_words = {'kiffé': 3.5, 'neuf': 3.5, 'bonne': 2.5, 'bon': 2.5, 'bel': 2.5, 'peu': -2.5}

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(feature_to_coef)
analyzer.lexicon.update(new_words)
lexicon = analyzer.lexicon


call = []
vs_compound = []
vs_pos = []
vs_neu = []
vs_neg = []
 
for i in range(0, len(datasettest)):
    call.append(corpustest[i])
    vs_compound.append(analyzer.polarity_scores(str(datasettest['General comments'][i]))['compound'])
    vs_pos.append(analyzer.polarity_scores(str(datasettest['General comments'][i]))['pos'])
    vs_neu.append(analyzer.polarity_scores(str(datasettest['General comments'][i]))['neu'])
    vs_neg.append(analyzer.polarity_scores(str(datasettest['General comments'][i]))['neg'])

final = pd.DataFrame({'Sentences': datasettest['General comments'],
                        'Compound': vs_compound,
                        'Positive': vs_pos,
                        'Neutral': vs_neu,
                        'Negative': vs_neg})
final = final[['Sentences', 'Compound','Positive', 'Neutral', 'Negative']]    


text = "Ecran de télévision horrible : très petit et surtout d'une définition exécrable. La durée du trajet obligerait à porter une attention toute particulière à ce point pour pouvoir profiter du bon choix de films proposés."
a = 0
for t in text.split() :
    try:
        a = a + lexicon[t]
        print(t,lexicon[t])    
    except:
        pass
print(a)



#Export dictionnary 
df = pd.DataFrame(data=lexicon, index=[0])
df = (df.T) #transpose
print (df)
df.to_excel('lexicon.xlsx')    