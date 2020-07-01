# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:06:16 2019

@author: m429673
"""

import os
import pandas as pd
import nltk
import re

path = "C:/Users/m429673/Desktop/Travail/Verbatims/Adhoc/Codir P/"
os.chdir(path)

#create dataset
Janvier = pd.read_excel('Janvier.xlsx',encoding='latin-1')
Fevrier = pd.read_excel('Fevrier.xlsx',encoding='latin-1')
Mars = pd.read_excel('Mars.xlsx',encoding='latin-1')

dataset = Janvier.append(Fevrier)
dataset = dataset.append(Mars)
dataset = dataset.reset_index(drop=True)


#def steps
def get_tokens(data):
    tokens = nltk.word_tokenize(data) 
    return tokens
def get_sentences(data):
    sentences = nltk.sent_tokenize(data)
    return sentences

#Select rows based on multiple column conditions --> using "&":
# Select Data From BDL or CDG
Airport = ['CDG', 'ORY'] 
BDL = dataset.loc[(dataset['FinArrSt'].isin(Airport))]
BDL = BDL.reset_index(drop=True)
FRA = dataset.loc[(dataset['FinDepSt'].isin(Airport))]
FRA = FRA.reset_index(drop=True)


#Adhoc

#--------------------------ReviewBDL----------------------------

ReviewBDL =[]
for i in range(0, len(BDL)):
    review = get_tokens(str(BDL['q89'][i]))
    review = ' '.join(review)
    ReviewBDL.append(review)
ReviewBDL = pd.DataFrame(data=ReviewBDL)
BDL = BDL.drop(['q89','q81_pos_open','q81_neg_open'],axis = 1)
BDL = BDL.assign( review = ReviewBDL)

#BaggageBDL
BaggageBDL = []
matchedLine = ''
stringToMatch = ['baggage', 'luggage']
for line in BDL['review']:
    if any(re.findall('|'.join(stringToMatch),line)):
        matchedLine = line
        BaggageBDL.append(matchedLine)
BaggageBDL = BDL[BDL['review'].str.contains('|'.join(stringToMatch))]

#LoungeBDL
LoungeBDL = []
matchedLine = ''
stringToMatch = ['lounge', 'salon']
for line in BDL['review']:
    if any(re.findall('|'.join(stringToMatch),line)):
         matchedLine = line
         LoungeBDL.append(matchedLine)

#------------------------ReviewFRA------------------------------
         
ReviewFRA =[]
for i in range(0, len(FRA)):
    review = get_tokens(str(FRA['q89'][i]))
    review = ' '.join(review)
    ReviewFRA.append(review)
ReviewFRA = pd.DataFrame(data=ReviewFRA)
FRA = FRA.drop(['q89','q81_pos_open','q81_neg_open'],axis = 1)
FRA = FRA.assign( review = ReviewFRA)
    
#BaggageCDG
BaggageCDG = []
matchedLine = ''
stringToMatch = ['baggage', 'luggage']
for line in FRA['review']:
    if any(re.findall('|'.join(stringToMatch),line)):
         matchedLine = line
         BaggageCDG.append(matchedLine)

#---------------------ReviewGeneral------------------------------

ReviewGen =[]
for i in range(0, len(dataset)):
    review = get_tokens(str(dataset['q89'][i]))
    review = ' '.join(review)
    ReviewGen.append(review)     
ReviewGen = pd.DataFrame(data=ReviewGen)
dataset = dataset.drop(['q89','q81_pos_open','q81_neg_open'],axis = 1)
dataset = dataset.assign( review = ReviewGen)
    
#Post flight
Postflight = []
matchedLine = ''
stringToMatch = ['border', 'douane', 'immigration']
for line in dataset['review']:
    if any(re.findall('|'.join(stringToMatch),line)):
         matchedLine = line
         Postflight.append(matchedLine)
         
#Food & Beverage
FandB = []
matchedLine = ''
stringToMatch = ['food', 'drink', 'nourriture', 'boisson','repas','vegetarian','restauration','delicious','appetizers']
for line in dataset['review']:
    if any(re.findall('|'.join(stringToMatch),line)):
         matchedLine = line
         FandB.append(matchedLine)


#-----------------test neg/pos each dataset-----------------------

corpustest =[]
for i in range(0, len(BaggageBDL)):
    review = get_tokens(str(BaggageBDL[i]))
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
    
analyzer = SentimentIntensityAnalyzer()
lexiconall = analyzer.lexicon
analyzer.lexicon.update(dictionnary3)
analyzer.lexicon.update(new_words)

call = []
vs_compound = []
vs_pos = []
vs_neu = []
vs_neg = []
  
for i in range(0, len(BaggageBDL)):
    call.append(BaggageBDL[i])
    vs_compound.append(analyzer.polarity_scores(str(BaggageBDL[i]))['compound'])
    vs_pos.append(analyzer.polarity_scores(str(BaggageBDL[i]))['pos'])
    vs_neu.append(analyzer.polarity_scores(str(BaggageBDL[i]))['neu'])
    vs_neg.append(analyzer.polarity_scores(str(BaggageBDL[i]))['neg'])
            
FinalBaggage = pd.DataFrame({'Sentences': BaggageBDL,
                        'Compound': vs_compound,
                        'Positive': vs_pos,
                        'Neutral': vs_neu,
                        'Negative': vs_neg})
FinalBaggage = FinalBaggage[['Sentences', 'Compound','Positive', 'Neutral', 'Negative']]       


BaggageBDL = pd.DataFrame(data=BaggageBD, index=[0]
.to_excel('foodvocab.xlsx')
