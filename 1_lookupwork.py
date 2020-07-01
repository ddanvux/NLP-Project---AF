# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:47:19 2019

@author: m429673
"""

import os
import pandas as pd
import nltk
import re

path = "C:/Users/m429673/Desktop/Travail/Verbatims/Adhoc/TDBrlt/"
os.chdir(path)
        
data = pd.read_excel('Source.xlsx',encoding='latin-1')

def get_tokens(data):
    tokens = nltk.word_tokenize(data) 
    return tokens
def get_sentences(data):
    sentences = nltk.sent_tokenize(data)
    return sentences


#word and similar word
corpus =[]
for i in range(0, len(data)):
    review = get_tokens(str(data['General comments'][i]))
    review = ' '.join(review)
    corpus.append(review)

Checkin = []
matchedLine = ''
stringToMatch = ['checkin', 'check in', 'check-in']
for line in corpus:
    if any(re.findall('|'.join(stringToMatch),line)):
         matchedLine = line
         Checkin.append(matchedLine)


#exact word only     
corpus =[]
for i in range(0, len(data)):
    review = get_tokens(str(data['General comments'][i]))
    review = ' '.join(review)
    review = review.split()
    corpus.append(review)
    
Checkin = []
matchedLine = ''
wordToMatch = ['checkin', 'check in', 'check-in']
for line in corpus:
   if any(word in line for word in wordToMatch):
         matchedLine = line
         Checkin.append(matchedLine)

#infos given
Infosgiven = []
matchedLine = ''
wordToMatch = ['infos', 'information', 'informations', 'information given', "announce", "announcement",'update','alert']
for line in Checkin:
   if any(word in line for word in wordToMatch):
         matchedLine = line
         matchedLine = ' '.join(matchedLine)
         Infosgiven.append(matchedLine)


#courtesy/helpfulness
Courtesy = []
matchedLine = ''
wordToMatch = ['helpful', 'professional', 'help','service']
for line in Checkin:
   if any(word in line for word in wordToMatch):
         matchedLine = line
         matchedLine = ' '.join(matchedLine)
         Courtesy.append(matchedLine)

#personal attention
Pattention = []
matchedLine = ''
wordToMatch = ['attention', 'personnel', 'staff', 'attendants', 'attendant','person']
for line in Checkin:
   if any(word in line for word in wordToMatch):
         matchedLine = line
         matchedLine = ' '.join(matchedLine)
         Pattention.append(matchedLine)




"""
#second way to extract exact word \b \b and also lookup the line up to multiple word at the same time
CheckinIG = [] #infos given
pat = re.compile(r"\bcheckin\b")
pat2 = re.compile(r"\binformation\b")
for line in Checkin:
    test = pat.findall(line)
    test2 = pat2.findall(line)
    if test != [] and test2 != []:
        matchedLine = line
        matchedLine = ''.join(matchedLine)
        CheckinIG.append(matchedLine)
"""


#negative/ positive

Checkin = []
matchedLine = ''
wordToMatch = ['checkin', 'check in', 'check-in']
for line in corpus:
   if any(word in line for word in wordToMatch):
         matchedLine = line
         matchedLine = ' '.join(matchedLine)
         Checkin.append(matchedLine)

corpustest =[]
for i in range(0, len(Checkin)):
    review = get_tokens(str(Checkin[i]))
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
    
for i in range(0, len(Checkin)):
    call.append(corpustest[i])
    vs_compound.append(analyzer.polarity_scores(str(Checkin[i]))['compound'])
    vs_pos.append(analyzer.polarity_scores(str(Checkin[i]))['pos'])
    vs_neu.append(analyzer.polarity_scores(str(Checkin[i]))['neu'])
    vs_neg.append(analyzer.polarity_scores(str(Checkin[i]))['neg'])
            
final = pd.DataFrame({'Sentences': Checkin,
                        'Compound': vs_compound,
                        'Positive': vs_pos,
                        'Neutral': vs_neu,
                        'Negative': vs_neg})
final = final[['Sentences', 'Compound','Positive', 'Neutral', 'Negative']]       

df = pd.DataFrame(data=final)
path = "C:/Users/m429673/Desktop/Travail/Verbatims/Adhoc/TDBrlt/"
os.chdir(path)
df.to_excel('finalllll.xlsx')  