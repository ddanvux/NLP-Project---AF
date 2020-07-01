import os
import pandas as pd #import export dataframe
import re #regex to remove

import nltk #natural language
from nltk.corpus import stopwords #corpus = corpora
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import FreqDist
from collections import Counter

pathorg = "C:/Users/m429673/Desktop/Travail/Project Python Verbatims/"
os.chdir(pathorg)

        
lexicon = pd.read_excel('lexicon.xlsx',encoding='latin-1')
lexicon2 = pd.read_excel('dict1.xlsx',encoding='latin-1')
lexicon3 = pd.read_excel('features1000.xlsx',encoding='latin-1')

dictionnary = lexicon.set_index('Lexicon')['Score'].to_dict()
dictionnary2 = lexicon2.set_index('Lexicon')['Score'].to_dict()
dictionnary3 = lexicon3.set_index('Lexicon')['Score'].to_dict()

path = "C:/Users/m429673/Desktop/Travail/Verbatims/Adhoc/BonappchaudCDG/"
os.chdir(path)
datasettest = pd.read_excel('BACCDGfinal.xlsx', encoding='latin-1')

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

corpustest =[]
for i in range(0, len(datasettest)):
    review = get_tokens(str(datasettest['q89'][i]))
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
    
new_words = {'kiffé': 3.5, 'neuf': 3.5, 'bonne': 2.5, 'bon': 2.5, 'bel': 2.5, 'peu': -2.5,
             'désuet': -2.5, 'désuets': -2.5, 'pas': -5, 'need': -3, 'needs': -3, 'à améliorer': -3,
             'moyen': -1, 'could': -1.5, 'coulds': -1.5, 'peut': -1.5, 'peuvent':-1.5, 'have to': -2,
             'isn': -5, 'is not': -5, '!!': -3, '!!!':-5}

analyzer = SentimentIntensityAnalyzer()
lexiconall = analyzer.lexicon
analyzer.lexicon.update(dictionnary3)
analyzer.lexicon.update(new_words)

call = []
vs_compound = []
vs_pos = []
vs_neu = []
vs_neg = []
 
for i in range(0, len(datasettest)):
    call.append(corpustest[i])
    vs_compound.append(analyzer.polarity_scores(str(datasettest['q89'][i]))['compound'])
    vs_pos.append(analyzer.polarity_scores(str(datasettest['q89'][i]))['pos'])
    vs_neu.append(analyzer.polarity_scores(str(datasettest['q89'][i]))['neu'])
    vs_neg.append(analyzer.polarity_scores(str(datasettest['q89'][i]))['neg'])
            
final = pd.DataFrame({'Sentences': datasettest['q89'],
                        'Compound': vs_compound,
                        'Positive': vs_pos,
                        'Neutral': vs_neu,
                        'Negative': vs_neg})
final = final[['Sentences', 'Compound','Positive', 'Neutral', 'Negative']]    




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


#check score of each words
text = "Excellent service! Le personnel de bord était très professionnel et agréable. En plus j’a Été surclassé. J’ai adoré mon expérience. N’hésitez pas à me surclasser de nouveau. Merci"
a = 0
for t in text.split() :
    try:
        a = a + lexiconall[t]
        print(t,lexiconall[t])    
    except:
        pass
print(a)


#Export dictionnary 
df = pd.DataFrame(data=final)
df = (df.T) #transpose
print (df)
df.to_excel('finalllll.xlsx')    


date = pd.DataFrame(data=datasettest['Date of flight'])
df.insert(loc = 0, column = 'date', value = date )


#get bigrams vocab
bigrams = [b for line in corpustest for b in zip(line.split(" ")[:-1], line.split(" ")[1:])]
bgrm = []
for i in range(0, len(corpustest)):
    bigrams = corpustest[i].split(" ")[:-1], corpustest[i].split(" ")[1:]
    bgrm.append(bigrams)
bigrams = [b for line in corpustest for b in zip(line.split(" ")[:-1], line.split(" ")[1:])]

def get_bigram_freqdist(bigrams):
    freq_dict = {}
    for bigram in bigrams:
        if freq_dict.get(bigram):
            freq_dict[bigram] += 1
        else:
            freq_dict[bigram] = 1
    counter = Counter(freq_dict)
    return counter
fqbigrams = get_bigram_freqdist(bigrams)
topbi = fqbigrams.most_common(50)

counter_sum = Counter()
for i in range(0, len(corpustest)):
    for line in corpustest:
        tokens = nltk.word_tokenize(line)
        bigrams = list(nltk.bigrams(line.split()))
        bigramsC = Counter(bigrams)
        tokensC = Counter(tokens)
        both_counters = bigramsC + tokensC
        counter_sum += both_counters

#DATA FREQUENCY
freqw = ' '.join(corpustest)
freqw = freqw.split()
Freq = FreqDist(freqw) #frequency of words
top = Freq.most_common(50)

Freq.plot(10) #count

Vocab = set(freqw)
long_words = [w for w in Vocab if len(w) > 5 and Freq[w] > 10 ]

key_words = [word.lower() for word in freqw if len(word) >= 5]
FreqKW = FreqDist(key_words)
topKW = FreqKW.most_common(100)
FreqKW.plot(30)

BA = ' '.join(freqw)
pattern = re.compile(r"\w+\s{1}froid\s{1}\w+")
pattern2 = re.compile(r"\w+\s{1}\w+\s{1}froid\s{1}\w+\s{1}\w+")
beforeafter = pattern.findall(BA)
beforeafter2 = pattern2.findall(BA)    