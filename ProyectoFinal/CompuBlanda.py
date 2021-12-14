import nltk
"""DESCARGAS"""
#nltk.download('vader_lexicon')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
""""""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from collections import Counter
from collections import OrderedDict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment

#here you read your .txt file
with open ('texto.txt','r') as texto:
    text = texto.read()

#here you tokenize your text
sentences_tokens = sent_tokenize(text)
tokens = word_tokenize(text)
# remove punctuations
tokens_nop = [ t for t in tokens if t not in string.punctuation]
# convert to lower case to standardize dataset
tokens_lower=[ t.lower() for t in tokens_nop ]
# remove stopwords
## here you can create your own list or rely on standard stopwords in nltk
stop = stopwords.words('english')
final_tokens=[ t for t in tokens_lower if t not in stop ]
#here you create the word counter to determine which words are the most used in the text
c = Counter(final_tokens)
final_tokens_count = OrderedDict(c.most_common(10))
#Lemmatize

#here you creater your sentimen analyzer
analyzer = SentimentIntensityAnalyzer()
compound = 0
for sentence in sentences_tokens:
    #print(sentence)
    scores = analyzer.polarity_scores(sentence)
    #print("Compound ", scores["compound"])
    compound += scores["compound"]

#here you search for the NN tags in the text
wdl = nltk.WordNetLemmatizer()
final_tokens_lem = [wdl.lemmatize(Word,'v') for Word in tokens_lower]
tags = nltk.pos_tag(final_tokens_lem)
#print(tags)
noun = [word for (word, tag) in tags if "NN" in tag]
subject = []
for i in range(len(tags)):
    if i > 1 and i < len(tags)-1:
        if tags[i][1] == "NN" or tags[i][1] == "NNS":
            if (tags[i+1][1] == "VBP" or tags[i-1][1] == "DT"):
                subject.append(tags[i][0])
#print(tags)
cc = Counter(subject)
final_charac = OrderedDict(cc.most_common())

#here you display all the results
print("---LAS 10 PALABRAS MAS USADAS EN EL TEXTO---")
for token in final_tokens_count:
    print(token)
print("\n")
print("---SENTIMIENTO ENCONTRADO EN EL TEXTO---")
print(compound/len(sentences_tokens))
print("\n")
print("---SUJETOS ENCONTRADOS EN EL TEXTO---")
for character in final_charac:
    print(character)