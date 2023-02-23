import pke
from flashtext import KeywordProcessor
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk
from nltk.tokenize import sent_tokenize
from summarizer import Summarizer
import nltk
import pprint
import itertools
import re
import string
import requests
import json
import random
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('popular')
f = open("sample.txt","r")
full_text = f.read()
model = Summarizer()
result = model(full_text, min_length=60, max_length = 500 , ratio = 0.4)
summarized_text = ''.join(result)

def get_nouns_multipartite(text):
    out = []
    extractor = pke.unsupervised.MultipartiteRank()
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    extractor.load_document(input=text,stoplist=stoplist,language='en')
    pos = {'PROPN'}
    #pos = {'VERB', 'ADJ', 'NOUN'}
    extractor.candidate_selection(pos=pos, )
    # 4. build the Multipartite graph and rank candidates using random walk,
    #    alpha controls the weight adjustment mechanism, see TopicRank for
    #    threshold/method parameters.
    extractor.candidate_weighting(alpha=1.1,
                                  threshold=0.75,
                                  method='average')
    keyphrases = extractor.get_n_best(n=20)
    for key in keyphrases:
        out.append(key[0])
    return out

keywords = get_nouns_multipartite(full_text) 
#print (keywords)
filtered_keys=[]
for keyword in keywords:
    if keyword.lower() in summarized_text.lower():
        filtered_keys.append(keyword)
        
#print (filtered_keys)

def tokenize_sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    # Remove any short sentences less than 20 letters.
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)
    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
        return keyword_sentences
    
sentences = tokenize_sentences(summarized_text)
keyword_sentence_mapping = get_sentences_for_keyword(filtered_keys, sentences)
        
#print (keyword_sentence_mapping)
#print(filtered_keys)

# Distractors from Wordnet
def get_distractors_wordnet(syn,word):
    distractors=[]
    word = word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors

def get_wordsense(sent,word):
    word= word.lower()
    
    if len(word.split())>0:
        word = word.replace(" ","_")
    
    
    synsets = wn.synsets(word,'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output =  adapted_lesk(sent, word, pos='n')
        lowest_index = min (synsets.index(wup),synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None