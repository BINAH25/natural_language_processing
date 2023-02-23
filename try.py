import pke
from summarizer import Summarizer
import nltk
import pprint
import itertools
import re
import string
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
print (keywords)
"""filtered_keys=[]
for keyword in keywords:
    if keyword.lower() in summarized_text.lower():
        filtered_keys.append(keyword)
        
print (filtered_keys)"""