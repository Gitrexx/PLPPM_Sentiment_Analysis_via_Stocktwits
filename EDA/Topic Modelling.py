
"""
text.info()
data.info()
aapl_1.shape[0]
"""

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import logging
import gensim 
from gensim import corpora
import pickle
import pyLDAvis
import pyLDAvis.gensim_models

path="C:/Users/User/Documents/EBAC-Specialist Cert-PLP/PLP Project"
stocklist=['aapl','fb','tsla','nvda','amzn']
random_state=10

#get bullish and bearish df for each stock
for stock in stocklist:
    exec(stock + "=pd.read_csv(path+'/Data/" + stock + "_pre_processed.csv',usecols=['body_preprocessed','sentiment'])")
    exec(stock + "_1=" + stock + "[" + stock + "['sentiment']=='Bullish']")
    exec(stock + "_0=" + stock + "[" + stock + "['sentiment']=='Bearish']")

#get minimum size of each sentiment
minsize0=10000000000000
minsize1=10000000000000
for stock in stocklist:
    exec("if " + stock + "_0.shape[0]<minsize0: minsize0=" + stock + "_0.shape[0]")
    exec("if " + stock + "_1.shape[0]<minsize1: minsize1=" + stock + "_1.shape[0]")
    
    
#stock balancing and concat

df0=[]
df1=[]
for stock in stocklist:
    exec(stock+"_sampled0=" + stock + "_0.sample(n=minsize0,random_state=random_state)")
    exec("df0.append(" + stock + "_sampled0)")
    exec(stock+"_sampled1=" + stock + "_1.sample(n=minsize1,random_state=random_state)")
    exec("df1.append(" + stock + "_sampled1)")
    
data0=pd.concat(df0,ignore_index=True)
data1=pd.concat(df1,ignore_index=True)

####################Preprocessing for bearish stocktwits
WNlemma = nltk.WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

def pre_process(text):
    tokens = tokenizer.tokenize(text)
    tokens=[ WNlemma.lemmatize(t.lower()) for t in tokens]
    tokens=[ t for t in tokens if t not in mystopwords]
    return(tokens)


excl_list=['cashtag_aapl','cashtag_qqq','cashtag_amd','cashtag_tsla'
           ,'cashtag_spy','apple','cashtag_msft','cashtag_goog','cashtag_fb'
           ,'cashtag_nflx','cashtag_tvix','cashtag_amzn','cashtag_nvda'
           ,'cashtag_gme','cashtag_avgo','cashtag_baba','cashtag_googl'
           ,'tesla','tsla','aapl','fb','nvda','amzn','amazon','think','get'
           #,'?','$','!',"'s",'.',",",':',"%",";","&","-","...","“","”"
           ,"ha",'party_popper','wa','u','x','er','bezos','let','gt','going'
           ,'1','2','3','5','4','20','7','6','10','15'
           ,'cashtag_amzn','cashtag_twtr','look','want','see','get','say','guy'
           ,'clinking_glasses','four_leaf_clover','money_bag','come'
           ,"rolling_on_the_floor_laughing",'cashtag_shop','cashtag_wmt'
           ,'face_with_tears_of_joy','red_apple','green_apple','mouth_face'
           ,'lol']

mystopwords=stopwords.words("english") + excl_list


#####for bearish stocktwits#########

text0=data0['body_preprocessed']
toks0 = text0.apply(pre_process)

# Use dictionary (built from corpus) to prepare a DTM (using frequency)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Filter off any words with document frequency less than 2, or appearing in more than 90% documents
dictionary0 = corpora.Dictionary(toks0)
dictionary0.filter_extremes(no_below=2, no_above=0.8)
corpus0 = [dictionary0.doc2bow(d) for d in toks0]

#save the corpus and dict
pickle.dump(corpus0, open(path+'/Models/corpus0.pkl', 'wb'))
dictionary0.save(path+'/Models/dictionary0.gensim')

num_topics=3
lda0_3 = gensim.models.ldamodel.LdaModel(corpus0, num_topics = num_topics, id2word = dictionary0, passes=10,chunksize=128,random_state=random_state)
lda0_3.save(path+'/Models/lda0_3.gensim')
lda0_3.show_topics()

num_topics=4
lda0_4 = gensim.models.ldamodel.LdaModel(corpus0, num_topics = num_topics, id2word = dictionary0, passes=10,chunksize=128,random_state=random_state)
lda0_4.save(path+'/Models/lda0_4.gensim')
lda0_4.show_topics()

num_topics=5
lda0_5 = gensim.models.ldamodel.LdaModel(corpus0, num_topics = num_topics, id2word = dictionary0, passes=10,chunksize=128,random_state=random_state)
lda0_5.save(path+'/Models/lda0_5.gensim')
lda0_5.show_topics()

num_topics=6
lda0_6 = gensim.models.ldamodel.LdaModel(corpus0, num_topics = num_topics, id2word = dictionary0, passes=10,chunksize=128,random_state=random_state)
lda0_6.save(path+'/Models/lda0_6.gensim')
lda0_6.show_topics()

#####for bullish stocktwits#########

text1=data1['body_preprocessed']
toks1 = text1.apply(pre_process)

# Use dictionary (built from corpus) to prepare a DTM (using frequency)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Filter off any words with document frequency less than 2, or appearing in more than 90% documents
dictionary1 = corpora.Dictionary(toks1)
dictionary1.filter_extremes(no_below=2, no_above=0.8)
corpus1 = [dictionary1.doc2bow(d) for d in toks1]

#save the corpus and dict
pickle.dump(corpus1, open(path + '/Models/corpus1.pkl', 'wb'))
dictionary1.save(path+'/Models/dictionary1.gensim')

num_topics=3
lda1_3 = gensim.models.ldamodel.LdaModel(corpus1, num_topics = num_topics, id2word = dictionary1, passes=10,chunksize=128,random_state=random_state)
lda1_3.save(path+'/Models/lda1_3.gensim')
lda1_3.show_topics()

num_topics=4
lda1_4 = gensim.models.ldamodel.LdaModel(corpus1, num_topics = num_topics, id2word = dictionary1, passes=10,chunksize=128,random_state=random_state)
lda1_4.save(path+'/Models/lda1_4.gensim')
lda1_4.show_topics()

num_topics=5
lda1_5 = gensim.models.ldamodel.LdaModel(corpus1, num_topics = num_topics, id2word = dictionary1, passes=10,chunksize=128,random_state=random_state)
lda1_5.save(path+'/Models/lda1_5.gensim')
lda1_5.show_topics()

num_topics=6
lda1_6 = gensim.models.ldamodel.LdaModel(corpus1, num_topics = num_topics, id2word = dictionary1, passes=10,chunksize=128,random_state=random_state)
lda1_6.save(path+'/Models/lda1_6.gensim')
lda1_6.show_topics()

#############visualization############

#load dict, corpus and model
dictionary_name='dictionary1'
corpus_name='corpus1'
model_name='lda1_5'

dictionary = gensim.corpora.Dictionary.load(path+'/Models/'+dictionary_name+'.gensim')
corpus = pickle.load(open(path + '/Models/'+corpus_name+'.pkl', 'rb'))
model = gensim.models.ldamodel.LdaModel.load(path+'/Models/'+model_name+'.gensim')

# Visualize the topics
pyLDAvis.enable_notebook()
lda_display = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
#pyLDAvis.display(lda_display)

#save the visualization as .html file
pyLDAvis.save_html(lda_display, path+'/Models/'+model_name+'_display.html')
#lda_display
