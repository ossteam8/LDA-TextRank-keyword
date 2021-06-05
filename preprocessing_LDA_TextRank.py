#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import platform
from collections import Counter

import numpy as np

# tokenizer import
from konlpy.tag import Okt, Komoran, Hannanum, Kkma

#운영체제에 따라 mecab설치 방법이 다름.
if platform.system() == "Windows":
    try:
        from eunjeon import Mecab
    except:
        print("please install eunjeon module")
else:  # Ubuntu일 경우
    from konlpy.tag import Mecab

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict

from functools import partial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import gensim
from gensim import corpora
import kss
import time
import warnings
warnings.filterwarnings(action='ignore')
from gensim.models import CoherenceModel
from gensim.models import LdaModel


# In[3]:


with open('economy_id.pickle', 'rb') as f:
    id_news = pickle.load(f)
with open('politic_id.pickle', 'rb') as f:
    id_news.extend(pickle.load(f))
with open('society_id.pickle', 'rb') as f:
    id_news.extend(pickle.load(f))


# In[4]:



with open('economy_contents.pickle', 'rb') as f:
    news = pickle.load(f)
with open('politic_contents.pickle', 'rb') as f:
    news.extend(pickle.load(f))
with open('society_contents.pickle', 'rb') as f:
    news.extend(pickle.load(f))

start = time.time()

def merge_news(news):
    lines = ""
    for i in news:
        lines = lines+i
    return lines


#불용어 사전에 "명" 추가
stop = "이후,부,대,시,바,외,속,차,점,단,후,듯,곳,필요,기여,이용,만,불,원,중,씨,헤럴드,전재,뿐,면,반,관련,기대,제기,우려,번,지적,배포,금지,말,건,분,회,간,내,수,거,게,명,직전,아,휴,아이구,아이쿠,아이고,어,나,우리,저희,따라,의해,을,를,에,의,가,으로,로,에게,뿐이다,의거하여,근거하여,입각하여,기준으로,예하면,예를 들면,예를 들자면,저,소인,소생,저희,지말고,하지마,하지마라,다른,물론,또한,그리고,비길수 없다,해서는 안된다,뿐만 아니라,만이 아니다,만은 아니다,막론하고,관계없이,그치지 않다,그러나,그런데,하지만,든간에,논하지 않다,따지지 않다,설사,비록,더라도,아니면,만 못하다,하는 편이 낫다,불문하고,향하여,향해서,향하다,쪽으로,틈타,이용하여,타다,오르다,제외하고,이 외에,이 밖에,하여야,비로소,한다면 몰라도,외에도,이곳,여기,부터,기점으로,따라서,할 생각이다,하려고하다,이리하여,그리하여,그렇게 함으로써,하지만,일때,할때,앞에서,중에서,보는데서,으로써,로써,까지,해야한다,일것이다,반드시,할줄알다,할수있다,할수있어,임에 틀림없다,한다면,등,등등,제,겨우,단지,다만,할뿐,딩동,댕그,대해서,대하여,대하면,훨씬,얼마나,얼마만큼,얼마큼,남짓,여,얼마간,약간,다소,좀,조금,다수,몇,얼마,지만,하물며,또한,그러나,그렇지만,하지만,이외에도,대해 말하자면,뿐이다,다음에,반대로,반대로 말하자면,이와 반대로,바꾸어서 말하면,바꾸어서 한다면,만약,그렇지않으면,까악,툭,딱,삐걱거리다,보드득,비걱거리다,꽈당,응당,해야한다,에 가서,각,각각,여러분,각종,각자,제각기,하도록하다,와,과,그러므로,그래서,고로,한 까닭에,하기 때문에,거니와,이지만,대하여,관하여,관한,과연,실로,아니나다를가,생각한대로,진짜로,한적이있다,하곤하였다,하,하하,허허,아하,거바,와,오,왜,어째서,무엇때문에,어찌,하겠는가,무슨,어디,어느곳,더군다나,하물며,더욱이는,어느때,언제,야,이봐,어이,여보시오,흐흐,흥,휴,헉헉,헐떡헐떡,영차,여차,어기여차,끙끙,아야,앗,아야,콸콸,졸졸,좍좍,뚝뚝,주룩주룩,솨,우르르,그래도,또,그리고,바꾸어말하면,바꾸어말하자면,혹은,혹시,답다,및,그에 따르는,때가 되어,즉,지든지,설령,가령,하더라도,할지라도,일지라도,지든지,몇,거의,하마터면,인젠,이젠,된바에야,된이상,만큼,어찌됏든,그위에,게다가,점에서 보아,비추어 보아,고려하면,하게될것이다,일것이다,비교적,좀,보다더,비하면,시키다,하게하다,할만하다,의해서,연이서,이어서,잇따라,뒤따라,뒤이어,결국,의지하여,기대여,통하여,자마자,더욱더,불구하고,얼마든지,마음대로,주저하지 않고,곧,즉시,바로,당장,하자마자,밖에 안된다,하면된다,그래,그렇지,요컨대,다시 말하자면,바꿔 말하면,즉,구체적으로,말하자면,시작하여,시초에,이상,허,헉,허걱,바와같이,해도좋다,해도된다,게다가,더구나,하물며,와르르,팍,퍽,펄렁,동안,이래,하고있었다,이었다,에서,로부터,까지,예하면,했어요,해요,함께,같이,더불어,마저,마저도,양자,모두,습니다,가까스로,하려고하다,즈음하여,다른,다른 방면으로,해봐요,습니까,했어요,말할것도 없고,무릎쓰고,개의치않고,하는것만 못하다,하는것이 낫다,매,매번,들,모,어느것,어느,로써,갖고말하자면,어디,어느쪽,어느것,어느해,어느 년도,라 해도,언젠가,어떤것,어느것,저기,저쪽,저것,그때,그럼,그러면,요만한걸,그래,그때,저것만큼,그저,이르기까지,할 줄 안다,할 힘이 있다,너,너희,당신,어찌,설마,차라리,할지언정,할지라도,할망정,할지언정,구토하다,게우다,토하다,메쓰겁다,옆사람,퉤,쳇,의거하여,근거하여,의해,따라,힘입어,그,다음,버금,두번째로,기타,첫번째로,나머지는,그중에서,견지에서,형식으로 쓰여,입장에서,위해서단지,의해되다,하도록시키다,뿐만아니라,반대로,전후,전자,앞의것,잠시,잠깐,하면서,그렇지만,다음에,그러한즉,그런즉,남들,아무거나,어찌하든지,같다,비슷하다,예컨대,이럴정도로,어떻게,만약,만일,위에서 서술한바와같이,인 듯하다,하지 않는다면,만약에,무엇,무슨,어느,어떤,아래윗,조차,한데,그럼에도 불구하고,여전히,심지어,까지도,조차도,하지 않도록,않기 위하여,때,시각,무렵,시간,동안,어때,어떠한,하여금,네,예,우선,누구,누가 알겠는가,아무도,줄은모른다,줄은 몰랏다,하는 김에,겸사겸사,하는바,그런 까닭에,한 이유는,그러니,그러니까,때문에,그,너희,그들,너희들,타인,것,것들,너,위하여,공동으로,동시에,하기 위하여,어찌하여,무엇때문에,붕붕,윙윙,나,우리,엉엉,휘익,윙윙,오호,아하,어쨋든,만 못하다,하기보다는,차라리,하는 편이 낫다,흐흐,놀라다,상대적으로 말하자면,마치,아니라면,쉿,그렇지 않으면,그렇지 않다면,안 그러면,아니었다면,하든지,아니면,이라면,좋아,알았어,하는것도,그만이다,어쩔수 없다,하나,일,일반적으로,일단,한켠으로는,오자마자,이렇게되면,이와같다면,전부,한마디,한항목,근거로,하기에,아울러,하지 않도록,않기 위해서,이르기까지,이 되다,로 인하여,까닭으로,이유만으로,이로 인하여,그래서,이 때문에,그러므로,그런 까닭에,알 수 있다,결론을 낼 수 있다,으로 인하여,있다,어떤것,관계가 있다,관련이 있다,연관되다,어떤것들,에 대해,이리하여,그리하여,여부,하기보다는,하느니,하면 할수록,운운,이러이러하다,하구나,하도다,다시말하면,다음으로,에 있다,에 달려 있다,우리,우리들,오히려,하기는한데,어떻게,어떻해,어찌됏어,어때,어째서,본대로,자,이,이쪽,여기,이것,이번,이렇게말하자면,이런,이러한,이와 같은,요만큼,요만한 것,얼마 안 되는 것,이만큼,이 정도의,이렇게 많은 것,이와 같다,이때,이렇구나,것과 같이,끼익,삐걱,따위,와 같은 사람들,부류의 사람들,왜냐하면,중의하나,오직,오로지,에 한하다,하기만 하면,도착하다,까지 미치다,도달하다,정도에 이르다,할 지경이다,결과에 이르다,관해서는,여러분,하고 있다,한 후,혼자,자기,자기집,자신,우에 종합한것과같이,총적으로 보면,총적으로 말하면,총적으로,대로 하다,으로서,참,그만이다,할 따름이다,쿵,탕탕,쾅쾅,둥둥,봐,봐라,아이야,아니,와아,응,아이,참나,년,월,일,령,영,일,이,삼,사,오,육,륙,칠,팔,구,이천육,이천칠,이천팔,이천구,하나,둘,셋,넷,다섯,여섯,일곱,여덟,아홉,령,영 "


news_doc = merge_news(news)
text_list = kss.split_sentences(news_doc)
k = []

stop_words = stop.split(',')
mecab = Mecab()
stime = time.time()
for i in news :
    sentences = mecab.nouns(i)
    result2 = []
    for word in sentences :
        if word not in stop_words:
            result2.append(word)
    k.append(result2)
print("split sentence finish:",time.time()-stime)
print(len(k))


bigram = gensim.models.Phrases(k, min_count = 5, threshold = 1)

bigram_model = gensim.models.phrases.Phraser(bigram)

bigram_document = [bigram_model[nouns] for nouns in k]

for i in range(len(bigram_document)):
    if word in stop_words:
        bigram_document.drop(i)


id2word = corpora.Dictionary(bigram_document)
corpus = [id2word.doc2bow(doc) for doc in bigram_document]




coherence_score=[]
t_min = 2
t_max = 20
for i in range(t_min,t_max):
    model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=i)
    coherence_model = CoherenceModel(model, texts=bigram_document, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model.get_coherence()
    print('n=',i,'\nCoherence Score: ', coherence_lda)
    coherence_score.append(coherence_lda)


# In[ ]:


co_sc = np.array(coherence_score)
NUM_TOPICS = np.argmax(co_sc)+t_min


# In[ ]:


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=id2word, passes=15)
topics = ldamodel.print_topics(num_words=5)

def get_topic_term_prob(lda_model):
    topic_term_freqs = lda_model.state.get_lambda()
    topic_term_prob = topic_term_freqs / topic_term_freqs.sum(axis=1)[:, None]
    return topic_term_prob

lda_model = LdaModel(corpus, id2word=id2word, num_topics=NUM_TOPICS)

topic_term_prob = get_topic_term_prob(lda_model)

prepared_data = gensimvis.prepare(lda_model, corpus, id2word)

pyLDAvis.display(prepared_data)

for i, topic_list in enumerate(ldamodel[corpus]):
    print(id_news[i],'번째 문서의 topic 비율은',topic_list)

pp = [[]]*(NUM_TOPICS+1)#NO Topic0
for i in range(NUM_TOPICS):
    pp[i] = prepared_data.sorted_terms(topic = i,_lambda=0.6).iloc[:20]

#index만 추출하기(index가 corpus의 index랑 동일함 ㅇㅇ)
idx_topic = [[]]*(NUM_TOPICS+1)
for i in range(1,len(pp)-1):
    idx_topic[i] = pp[i].iloc[:20].index

def build_NT_list():
    #list size: NUM_TOPICS+1
    NT_list = []
    for i in range(NUM_TOPICS+1):
        NT_list.append([])
    return

#topic별 clustering을 위한 topic_docs 초기화(list[[]]*n 이런 식으로 하면 안됨)
topic_docs = build_NT_list()
topic_docs_save = build_NT_list()

#append doc_index in the topic_docs cluster
for i, topic_list in enumerate(ldamodel[corpus]):
    topic_list.sort(reverse=True,key = lambda element:element[1])
    n = topic_list[0][0] + 1
    topic_docs[n].append([i,topic_list[0][1]])

#append doc_index in the topic_docs cluster
for i, topic_list in enumerate(ldamodel[corpus]):
    topic_list.sort(reverse=True,key = lambda element:element[1])
    n = topic_list[0][0] + 1
    topic_docs_save[n].append([id_news[i],topic_list[0][1]])


# #반환해줄 실제 news index와 가중치 저장해둔 list

for i in topic_docs:
    i.sort(reverse=True,key = lambda element:element[1])

topic_cluster = []
for i in range(NUM_TOPICS+1):
    topic_cluster.append("")
for i in range(1,NUM_TOPICS+1):
    for j in topic_docs[i]:
        topic_cluster[i] = topic_cluster[i]+news[j[0]]

for i, topic_list in enumerate(ldamodel[corpus]):
    topic_list.sort(reverse=True,key= lambda element:element[1])
    print(i,'번째 문서의 topic 비율은',topic_list)

topic_cluster_sentences = build_NT_list()
tcs = build_NT_list()

for i in range(1,NUM_TOPICS+1):
    mecab = Mecab()
    stop_words = stop.split(',')
    topic_cluster_sentences[i] = kss.split_sentences(topic_cluster[i])
    for j in topic_cluster_sentences[i]:
        sen_word = []
        sentences = mecab.nouns(j)
        for word in sentences :
            if word not in stop_words:
                sen_word.append(word)
        tcs[i].append(sen_word)

bigram_docs = build_NT_list()
#construct bigram_docs --> use the model made before
for i in range(1,NUM_TOPICS+1):
    bigram_docs[i] = [bigram_model[nouns] for nouns in tcs[i]]

corpus_docs = build_NT_list()
for i in range(1,NUM_TOPICS+1):    
    corpus_docs[i] = [id2word.doc2bow(doc) for doc in bigram_docs[i]]


corp_doc_ref = build_NT_list()


stime = time.time()
#corpus에서 index만 추출 (index,빈도수) 형태임
for i in range(1,NUM_TOPICS+1):
    if len(corpus_docs[i]) is not 0:
        for j in range(len(corpus_docs[i])):
            corp_doc_ref[i].append([])
            for k in range(len(corpus_docs[i][j])):
                a = corpus_docs[i][j][k][0]
                corp_doc_ref[i][j].append(a)

print(time.time()-stime)

#키워드 포함된 문장만 추출
stime = time.time()
corp_doc_topic = build_NT_list()

for i in range(1,len(corp_doc_ref)):
    for j in range(len(corp_doc_ref[i])):
        if len(set(idx_topic[i-1]).intersection(corp_doc_ref[i][j])) is not 0:
            corp_doc_topic[i].append(bigram_docs[i][j])

print(time.time()-stime)


# corp_doc_topic[N][M] : N개의 topic, M개의 문장 키워드 포함한 문장만 뽑아낸 것

def vectorize_sents(corp_doc_topic = corp_doc_topic, min_count=2, tokenizer="mecab", noun=True):
    vectorizer = CountVectorizer(tokenizer=lambda x: x,lowercase=False)
    vec = vectorizer.fit_transform(corp_doc_topic)
    vocab_idx = vectorizer.vocabulary_
    idx_vocab = {idx: vocab for vocab, idx in vocab_idx.items()}
    
    return vec, vocab_idx, idx_vocab


def word_similarity_matrix(x, min_sim=0.3):
    sim_mat = 1 - pairwise_distances(x.T, metric="cosine")
    sim_mat[np.where(sim_mat <= min_sim)] = 0

    return sim_mat


def word_graph(
    corp_doc_topic = corp_doc_topic,
    min_count=2,
    min_sim=0.3,
    tokenizer="mecab",
    noun=True,
):

    mat, vocab_idx, idx_vocab = vectorize_sents(
        corp_doc_topic,min_count=min_count, tokenizer=tokenizer, noun=noun
    )

    mat = word_similarity_matrix(mat, min_sim=min_sim)

    return mat, vocab_idx, idx_vocab


def pagerank(x: np.ndarray, df=0.85, max_iter=50, method="iterative"):

    assert 0 < df < 1

    A = normalize(mat, axis=0, norm="l1")
    N = np.ones(A.shape[0]) / A.shape[0]

    if method == "iterative":
        R = np.ones(A.shape[0])
        for _ in range(max_iter):
            R = df * np.matmul(A, R) + (1 - df) * N
    elif method == "algebraic":
        R = np.linalg.inv((I - df * A))
        R = np.matmul(R, (1 - df) * N)

    return R


keywords = []


for i in range(len(corp_doc_topic)):
    if len(corp_doc_topic[i])!= 0:
        mat,vocab_idx, idx_vocab = word_graph(corp_doc_topic[i])
        R = pagerank(mat,method ="iterative")
        topk = 10
        idxs = R.argsort()[-topk:]
        keywords.append([(idx, R[idx], idx_vocab[idx]) for idx in reversed(idxs)])
        #keyword = [(R[idx]) for idx in reversed(idxs)]


for i in keywords:
    print(i[0])


end = time.time()



print(end-start)


def save_topics():
    ext_topic_cluster = dict()
    tc = []
    for i in range(1,NUM_TOPICS):
        top_save = dict()
        for j in range(len(topic_docs_save[i])):
            top_save[topic_docs_save[i][j][0]] = topic_docs_save[i][j][1]
        ext_topic_cluster[i] = [keywords[i-1],top_save]
    return ext_topic_cluster, NUM_TOPICS

class LDAKeyExtractor:
    def __init__(self):
        self.stop_words = stop.split(',')

    def extract_keyword(self,news,id_news):
        self.news_doc = merge_news(news)
        self.news = news
        self.id_news = id_news
