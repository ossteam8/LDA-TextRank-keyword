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

from gensim.models import LdaModel


class Preprocessor:
    def __init__(self,news,id_news):
        f = open("stop.txt", 'r')
        self.stop_words = f.read().split(',')
        f.close()
        self.news = news
        self.id_news = id_news


    def merge_news(self,news):
        lines = ""
        for i in news:
            lines = lines + i
        return lines

    def noun_extractor(self,news):
        nouns = []
        mecab = Mecab()
        for i in news:
            sentences = mecab.nouns(i)
            result = []
            for word in sentences:
                if word not in self.stop_words:
                    result.append(word)
            nouns.append(result)
        return nouns

    def construct_bigram_doc(self):
        bigram = gensim.models.Phrases(self.nouns, min_count=5, threshold=1)
        bigram_model = gensim.models.phrases.Phraser(bigram)

        bigram_document = [bigram_model[x] for x in self.nouns]
        return bigram_model, bigram_document

    def preprocess(self):
        self.news_doc = merge_news(self.news)
        self.nouns = noun_extractor(self.news)
        self.bigram_model, self.bigram_document = construct_bigram_doc()
        self.id2word = corpora.Dictionary(self.bigram_document)
        self.corpus = [self.id2word.doc2bow(doc) for doc in self.bigram_document]
        self.NUM_TOPICS = compute_coherence()
        return  self.id2word, self.corpus, self.NUM_TOPICS

    def build_NT_list(self):
        # list size: NUM_TOPICS+1
        NT_list = []
        for i in range(self.NUM_TOPICS + 1):
            NT_list.append([])
        return NT_list

    def compute_coherence(self,t_min=2,t_max=20):
        coherence_score = []
        for i in range(t_min, t_max):
            model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=i)
            coherence_model = CoherenceModel(model, texts=bigram_document, dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model.get_coherence()
            print('n=', i, '\nCoherence Score: ', coherence_lda)
            coherence_score.append(coherence_lda)
        co_sc = np.array(coherence_score)
        NUM_TOPICS = np.argmax(co_sc) + t_min
        return NUM_TOPICS

    def cluster_extract_sentences(self,news,id_news,corpus,ldamodel):
        topic_docs = build_NT_list()
        topic_docs_save = build_NT_list()
        for i, topic_list in enumerate(ldamodel[corpus]):
            topic_list.sort(reverse=True, key=lambda element: element[1])
            n = topic_list[0][0] + 1
            topic_docs[n].append([i, topic_list[0][1]])
        for i in topic_docs:
            i.sort(reverse=True, key=lambda element: element[1])

        topic_cluster = []
        for i in range(self.NUM_TOPICS + 1):
            topic_cluster.append("")
        for i in range(1, self.NUM_TOPICS + 1):
            for j in topic_docs[i]:
                topic_cluster[i] = topic_cluster[i] + news[j[0]]

        topic_cluster_sentences = build_NT_list()
        tcs = build_NT_list()

        for i in range(1, NUM_TOPICS + 1):
            mecab = Mecab()
            topic_cluster_sentences[i] = kss.split_sentences(topic_cluster[i])
            for j in topic_cluster_sentences[i]:
                sen_word = []
                sentences = mecab.nouns(j)
                for word in sentences:
                    if word not in self.stop_words:
                        sen_word.append(word)
                tcs[i].append(sen_word)
