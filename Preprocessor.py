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
        return  self.id2word, self.corpus

    
