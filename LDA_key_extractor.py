#!/usr/bin/env python
# coding: utf-8

import Preprocessor

import pyLDAvis.gensim_models as gensimvis

import warnings
warnings.filterwarnings(action='ignore')
from gensim.models import LdaModel


class LDAKeyExtractor:
    def __init__(self,stop = ""):
        self.stop_words = stop.split(',')

    def get_topic_term_prob(lda_model):
        topic_term_freqs = lda_model.state.get_lambda()
        topic_term_prob = topic_term_freqs / topic_term_freqs.sum(axis=1)[:, None]
        return topic_term_prob

    def build_NT_list(NUM_TOPICS):
        # list size: NUM_TOPICS+1
        NT_list = []
        for i in range(NUM_TOPICS + 1):
            NT_list.append([])
        return NT_list

    def extract_keyword(self,corpus,NUM_TOPICS,id2word):
        self.lda_model = LdaModel(corpus, id2word=id2word, num_topics=NUM_TOPICS)
        self.topic_term_prob = get_topic_term_prob(lda_model)
        self.prepared_data = gensimvis.prepare(lda_model, corpus, id2word)
        self.pp = build_NT_list(NUM_TOPICS) # NO Topic0
        for i in range(NUM_TOPICS+1):
            self.pp[i] = prepared_data.sorted_terms(topic=i, _lambda=0.6).iloc[:20]
        self.idx_topic = build_NT_list(NUM_TOPICS)
        for i in range(1, NUM_TOPICS + 1):
            self.idx_topic[i] = pp[i].iloc[:20].index
        return self.idx_topic











