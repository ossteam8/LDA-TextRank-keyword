#!/usr/bin/env python
# coding: utf-8


import pyLDAvis.gensim_models as gensimvis
import warnings
warnings.filterwarnings(action='ignore')
from gensim.models import LdaModel

class LDAKeyExtractor:
    def __init__(self,NUM_TOPICS):
        self.NUM_TOPICS = NUM_TOPICS

    def get_topic_term_prob(self,lda_model):
        topic_term_freqs = lda_model.state.get_lambda()
        topic_term_prob = topic_term_freqs / topic_term_freqs.sum(axis=1)[:, None]
        return topic_term_prob

    def build_NT_list(self,NUM_TOPICS):
        # list size: NUM_TOPICS+1
        NT_list = []
        for i in range(NUM_TOPICS + 1):
            NT_list.append([])
        return NT_list

    def extract_keyword(self,corpus,id2word):
        lda_model = LdaModel(corpus, id2word=id2word, num_topics=self.NUM_TOPICS)
        topic_term_prob = self.get_topic_term_prob(lda_model)
        prepared_data = gensimvis.prepare(lda_model, corpus, id2word)
        pp = self.build_NT_list(self.NUM_TOPICS) # NO Topic0
        for i in range(self.NUM_TOPICS+1):
            pp[i] = prepared_data.sorted_terms(topic=i, _lambda=0.6).iloc[:20]
        idx_topic = self.build_NT_list(self.NUM_TOPICS)
        for i in range(1, self.NUM_TOPICS + 1):
            idx_topic[i] = pp[i].iloc[:20].index
        return idx_topic, lda_model, topic_term_prob












