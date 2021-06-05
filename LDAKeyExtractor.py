#!/usr/bin/env python
# coding: utf-8

import Preprocessor

import pyLDAvis.gensim_models as gensimvis
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
from gensim.models import LdaModel
from gensim.models import CoherenceModel

class LDAKeyExtractor:
    def __init__(self):
        self.preprocessor = Preprocessor()

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

    def extract_keyword(self,corpus,id2word):
        self.NUM_TOPICS = compute_coherence()
        self.lda_model = LdaModel(corpus, id2word=id2word, num_topics=self.NUM_TOPICS)
        self.topic_term_prob = get_topic_term_prob(lda_model)
        self.prepared_data = gensimvis.prepare(lda_model, corpus, id2word)
        self.pp = build_NT_list(self.NUM_TOPICS) # NO Topic0
        for i in range(self.NUM_TOPICS+1):
            self.pp[i] = prepared_data.sorted_terms(topic=i, _lambda=0.6).iloc[:20]
        self.idx_topic = build_NT_list(self.NUM_TOPICS)
        for i in range(1, self.NUM_TOPICS + 1):
            self.idx_topic[i] = pp[i].iloc[:20].index
        return self.idx_topic

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











