
import platform

import numpy as np

#운영체제에 따라 mecab설치 방법이 다름.
if platform.system() == "Windows":
    try:
        from eunjeon import Mecab
    except:
        print("please install eunjeon module")
else:  # Ubuntu일 경우
    from konlpy.tag import Mecab

import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import kss
import warnings
warnings.filterwarnings(action='ignore')


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

    def construct_bigram_doc(self,nouns):
        bigram = gensim.models.Phrases(nouns, min_count=5, threshold=1)
        bigram_model = gensim.models.phrases.Phraser(bigram)

        bigram_document = [bigram_model[x] for x in nouns]
        return bigram_model, bigram_document

    def preprocess(self):
        self.news_doc = self.merge_news(self.news)
        self.nouns = self.noun_extractor(self.news)
        self.bigram_model, self.bigram_document = self.construct_bigram_doc(self.nouns)
        self.id2word = corpora.Dictionary(self.bigram_document)
        self.corpus = [self.id2word.doc2bow(doc) for doc in self.bigram_document]
        self.NUM_TOPICS = self.compute_coherence()
        return  self.id2word, self.corpus, self.NUM_TOPICS

    def build_NT_list(self):
        # list size: NUM_TOPICS+1
        NT_list = []
        for i in range(self.NUM_TOPICS + 1):
            NT_list.append([])
        return NT_list

    def compute_NUM_TOPICS(self,t_min=2,t_max=20):
        coherence_score = []
        for i in range(t_min, t_max):
            model = gensim.models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.id2word, num_topics=i)
            coherence_model = CoherenceModel(model, texts=self.bigram_document, dictionary=self.id2word, coherence='c_v')
            coherence_lda = coherence_model.get_coherence()
            print('n=', i, '\nCoherence Score: ', coherence_lda)
            coherence_score.append(coherence_lda)
        co_sc = np.array(coherence_score)
        NUM_TOPICS = np.argmax(co_sc) + t_min
        return NUM_TOPICS

    def cluster_extract_sentences(self,ldamodel,idx_topic):
        topic_docs = self.build_NT_list()
        topic_docs_save = self.build_NT_list()
        for i, topic_list in enumerate(ldamodel[self.corpus]):
            topic_list.sort(reverse=True, key=lambda element: element[1])
            n = topic_list[0][0] + 1
            topic_docs[n].append([i, topic_list[0][1]])

        for i, topic_list in enumerate(ldamodel[self.corpus]):
            topic_list.sort(reverse=True, key=lambda element: element[1])
            n = topic_list[0][0] + 1
            topic_docs_save[n].append([self.id_news[i], topic_list[0][1]])


        for i in topic_docs:
            i.sort(reverse=True, key=lambda element: element[1])

        topic_cluster = []
        for i in range(self.NUM_TOPICS + 1):
            topic_cluster.append("")
        for i in range(1, self.NUM_TOPICS + 1):
            for j in topic_docs[i]:
                topic_cluster[i] = topic_cluster[i] + self.news[j[0]]

        topic_cluster_sentences = self.build_NT_list()
        tcs = self.build_NT_list()

        for i in range(1, self.NUM_TOPICS + 1):
            mecab = Mecab()
            topic_cluster_sentences[i] = kss.split_sentences(topic_cluster[i])
            for j in topic_cluster_sentences[i]:
                sen_word = []
                sentences = mecab.nouns(j)
                for word in sentences:
                    if word not in self.stop_words:
                        sen_word.append(word)
                tcs[i].append(sen_word)
        bigram_docs = self.build_NT_list()
        for i in range(1, len(tcs)):
            bigram_docs[i] = [self.bigram_model[nouns] for nouns in tcs[i]]

        corpus_docs = self.build_NT_list()
        for i in range(1, self.NUM_TOPICS + 1):
            corpus_docs[i] = [self.id2word.doc2bow(doc) for doc in bigram_docs[i]]
        corp_doc_ref = self.build_NT_list()
        for i in range(1, self.NUM_TOPICS + 1):
            if len(corpus_docs[i]) is not 0:
                for j in range(len(corpus_docs[i])):
                    corp_doc_ref[i].append([])
                    for k in range(len(corpus_docs[i][j])):
                        a = corpus_docs[i][j][k][0]
                        corp_doc_ref[i][j].append(a)

        corp_doc_topic = self.build_NT_list()
        for i in range(self.NUM_TOPICS + 1):
            corp_doc_topic.append([])
        for i in range(1, len(corp_doc_ref)):
            for j in range(len(corp_doc_ref[i])):
                if len(set(idx_topic[i]).intersection(corp_doc_ref[i][j])) is not 0:
                    corp_doc_topic[i].append(bigram_docs[i][j])

        return corp_doc_topic,topic_docs_save