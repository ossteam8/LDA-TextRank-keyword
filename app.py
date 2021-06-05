import preprocessor
import LDAkey_extractor
import textrank

class LDA_TR:
    def __init__(self,news,id_news):
        self.news = news
        self.id_news = id_news
        self.preprocessor = preprocessor(self.news, self.id_news)
        self.NUM_TOPICS = self.preprocessor.compute_NUM_TOPICS()
        self.lda_extractor = LDAkey_extractor(self.NUM_TOPICS)

    def save_topics(self):
        self.id2word,self.corpus,self.NUM_TOPICS = preprocessor.preprocess()
        self.corp_doc_topic, self.topic_docs_save = preprocessor.cluster_extract_sentences(self.ldamodel,self.idx_topic)
