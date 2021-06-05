from preprocessor import Preprocessor
from LDAkey_extractor import LDAKeyExtractor
from textrank import TextRank
import pickle

class LDA_TR:
    def __init__(self,news,id_news):
        self.news = news
        self.id_news = id_news
        self.preprocessor = Preprocessor(self.news, self.id_news)

    def save_topics(self):
        self.id2word,self.corpus,self.NUM_TOPICS = self.preprocessor.preprocess()
        print("preprocessing done")
        self.lda_extractor = LDAKeyExtractor(self.NUM_TOPICS)
        self.idx_topic,self.lda_model = self.lda_extractor.extract_keyword(self.corpus,self.id2word)
        print("lda modeling")
        self.corp_doc_topic, self.topic_docs_save = Preprocessor.cluster_extract_sentences(self.ldamodel,self.idx_topic)
        print("clustering done")

        self.textrank = TextRank(self.corp_doc_topic)
        print("textrank done")
        self.keywords = self.textrank.keyword_extraxtor()
        print("keyword extracted")
        ext_topic_cluster = dict()
        tc = []
        for i in range(1, self.NUM_TOPICS+1):
            top_save = dict()
            for j in range(len(self.topic_docs_save[i])):
                top_save[self.topic_docs_save[i][j][0]] = self.topic_docs_save[i][j][1]
            ext_topic_cluster[i] = [self.keywords[i - 1], top_save]
        return ext_topic_cluster, self.NUM_TOPICS



with open('sample_data/economy_id.pickle', 'rb') as f:
    id_news = pickle.load(f)
with open('sample_data/politic_id.pickle', 'rb') as f:
    id_news.extend(pickle.load(f))
with open('sample_data/society_id.pickle', 'rb') as f:
    id_news.extend(pickle.load(f))

with open('sample_data/economy_contents.pickle', 'rb') as f:
    news = pickle.load(f)
with open('sample_data/politic_contents.pickle', 'rb') as f:
    news.extend(pickle.load(f))
with open('sample_data/society_contents.pickle', 'rb') as f:
    news.extend(pickle.load(f))


def run():
    lda_tr = LDA_TR(news, id_news)
    etc, num = lda_tr.save_topics().multiprocessing.freeze_support()
    print('loop')
    print(etc[0][2])
if __name__ == '__main__':
    run()




