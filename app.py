from preprocessor import Preprocessor
from LDAkey_extractor import LDAKeyExtractor
from textrank import TextRank
import pickle
from multiprocessing import freeze_support
import time

class LDA_TR:
    def __init__(self):
        self.preprocessor = Preprocessor()

    def save_topics(self,news,id_news):
        id2word,corpus,NUM_TOPICS,bigram_model = self.preprocessor.preprocess(news)
        print("preprocessing done")
        lda_extractor = LDAKeyExtractor(NUM_TOPICS)
        idx_topic,lda_model,_ = lda_extractor.extract_keyword(corpus,id2word)
        print("lda modeling")
        corp_doc_topic, topic_docs_save = self.preprocessor.cluster_extract_sentences(lda_model,idx_topic,corpus,news,
                                                                                      id_news,NUM_TOPICS,id2word,bigram_model)
        print("clustering done")

        textrank = TextRank(corp_doc_topic)
        print("textrank done")
        keywords = textrank.extract_keyword()
        print("keyword extracted")
        ext_topic_cluster = dict()
        print("힝")
        for i in range(1, NUM_TOPICS+1):
            top_save = dict()
            for j in range(len(topic_docs_save[i])):
                top_save[topic_docs_save[i][j][0]] = topic_docs_save[i][j][1]
            ext_topic_cluster[i] = [keywords[i - 1], top_save]
        return ext_topic_cluster, NUM_TOPICS






def run():
    for n in range(5):
        freeze_support()
        st = time.time()
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
        lda_tr = LDA_TR()
        etc, num = lda_tr.save_topics(news,id_news)
        print('loop')
        print(num)
        print(time.time()-st)
        print(len(news))
        save_file = [[]]
        for i in range(1,num+1):
            save_file.append(sorted(etc[i][0],reverse=True, key = lambda x: len(x[2])))
            print(etc[i][0])
        with open("data%d.pickle"% n, "wb") as fw:
            pickle.dump(save_file, fw)

if __name__ == '__main__':
    run()





