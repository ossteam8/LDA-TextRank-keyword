# Abstract
### Korean keyword extractor
Combined LDA and TextRank Algorithm 

# Running project 
install [requirements.txt](requirements.txt)

In order to install **Mecab** please use the line below at your terminal
```
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

## Actual Running
### Required
1. Articles
\["article1","article2",...,"articleN"\] type: list
2. Id of Articles
\[1,2,3,4,...,N\] type: list
does not need to be in-order
3. stop.txt
a,is,...,@@@ -> distinguished by commas(,) in .txt file

**[stop.txt](stop.txt)** is prepared and provided by [linkyouhj](https://github.com/linkyouhj) and [Chae Hui Seon](https://github.com/chaehuiseon)

### For demo
```
python3 app.py
```
This will run with sample data (about 4000 news articles and id)

# Keyword Extracting
Reference: [https://www.koreascience.or.kr/article/JAKO202028851207548.pdf](https://www.koreascience.or.kr/article/JAKO202028851207548.pdf)

Below is the Keyword-Extracting process
1. LDA 
2. Choose news article's sentences which contributes to each topics
3. TextRank


## Preprocessing
### [preprocessor.py](preprocessor.py)

LDA토픽 모델링을 위해 다음과 같은 순서로 문서들을 전처리한다.

```
(1) 명사 추출 => (2) 불용어 제거 => (3) N-gram
```

먼저, 전처리를 위해 입력되는 문서(뉴스 기사들)는 ["기사1","기사2","기사3",....,"기사N"]의 형식이다.

명사 추출은 한글 형태소 분석기 Mecab을 사용한다.

사용자 단어 사전을 구축하여 형태소 분석이 잘 되지 않아 추출되지 않는 명사를 잘 인식할 수 있도록 돕는다.

사용자 단어 사전 설치 방법 : 

사용자 단어 사전 다운 :

다음으로 불용어로 판단되는 단어들을 삭제 한다.

마지막으로 복합명사를 처리하고, 뉴스 기사에 자주 등장하는 단어 중에, 연속적으로 의미 있는 단어로 구성된 문구를 처리하기 위해 N-gram으로 토큰화하여 코퍼스를 준비한다.



## LDA
### [LDAkey_extractor](LDAkey_extractor)
Gensim's LDA topic modeling algorithm implemented

Reference: [https://lovit.github.io/nlp/2018/09/27/pyldavis_lda/](https://lovit.github.io/nlp/2018/09/27/pyldavis_lda/)

## TextRank
### [textrank.py](textrank.py)

Reference: [https://lovit.github.io/nlp/2019/04/30/textrank/](https://lovit.github.io/nlp/2019/04/30/textrank/)

LDA를 통해 선정된 각 토픽별로 토픽에 기여하는 문서들을 대상으로 Relevance Top 20 단어를 포함하는 문장들을 추출한다. 추출된 문장들에 대해서 앞에서 한 것과 같이 전처리 한다.

따라서 TextRank에 적용되는 입력 형식은 

[ [topic1_문장1,topic1_문장2,...,topic1_문장N],[topic2_문장1,topic2_문장2,...,topic2_문장N],...,[topicN_문장1,topicN_문장2,...,topicN_문장N] ]이다.

이후, Textrank를 알고리즘을 사용하여 각 토픽을 대표하는 단어로 토픽 키워드들을 추출한다. TextRank 알고리즘은 word graph를 구축한 뒤, Graph ranking 알고리즘인 PageRank 를 이용하여 키워드를 추출한다.
