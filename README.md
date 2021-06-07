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
Below is the Keyword-Extracting process
1. LDA 
2. Choose news article's sentences which contributes to each topics
3. TextRank

## Preprocessing
### [preprocessor.py](preprocessor.py)
Used Mecab and user-made stop-words dictionary

## LDA
### [LDAkey_extractor](LDAkey_extractor)
Gensim's LDA topic modeling algorithm implemented

## TextRank
### [textrank.py](textrank.py)
