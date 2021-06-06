# Running project 
install [requirements.txt](requirements.txt)
In order to install **Mecab** please use the line below at your terminal
```
$ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

## Actual Running
### Required
1. Articles
\["article1","article2",...,"articleN"\] in the for of a list
2. Id of Articles
\[1,2,3,4,...,N\] does not need to be in-order
3. stop.txt
a,is,...,@@@ -> distinguished by commas(,) in .txt file
**[stop.txt](stop.txt)** is prepared by [linkyouhj](https://github.com/linkyouhj) and [Chae Hui Seon](https://github.com/chaehuiseon)
# Keyword Extracting
Below is the Keyword-Extracting process
1. LDA 
2. Choose news article's sentences which contributes to each topics
3. TextRank

## Preprocessing
Used Mecab and user-made stop-words dictionary

## LDA
Gensim's LDA topic modeling algorithm implemented

## TextRank
