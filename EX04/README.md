### Aiffel_peer_review(5/23)
### Team : 심재형
### Reviewer : 김신성
-----------------------------------------------------------------------
## I review project 1,2 following above rules
- 1.Did the code work properly and fix the given issue?
- 2.Did I look at the comments and understand the author's code? And it is suitable?
- 3.Is there a possibility that the code will cause an error?
- 4.Did the code writer understand and write the code correctly?
- 5.Is the code concise and expandable?
- 6.etc
-----------------------------------------------------------------------
## Project 4
- Dataset : Korean movie review Word text
- Problem : Encoding and decoding word context & Learning NLP model

### Data load, Handling missing data
```python
from konlpy.tag import Mecab
import numpy as np
from collections import Counter

tokenizer = Mecab()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

def load_data(train_data, test_data, num_words=10000):
    train_data.drop_duplicates(subset=['document'], inplace=True)
    train_data = train_data.dropna(how = 'any') 
    test_data.drop_duplicates(subset=['document'], inplace=True)
    test_data = test_data.dropna(how = 'any') 
    
    X_train = []
    for sentence in train_data['document']:
        temp_X = tokenizer.morphs(sentence) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_train.append(temp_X)

    X_test = []
    for sentence in test_data['document']:
        temp_X = tokenizer.morphs(sentence) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_test.append(temp_X)
    
    words = np.concatenate(X_train).tolist()
    counter = Counter(words)
    counter = counter.most_common(10000-4)
    vocab = ['<PAD>', '<BOS>', '<UNK>', '<UNUSED>'] + [key for key, _ in counter]
    word_to_index = {word:index for index, word in enumerate(vocab)}
        
    def wordlist_to_indexlist(wordlist):
        return [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in wordlist]
        
    X_train = list(map(wordlist_to_indexlist, X_train))
    X_test = list(map(wordlist_to_indexlist, X_test))
    
    return X_train, np.array(list(train_data['label'])), X_test, np.array(list(test_data['label'])), word_to_index
    
X_train, y_train, X_test, y_test, word_to_index = load_data(train_data, test_data)
```
"Each line has a proper footnote, which is very easy to read, and the code is very concise." GOOD!

### Word encoding and decoding
```python
# 문장 1개를 활용할 딕셔너리와 함께 주면, 단어 인덱스 리스트 벡터로 변환해 주는 함수입니다. 
# 단, 모든 문장은 <BOS>로 시작하는 것으로 합니다. 
def get_encoded_sentence(sentence, word_to_index):
    return [word_to_index['<BOS>']]+[word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence.split()]

# 여러 개의 문장 리스트를 한꺼번에 단어 인덱스 리스트 벡터로 encode해 주는 함수입니다. 
def get_encoded_sentences(sentences, word_to_index):
    return [get_encoded_sentence(sentence, word_to_index) for sentence in sentences]

# 숫자 벡터로 encode된 문장을 원래대로 decode하는 함수입니다. 
def get_decoded_sentence(encoded_sentence, index_to_word):
    return ' '.join(index_to_word[index] if index in index_to_word else '<UNK>' for index in encoded_sentence[1:])  #[1:]를 통해 <BOS>를 제외

# 여러 개의 숫자 벡터로 encode된 문장을 한꺼번에 원래대로 decode하는 함수입니다. 
def get_decoded_sentences(encoded_sentences, index_to_word):
    return [get_decoded_sentence(encoded_sentence, index_to_word) for encoded_sentence in encoded_sentences]
```
"Footnotes are appropriate and codes are concise. I think it will be more understandable if you show output with a few head contexts." Well done!

```python
from gensim.models.keyedvectors import Word2VecKeyedVectors
word_vectors = Word2VecKeyedVectors.load_word2vec_format(word2vec_file_path, binary=False)
vector = word_vectors['재미']
vector
word_vectors.similar_by_word("쓰레기")
[('개뿔', 0.9453624486923218),
 ('조잡', 0.94145667552948),
 ('민망', 0.9368619918823242),
 ('배슬기', 0.9357537627220154),
 ('짜집', 0.9355328679084778),
 ('체자', 0.934526264667511),
 ('잡탕', 0.9339403510093689),
 ('초보', 0.9310812950134277),
 ('볼라', 0.930091142654419),
 ('혐', 0.9273268580436707)]
  ```
"It was a good code that intuitively showed that the similarity with the words that felt close came out similar." Very good!!

### Conclusion 
전체적으로 코딩이 간결하고 각 줄의 설명이 직관적이라 이해하기 쉬웠습니다.
나중에 다른 NLP모델을 배운다면 오늘 했던 Word context를 가지고 학습시켜보는것도 재미있을것 같습니다.

수고하셨습니다!

## 1,2,3,4,5 pass (O)
-----------------------------------------------------------------------
