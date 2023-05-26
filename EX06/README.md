## Aiffel_peer_review(5/26)
## Team : 심재형
## Reviewer : 김신성
-----------------------------------------------------------------------
## I review project 6 following above rules
- 1.Did the code work properly and fix the given issue?
- 2.Did I look at the comments and understand the author's code? And it is suitable?
- 3.Is there a possibility that the code will cause an error?
- 4.Did the code writer understand and write the code correctly?
- 5.Is the code concise and expandable?
- 6.etc
-----------------------------------------------------------------------
## Project 6
- Dataset : News text, headline dataset
- Problem : Generate summary text for using Seq2seq with Attention mechanism

### STep1 : Preprocessing 

   Data Load, pre processing(Drop missing data, Non-unique data, tokenize etc...)
   
### Sentence preprocessing
```python
def preprocess_sentence(sentence, remove_stopwords=True):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = BeautifulSoup(sentence, "lxml").text # <br />, <a href = ...> 등의 html 태그 제거
    sentence = re.sub(r'\([^)]*\)', '', sentence) # 괄호로 닫힌 문자열 (...) 제거 Ex) my husband (and myself!) for => my husband for
    sentence = re.sub('"','', sentence) # 쌍따옴표 " 제거
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")]) # 약어 정규화
    sentence = re.sub(r"'s\b","", sentence) # 소유격 제거. Ex) roland's -> roland
    sentence = re.sub("[^a-zA-Z]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub('[m]{2,}', 'mm', sentence) # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah
    
    # 불용어 제거 (Text)
    if remove_stopwords:
        tokens = ' '.join(word for word in sentence.split() if not word in stopwords.words('english') if len(word) > 1)
    # 불용어 미제거 (Summary)
    else:
        tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    return tokens
```
"The annotation is neat and the coding is neat overall" => Good job!


# Verg good point!!
```python
# 전체 Text 데이터에 대한 전처리 : 10분 이상 시간이 걸릴 수 있습니다. 
clean_text = []
# 인덱스 초기화
data.reset_index(drop=True, inplace=True)

# [[YOUR CODE]]
for i in range(0, len(data)):
    sen_data = preprocess_sentence(data["text"][i])
    clean_text.append(sen_data)
    if i / 10000 in list(range(1,10)):
        print(f"{i}번 째 반복중")
# 전처리 후 출력
print("Text 전처리 후 결과: ", clean_text[:5])
```
#### Outputs
```
10000번 째 반복중
20000번 째 반복중
30000번 째 반복중
40000번 째 반복중
50000번 째 반복중
60000번 째 반복중
70000번 째 반복중
80000번 째 반복중
90000번 째 반복중
```

"It is improtant to check running progress. This code can check running progress every 10000 iter times" =>  Great!!!

### Outlier data drop
```python
data_t = data # 혹시모를 데이터 복사
data = data[(data['text'].apply(lambda x: len(x.split()) <= text_max_len)) & (data['headlines'].apply(lambda x: len(x.split()) <= headlines_max_len))]
print(data)
print('전체 샘플수 :', (len(data)))
  ```
"Very simple and clever code by using 'labmda' and 'apply' method" => Good!!!
"Copy original data is also important things :)" Very Good!
 
```python
# 요약 데이터에는 시작 토큰과 종료 토큰을 추가한다.
data['decoder_input'] = data['headlines'].apply(lambda x : 'sostoken '+ x)
data['decoder_target'] = data['headlines'].apply(lambda x : x + ' eostoken')
data.head()
  ```
 "Simple and good code for padding pre, post token"
 
```python
from tensorflow.keras.layers import AdditiveAttention

# 어텐션 층(어텐션 함수)
attn_layer = AdditiveAttention(name='attention_layer')

# 인코더와 디코더의 모든 time step의 hidden state를 어텐션 층에 전달하고 결과를 리턴
attn_out = attn_layer([decoder_outputs, encoder_outputs])


# 어텐션의 결과와 디코더의 hidden state들을 연결
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

# 디코더의 출력층
decoder_softmax_layer = Dense(tar_vocab, activation='softmax')
decoder_softmax_outputs = decoder_softmax_layer(decoder_concat_input)

# 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()
```
"We applied "attention mechanism" well to a given dataset. For more logical calculations and details of "attention mechanism", please refer to the link below"

Link

https://arxiv.org/abs/1706.03762



### Conclusion

전체적으로 어떻게 text dataset을 전처리하고 Seq2seq 모델이 학습되는지에 대해서 잘 이해하고 있습니다.

추가적으로 어떻게 해야 조금 더 편하게, 능률을 높이면서 코딩을 할 수 있는지 알고 있는것 같습니다. 

각 줄에 충분한 주석과 시간이 많이 소요되는 작업을 할때 일정 길이만큼 Check point를 두는 것 역시 아주 좋은 방법이라고 생각합니다. 배워가요!!

앞으로 transformer 관련한 model architecture를 많이 공부할 것 같은데 "Attention is all you need" 라는 어텐션 메커니즘을 제시한 original paper에 대해서도 간단히 읽어보면 앞으로도 많은 도움이 될 것 같습니다.(링크 첨부하였습니다)

점점 더 간결하고 똑똑하게 코딩하시는 것 같습니다 수고하셨습니다!!


## 1,2,3,4,5 pass (O) + alpha 6(O) ( check point, Orinial data copy and using to summary function)  
-----------------------------------------------------------------------
