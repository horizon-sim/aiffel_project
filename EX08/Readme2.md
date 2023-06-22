## Aiffel_peer_review(5/31)
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
## Project 8
- Dataset : Korea Chatbot
- Problem : training koeran text dataset using tranformer and inference step

   
### Data load and preprocessing
```python
data_dir = os.getenv("HOME") + "/aiffel/transformer_chatbot/data/ChatbotData .csv"
print(data_dir)
df = pd.read_csv(data_dir)

# 전처리 함수
def preprocess_sentence(sentence):
  # 입력받은 sentence를 소문자로 변경하고 양쪽 공백을 제거
  sentence = sentence.lower().strip()

  # 단어와 구두점(punctuation) 사이의 거리를 만듭니다.
  # 학생과 마침표 사이에 거리를 만듭니다.
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)

  # (a-z, A-Z, ".", "?", "!", ",")를 제외한 모든 문자를 공백인 ' '로 대체합니다.
  sentence = re.sub(r"[^가-힣0-1ㄱ-ㅎ?.!,]+", " ", sentence)
  sentence = sentence.strip()
  return sentence
  
print('전처리 후의 5번째 질문 샘플: {}'.format(questions[5]))
print('전처리 후의 5번째 답변 샘플: {}'.format(answers[5]))

전처리 후의 5번째 질문 샘플: 카드 망가졌어
전처리 후의 5번째 답변 샘플: 다시 새로 사는 게 마음 편해요 .
```
"Now I think I'm very good at dataload, preprocessing. It would be better if you could mention padding as well." => Good!


# Tokenizer , Data sampling batch
```python
# 질문과 답변 데이터셋에 대해서 Vocabulary 생성
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)

BATCH_SIZE = 64
BUFFER_SIZE = 20000

# 디코더는 이전의 target을 다음의 input으로 사용합니다.
# 이에 따라 outputs에서는 START_TOKEN을 제거하겠습니다.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
```
"The tokenizer was applied well and it was a very appropriate footnote to mention any start token removal" => Good!!

#### Scaled dat product attention
```python
# 스케일드 닷 프로덕트 어텐션 함수
def scaled_dot_product_attention(query, key, value, mask):
  # 어텐션 가중치는 Q와 K의 닷 프로덕트
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # 가중치를 정규화
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # 패딩에 마스크 추가
  if mask is not None:
    logits += (mask * -1e9)

  # softmax적용
  attention_weights = tf.nn.softmax(logits, axis=-1)

  # 최종 어텐션은 가중치와 V의 닷 프로덕트
  output = tf.matmul(attention_weights, value)
  return output
```
"The annotation is neat and the coding is neat overall" => Good!
"It would be better if the exact key query value dimension of the model is mentioned as well" => Well done!!

### CustomSchedul
```python
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  ```
"I think it will be a better tutorial if you explain the necessity and effectiveness of Custom Schedul together" => good!!

 




### Conclusion
전체적인 흐름에 대해서 잘 인지하고 있다. 토크나이저가 작동하는 방식이나 start token을 제거하는 것에 대한 설명도 깔끔하였다. 각 모델에서 key query value dimension에 대한 구체적인 설명이나 CustomSchedul 의 효과에 대해서도 추가적으로 언급하면 
더욱 좋은 코드가 될것 같습니다!!.

수고수고링이요!


## 1,2,3,4,5 pass (O) 
-----------------------------------------------------------------------
