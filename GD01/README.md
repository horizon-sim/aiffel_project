# Code Peer Review
- 코더 : 심재형
- 리뷰어 : 이동익


- [🔺] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  -정상적으로 동작합니다
  - 학습이 완료되어 결과를 비교하지 못한 점이 있으나, 데이터 전처리와 모델 설계, 학습이 잘 진행되었습니다.
- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
  -각 코드별로 설명이 필요한 부분에 주석이 잘 들어가있었습니다
- [❌] 3.코드가 에러를 유발한 가능성이 있나요?
- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
- [⭕] 5.코드가 간결한가요?
  -간결합니다

# 예시
```python
def build_resnet_block(input_x, num_cnn=3, channel=64, block_num=1,is_50 = False,is_plain = False)

def build_resnet(input_shape,
              num_cnn_list=[3,4,6,3],
              channel_list=[64,128,256,512]

    ####중략####
                 
    # config list들의 길이만큼 반복해서 블록을 생성합니다.
    for i, (num_cnn, channel) in enumerate(zip(num_cnn_list, channel_list)):
        output = build_resnet_block(
            output,
            num_cnn=num_cnn, 
            channel=channel,
            block_num=i,
            is_50 = is_50,
            is_plain = is_plain
        )   
```
이런식으로 레이어블록을 만드는 함수를 통해 코드를 간결하게 적어주신 부분이 좋았습니다.

'''python
ResNet_34 = build_resnet((IMAGE_SIZE, IMAGE_SIZE, 3), is_50=False, is_plain=False)
ResNet_50 = build_resnet((IMAGE_SIZE, IMAGE_SIZE, 3), is_50=True, is_plain=False)
Plain_34 = build_resnet((IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=2, is_50=False, is_plain=True)
Plain_50 = build_resnet((IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=2, is_50=True, is_plain=True)
'''
이런식으로 모델을 간편히 가져올 수 있다는 점을 배웠습니다.

'''
137/547 [======>.......................] - ETA: 1:10 - loss: 2.3804 - accuracy: 0.4854
Corrupt JPEG data: 65 extraneous bytes before marker 0xd9
154/547 [=======>......................] - ETA: 1:07 - loss: 2.2012 - accuracy: 0.4872
Corrupt JPEG data: 228 extraneous bytes before marker 0xd9
202/547 [==========>...................] - ETA: 58s - loss: 1.8631 - accuracy: 0.4915
Corrupt JPEG data: 396 extraneous bytes before marker 0xd9
206/547 [==========>...................] - ETA: 57s - loss: 1.8412 - accuracy: 0.4917
'''
시간이 부족해 모델을 미처 훈련하지 못한 점이 아쉽지만 훈련이 진행되는 것을 확인할 수 있었습니다.
