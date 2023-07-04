# 아이펠캠퍼스 온라인4기 피어코드리뷰[23.07.04]

- 코더 : 심재형
- 리뷰어 : 부석경

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[⭕] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
|평가문항|상세기준|완료여부|
|-------|---------|--------|
|1. KITTI 데이터셋에 대한 분석이 체계적으로 진행되었다.|KITTI 데이터셋 구조와 내용을 파악하고 이를 토대로 필요한 데이터셋 가공을 정상 진행하였다.| X |
|2. RetinaNet 학습이 정상적으로 진행되어 object detection 결과의 시각화까지 진행되었다.|바운딩박스가 정확히 표시된 시각화된 이미지를 생성하였다.| O |
|3. 자율주행 Object Detection 테스트시스템 적용결과 만족스러운 정확도 성능을 달성하였다.|테스트 수행결과 90% 이상의 정확도를 보였다.| O |

```python
.
.
.
Epoch 00007: saving model to /aiffel/aiffel/object_detection/data/check/weights_epoch_7
Epoch 8/10
3173/3173 [==============================] - 1617s 508ms/step - loss: 0.2652 - accuracy: 0.2015 - val_loss: 1.0345 - val_accuracy: 0.1861

Epoch 00008: saving model to /aiffel/aiffel/object_detection/data/check/weights_epoch_8
Epoch 9/10
1781/3173 [===============>..............] - ETA: 11:27 - loss: 0.2399 - accuracy: 0.2058
중간에 커널이 꺼지는바람에 history정보가 전부 소실되었다. 어쩔수없다 일단 8 epoch까지만 구현하자
```
위와 같이 학습했습니다.

![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/71332005/23918112-6614-4cc0-ad48-c3e16b65cd24)

### **[⭕] 주석을 보고 작성자의 코드가 이해되었나요?**
```python
def visualize_bbox(input_image, object_bbox):
    input_image = copy.deepcopy(input_image)  # 입력 이미지의 복사본을 만듭니다.
    draw = ImageDraw.Draw(input_image)  # 입력 이미지에 대한 ImageDraw 객체를 생성합니다.
    
    # 바운딩 박스 좌표(x_min, x_max, y_min, y_max) 구하기
    width, height = img.size  # 이미지의 너비와 높이를 가져옵니다.
    x_min = object_bbox[:,1] * width  # 객체 바운딩 박스의 x_min 좌표를 계산합니다.
    x_max = object_bbox[:,3] * width  # 객체 바운딩 박스의 x_max 좌표를 계산합니다.
    y_min = height - object_bbox[:,0] * height  # 객체 바운딩 박스의 y_min 좌표를 계산합니다.
    y_max = height - object_bbox[:,2] * height  # 객체 바운딩 박스의 y_max 좌표를 계산합니다.
    
    # 바운딩 박스 그리기
    rects = np.stack([x_min, y_min, x_max, y_max], axis=1)  # 바운딩 박스의 좌표를 하나의 배열로 묶습니다.
    for _rect in rects:  # 각각의 바운딩 박스에 대해 반복합니다.
        draw.rectangle(_rect, outline=(255,0,0), width=2)  # 바운딩 박스를 그립니다.

    return input_image

visualize_bbox(img, objects['bbox'].numpy())
```

### **[❌] 코드가 에러를 유발할 가능성이 있나요?**
찾지 못했습니다.

### **[⭕] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
네, Anchor와 bounding box에 대해 질문했습니다. 잘 알고 답해주었습니다.


### **[⭕] 코드가 간결한가요?**
원하는 레이어의 CAM을 모두 모아 출력하는 코드가 간편하고 좋아보입니다.



----------------------------------------------
### **참고 링크 및 코드 개선**
----------------------------------------------

1. 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
2. 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
