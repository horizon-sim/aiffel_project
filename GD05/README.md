# 아이펠캠퍼스 온라인4기 피어코드리뷰[23.07.06]

- 코더 : 심재형
- 리뷰어 : 부석경

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[⭕] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
|평가문항|상세기준|완료여부|
|-------|---------|--------|
|1. U-Net을 통한 세그멘테이션 작업이 정상적으로 진행되었는가?|	KITTI 데이터셋 구성, U-Net 모델 훈련, 결과물 시각화의 한 사이클이 정상 수행되어 세그멘테이션 결과 이미지를 제출하였다.| ![image](https://github.com/horizon-sim/aiffel_project/assets/71332005/c5a8d69b-25eb-490d-a1bd-3f7e060b25f6)|
|2. U-Net++ 모델이 성공적으로 구현되었는가?|	U-Net++ 모델을 스스로 구현하여 학습 진행 후 세그멘테이션 결과까지 정상 진행되었다.| ![image](https://github.com/horizon-sim/aiffel_project/assets/71332005/4e531536-b91e-426c-ad99-9184a0d443c4)
|3. U-Net과 U-Net++ 두 모델의 성능이 정량적/정성적으로 잘 비교되었는가?|U-Net++ 의 세그멘테이션 결과 사진과 IoU 계산치를 U-Net과 비교하여 우월함을 확인하였다.| ![image](https://github.com/horizon-sim/aiffel_project/assets/71332005/a708a6ab-2a52-43e9-b86e-0eb56d6a0dbc) |

### **[⭕] 주석을 보고 작성자의 코드가 이해되었나요?**
```python
def build_model(input_shape=(224, 224, 3)):
    # 입력 레이어
    inputs = Input(input_shape)

    # 첫 번째 Contracting Path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 두 번째 Contracting Path
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 중간 Convolutional Layer
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # 첫 번째 Expanding Path
    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = Conv2D(128, 2, activation='relu', padding='same')(up1)
    merge1 = concatenate([conv2, up1], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    # 두 번째 Expanding Path
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = Conv2D(64, 2, activation='relu', padding='same')(up2)
    merge2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # 출력 레이어
    output = Conv2D(1, 1, activation='sigmoid')(conv5)

    # 모델 생성
    model = Model(inputs=inputs, outputs=output)

    return model
```
순서에 따라 주석을 달아주셨습니다.  

### **[❌] 코드가 에러를 유발할 가능성이 있나요?**   
* 찾지 못했습니다.

### **[⭕] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)   
* 전체적으로 층 하나를 노드보다 줄였습니다. 그에 따라 Pameter도 줄어들고 하였습니다. 그러나 제가 보기에 U-Net++의 잔차나 deep supervision이 빠져있습니다.
```python
def conv_block(inputs, filters, kernel_size=3):
    x = Conv2D(filters, kernel_size, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x

def build_model(input_shape=(224, 224, 3)):
    # 입력 레이어
    inputs = Input(input_shape)

    # 첫 번째 Contracting Path
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 두 번째 Contracting Path
    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 세 번째 Contracting Path
    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 네 번째 Contracting Path
    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Expanding Path
    up4 = UpSampling2D(size=(2, 2))(pool4)
    up4 = Conv2D(256, 2, activation='relu', padding='same')(up4)
    merge4 = concatenate([conv4, up4], axis=3)
    conv5 = conv_block(merge4, 512)

    up3 = UpSampling2D(size=(2, 2))(conv5)
    up3 = Conv2D(128, 2, activation='relu', padding='same')(up3)
    merge3 = concatenate([conv3, up3], axis=3)
    conv6 = conv_block(merge3, 256)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    up2 = Conv2D(64, 2, activation='relu', padding='same')(up2)
    merge2 = concatenate([conv2, up2], axis=3)
    conv7 = conv_block(merge2, 128)

    up1 = UpSampling2D(size=(2, 2))(conv7)
    up1 = Conv2D(32, 2, activation='relu', padding='same')(up1)
    merge1 = concatenate([conv1, up1], axis=3)
    conv8 = conv_block(merge1, 64)

    # 출력 레이어
    output = Conv2D(1, 1, activation='sigmoid')(conv8)

    # 모델 생성
    model = Model(inputs=inputs, outputs=output)

    return model
```
### **[⭕] 코드가 간결한가요?**   
* 네 불필요한 코드는 없어 보입니다.
----------------------------------------------
### **참고 링크 및 코드 개선**


----------------------------------------------



