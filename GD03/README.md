# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 심재형
- 리뷰어 : 이성주

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[⭕] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
|평가문항|상세기준|완료여부|
|-------|---------|--------|
| 1. CAM을 얻기 위한 기본모델의 구성과 학습이 정상 진행되었는가? |ResNet50 + GAP + DenseLayer 결합된 CAM 모델의 학습과정이 안정적으로 수렴하였다.|![image](https://github.com/horizon-sim/aiffel_project/assets/29011595/95365804-885d-4e79-963d-7e16fb248b48) 위와 같이 CAM 모델로 학습하여 학습과정이 안정적으로 수렴한것을 확인하였습니다.|
| 2. 분류근거를 설명 가능한 Class activation map을 얻을 수 있는가? | CAM 방식과 Grad-CAM 방식의 class activation map이 정상적으로 얻어지며, 시각화하였을 때 해당 object의 주요 특징 위치를 잘 반영한다.|![image](https://github.com/horizon-sim/aiffel_project/assets/29011595/a46dd2ef-a70e-4d78-8613-4047f21e734a) CAM방식의 activation map이 정상적인지 의문이지만 아래 박스를 그린것은 잘 그린것으로 보아 activation map이 잘 표현된것 같고, Grad-CAM 방식의 class activation map은 특징 위치를 잘 반영하였습니다.|
|3. 인식결과의 시각화 및 성능 분석을 적절히 수행하였는가? | CAM과 Grad-CAM 각각에 대해 원본이미지합성, 바운딩박스, IoU 계산 과정을 통해 CAM과 Grad-CAM의 object localization 성능이 비교분석되었다.| ![image](https://github.com/horizon-sim/aiffel_project/assets/29011595/79eec3e8-db00-46b1-889e-82e335ac7aa7) IOU가 CAM방식만 있고, Grad-CAM방식은 출력되지 않았습니다.|


### **[⭕] 주석을 보고 작성자의 코드가 이해되었나요?**
![image](https://github.com/horizon-sim/aiffel_project/assets/29011595/cda59acd-37f3-4307-abf5-614aac82833a)

 - 주석을 보고 코드가 이해하는 바가 무었인지 이해 했습니다.

### **[❌] 코드가 에러를 유발할 가능성이 있나요?**
 - 없습니다.
### **[⭕] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
 - 네 이해했습니다.
### **[⭕] 코드가 간결한가요?**
![image](https://github.com/horizon-sim/aiffel_project/assets/29011595/8ae90a55-27a5-4c7c-83f9-c702626b5a47)

- 위와 같이 callback 으로 checkpoint를 저장하여 잘 활용한 것 같습니다.
----------------------------------------------
### **참고 링크 및 코드 개선**
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
![image](https://github.com/horizon-sim/aiffel_project/assets/29011595/637e96c5-a7f9-4bc5-9360-2adc7fce2d30)
주석으로 장난금지..

----------------------------------------------
