아이펠캠퍼스 온라인4기 피어코드리뷰 []

- 코더 : 심재형
- 리뷰어 : 이성주

----------------------------------------------

**PRT(PeerReviewTemplate)**

** [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   
|평가문항|상세기준|평가|
|--------|--------|--------|
|1. Text recognition을 위해 특화된 데이터셋 구성이 체계적으로 진행되었다.|텍스트 이미지 리사이징, ctc loss 측정을 위한 라벨 인코딩, 배치처리 등이 적절히 수행되었다.|![image](https://github.com/horizon-sim/aiffel_project/assets/29011595/9067a32e-043b-424a-a1ae-a0ed73af2e37) 데이터셋 구성도 잘되어있고, 라벨 인코딩 및 디코딩이 잘되고 있음을 확인하였습니다.|
|2. CRNN 기반의 recognition 모델의 학습이 정상적으로 진행되었다.|학습결과 loss가 안정적으로 감소하고 대부분의 문자인식 추론 결과가 정확하다.|![image](https://github.com/horizon-sim/aiffel_project/assets/29011595/3cfc5f66-9fd6-453c-97ac-587dfa751931) loss가 줄어들며 안정적으로 감소하였습니다. |
|3. keras-ocr detector와 CRNN recognizer를 엮어 원본 이미지 입력으로부터 text가 출력되는 OCR이 End-to-End로 구성되었다.| 샘플 이미지를 원본으로 받아 OCR 수행 결과를 리턴하는 1개의 함수가 만들어졌다.| ![image](https://github.com/horizon-sim/aiffel_project/assets/29011595/2978f574-34fb-483e-a201-8ae096c81817) sample이미지를 받아서 ocr은 잘 진행 되었지만 한개의 함수는 아니지만 구분 구분 함수화를 잘 진행하였습니다. |

** [O] 주석을 보고 작성자의 코드가 이해되었나요?
![image](https://github.com/horizon-sim/aiffel_project/assets/29011595/5e220156-b075-406e-89a5-8829c5900144)
 - line 별 주석을 달아 코드 이해가 쉬웠습니다.

** [X] 코드가 에러를 유발할 가능성이 있나요?
 - 없습니다.
   
** [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
 - 잘 작성했다고 합니다.(feat. gpt4)

** [O] 코드가 간결한가요?
   ![image](https://github.com/horizon-sim/aiffel_project/assets/29011595/223a0e00-8c33-404a-8b8d-4e8d3d2a44aa)
 -  위와 같이 함수화를 통해 간결히 작성하였습니다.


----------------------------------------------

참고 링크 및 코드 개선
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
