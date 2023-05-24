### Aiffel_peer_review(5/23)
### Team : 심재형
### Reviewer : 김신성
-----------------------------------------------------------------------
## I review project 3 following above rules
- 1.Did the code work properly and fix the given issue?
- 2.Did I look at the comments and understand the author's code? And it is suitable?
- 3.Is there a possibility that the code will cause an error?
- 4.Did the code writer understand and write the code correctly?
- 5.Is the code concise and expandable?
- 6.etc
-----------------------------------------------------------------------
## Project 3
- Dataset : Given image and sticker
- Problem : Apply landmark points for given Image dataset and merge with sticker. A quick analysis of why rotating or changing pictures doesn't work well

### Data load, face box
```python
my_image_path = './data/karina.jpg'
img_bgr = cv2.imread(my_image_path)    # OpenCV로 이미지를 불러옵니다
img_show = img_bgr.copy()      # 출력용 이미지를 따로 보관합니다
plt.imshow(img_bgr) # 이미지를 출력하기 위해 출력할 이미지를 올려준다. (실제 출력은 하지 않음)
plt.show()

# 얼굴 감지기
detector_hog = dlib.get_frontal_face_detector()

# 이미지 피라미드(작게 촬영된걸 크게 보기위함)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
dlib_rects = detector_hog(img_rgb, 1)  
print(dlib_rects)

# 찾은 얼굴 영역 박스로 감싸기
# 리스트에 감싸져있으니 for문으로 풀어준다.
for dlib_rect in dlib_rects: # 찾은 얼굴 영역의 좌표
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA) # 시작점의 좌표와 종료점 좌표로 직각 사각형을 그림

img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()
```
"Image data is well loaded and the detecting box is well expressed" => Good!!

```python
# landmark 지정
model_path = './data/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)
    # dlib 라이브러리의 shape_predictor 함수를 이용하여 모델을 불러옴
    # landmark_predictor는 RGB이미지와 dlib.rectangle를 입력 받고 dlib.full_object_detection를 반환
    # dlib.rectangle: 내부를 예측하는 박스
    # dlib.full_object_detection: 각 구성 요소의 위치와, 이미지 상의 객체의 위치를 나타냄

# 랜드마크 위치 저장
list_landmarks = []

# 얼굴 영역 박스 마다 face landmark를 찾아냅니다
# face landmark 좌표를 저장해둡니다
for dlib_rect in dlib_rects:
    
    # 모든 landmark의 위치정보를 points 변수에 저장
    points = landmark_predictor(img_rgb, dlib_rect)
    
    # 각각의 landmark 위치정보를 (x,y) 형태로 변환하여 list_points 리스트로 저장
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    
    # list_landmarks에 랜드마크 리스트를 저장
    list_landmarks.append(list_points)

print(len(list_landmarks[0]))
    # 얼굴이 n개인 경우 list_landmarks는 n개의 원소를 갖고
    # 각 원소는 68개의 랜드마크 위치가 나열된 list 
    # list_landmarks의 원소가 1개이므로 list_landmarks[1]을 호출하면 IndexError가 발생
```
"Found a landmark for a given image and each line was written legibly, including detailed descriptions."

```python
# 이미지가 정확히 어디 들어가는지 확인
cv2.rectangle(img_show, (refined_x, refined_y), (refined_x+img_sticker.shape[0], refined_y+img_sticker.shape[1]), (0,0,255), 2, lineType=cv2.LINE_AA) # 시작점의 좌표와 종료점 좌표로 직각 사각형을 그림
img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()
  ```

![image](https://github.com/horizon-sim/aiffel_project/assets/91248817/4d6a7e24-4544-4fbc-b108-58f6a15f73c1)

"It is expressed by visualizing exactly how much the size of the reset sticker occupies in the given image data" => Verg good!!

```python
sticker_area = img_bgr[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x + img_sticker.shape[1]]
img_bgr[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x + img_sticker.shape[1]] = cv2.addWeighted(sticker_area, 0.5, np.where(img_sticker==0, img_sticker, sticker_area).astype(np.uint8), 0.5, 0)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.show()
  ```
"cv2.addWeighted(sticker_area, 0.5, np.where(img_sticker==0, img_sticker, sticker_area).astype(np.uint8), 0.5, 0)"
addWeighted 함수를 이용하여 sticker 의 투명도를 잘 조절 하였다. 

```python
my_image_catpilter("./data/karina2.jpg", "./data/cat-whiskers.png", "./data/shape_predictor_68_face_landmarks.dat")
```
![image](https://github.com/horizon-sim/aiffel_project/assets/91248817/954059cf-70b0-424a-b74e-90cb3aed3c97)

"얼굴 각도가 틀어져있는 사진에 대한 스티커 적용에 대한 예시를 잘 시각화 시켰다."

 이에 대한 해석과 어떻게 하면 잘 적용할 수 있는지에 대한 해결 코드도 같이 있으면 더욱 좋을 것 같습니다.
 
 여러가지 사진을 가지고 적용해보는 등, 다양한 사진에 대해 적용해보는 실습도 정말 좋았습니다.
 
```python
import seaborn as sns

def my_image_catpilter(my_img_path, cat_pilter_path, model_path):
    img_bgr = cv2.imread(my_img_path)
    img_show = img_bgr.copy()
    
    # 얼굴 감지 부분
    detector_hog = dlib.get_frontal_face_detector()
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    dlib_rects = detector_hog(img_rgb, 1)

    # 찾은 얼굴 영역 박스로 감싸기
    for dlib_rect in dlib_rects:
        l = dlib_rect.left()
        t = dlib_rect.top()
        r = dlib_rect.right()
        b = dlib_rect.bottom()

        cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

    # img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    
    
    landmark_predictor = dlib.shape_predictor(model_path)
    list_landmarks = []

    for dlib_rect in dlib_rects:
        
        points = landmark_predictor(img_rgb, dlib_rect)
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        list_landmarks.append(list_points)
    for landmark in list_landmarks:
        for point in landmark:
            cv2.circle(img_show, point, 2, (0, 255, 255), -1)

    # img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    
    for dlib_rect, landmark in zip(dlib_rects, list_landmarks): # 얼굴 영역을 저장하고 있는 값과 68개의 랜드마크를 저장하고 있는 값으로 반복문 실행
        x = landmark[34][0] # 코 부위의 x값
        y = landmark[34][1] # 코 부위의 y값
        w = h = dlib_rect.width()
        
    # 고양이 수염을 얼굴 크기에 맞게 리사이즈
    img_sticker = cv2.imread(cat_pilter_path)
    img_sticker = cv2.resize(img_sticker, (w,h))
    
    refined_x = x - w//2
    refined_y = y - h//2
    print (f'(x,y) : ({refined_x},{refined_y})')
    
    # 음수일 경우
    if refined_x < 0: 
        img_sticker = img_sticker[:, -refined_x:]
        refined_x = 0
    if refined_y < 0:
        img_sticker = img_sticker[-refined_y:, :]
        refined_y = 0
        
    # 이미지가 정확히 어디 들어가는지 확인
    
    cv2.rectangle(img_show, (refined_x, refined_y), (refined_x+img_sticker.shape[0], refined_y+img_sticker.shape[1]), (0,0,255), 2, lineType=cv2.LINE_AA) # 시작점의 좌표와 종료점 좌표로 직각 사각형을 그림
    img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.imshow(img_show_rgb)
    
    # 완성작품을보자
    sticker_area = img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
    img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = np.where(img_sticker==255, sticker_area, img_sticker).astype(np.uint8)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.show()
 ```
전체적인 실습 구현을 한번에 적용해 볼 수 있게 함수 형태로 만들어 번거로운 사용 없이 간결하게 코딩해놓았습니다. Very good!!

About step5

각 질문들에 대한 문제점과 해결방안에 대한 자료를 충분히 공부하여 보완한다면 앞으로 이미지 데이터를 다루는것에 있어 더욱 많은 관점을 담은 좋은 Small project가 될 것이라고 생각합니다.

크 역시 낭만코딩

## 1,2,3,4,5 pass (O)
 
-----------------------------------------------------------------------
