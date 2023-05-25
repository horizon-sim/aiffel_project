### Aiffel_peer_review(5/25)
### Team : 심재형
### Reviewer : 김신성
-----------------------------------------------------------------------
## I review project 5 following above rules
- 1.Did the code work properly and fix the given issue?
- 2.Did I look at the comments and understand the author's code? And it is suitable?
- 3.Is there a possibility that the code will cause an error?
- 4.Did the code writer understand and write the code correctly?
- 5.Is the code concise and expandable?
- 6.etc
-----------------------------------------------------------------------
## Project 5
- Dataset : Given background and foreground images
- Problem : Image segmentation using given packages and merge them for several ways

### Data, Pre-trained weight load
```python
저장할 파일 이름을 결정합니다
# 1. os.getenv(x)함수는 환경 변수x의 값을 포함하는 문자열 변수를 반환합니다. model_dir 에 "/aiffel/human_segmentation/models" 저장
# 2. #os.path.join(a, b)는 경로를 병합하여 새 경로 생성 model_file 에 "/aiffel/aiffel/human_segmentation/models/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5" 저장
# 1
model_dir = os.getenv('HOME')+'/aiffel/human_segmentation/models' 
# 2
model_file = os.path.join(model_dir, 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5') 

# PixelLib가 제공하는 모델의 url입니다
model_url = 'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5' 

# 다운로드를 시작합니다
urllib.request.urlretrieve(model_url, model_file)
```
"It's easy now :)"

### Segmentation, Mask
```python
# output의 픽셀 별로 색상이 seg_color와 같다면 1(True), 다르다면 0(False)이 됩니다
# seg_color 값이 person을 값이 므로 사람이 있는 위치를 제외하고는 gray로 출력
# cmap 값을 변경하면 다른 색상으로 확인이 가능함
seg_map = np.all(output==seg_color, axis=-1) 
print(seg_map.shape) 
plt.imshow(seg_map, cmap='gray')
plt.show()

# 원본이미지를 img_show에 할당한뒤 이미지 사람이 있는 위치와 배경을 분리해서 표현한 color_mask 를 만든뒤 두 이미지를 합쳐서 출력
img_show = img_orig.copy()

# True과 False인 값을 각각 255과 0으로 바꿔줍니다
img_mask = seg_map.astype(np.uint8) * 255

# 255와 0을 적당한 색상으로 바꿔봅니다
color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)

# 원본 이미지와 마스트를 적당히 합쳐봅니다
# 0.6과 0.4는 두 이미지를 섞는 비율입니다.
img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.4, 0.0)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()
```
#### Outputs

![image](https://github.com/horizon-sim/aiffel_project/assets/91248817/542952df-8ac0-4191-865c-7dfbb1fffa59)
![image](https://github.com/horizon-sim/aiffel_project/assets/91248817/519bd365-32b9-4b37-a3c3-fb56b785c6ec)

"I've put footnotes to make it easier to understand how the Segmentaion function works, and I understand it well."
=> Good!

### Merge to Funtion
```python
def my_gausian_filter(img_path, model):
    img_orig = cv2.imread(img_path)
    real_orig_img = img_orig.copy()
    
    # segmentAsPascalvoc()함 수 를 호출 하여 입력된 이미지를 분할, 분할 출력의 배열을 가져옴, 분할 은 pacalvoc 데이터로 학습된 모델을 이용 
    segvalues, output = model.segmentAsPascalvoc(img_path)   

    #pascalvoc 데이터의 라벨종류
    LABEL_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ]
  ```
 "Overall, there's no big problem solving a given task, but wouldn't we be able to do more if we could make it more efficient to receive variables?"
 => Good!!
  
 "The middle part was very good There was no room for confusion, and the local variable name was written well"
=>Well done!
 
 ```
if "cat" in class_name:
        seg_color = (0,0,64)
    elif "dog" in class_name:
        seg_color = (128,0,64)
    elif "person" in class_name:
        seg_color = (128,128,192) 
```

"Rather than cat, dog, and person, it will be more versatile coding if it is divided into general variable names and corresponding cases."
=> Well done!

```python
img_path = os.getenv('HOME')+'/aiffel/data_karina_img/karina2.jpg' 
my_gausian_filter(img_path, model)
  ```
  
![image](https://github.com/horizon-sim/aiffel_project/assets/91248817/b45faa3f-c74b-4f9d-966d-a597c79d65d1)

"The visualization method that shows segmentation, mask, and output step by step is very good. I summarized the flow of the code"
=> Great!!

![image](https://github.com/horizon-sim/aiffel_project/assets/91248817/a1c9b923-fc59-4ea3-831e-94cab0290c12)
## 고양이가 매우 커엽습니다. 최고!


### Conclusion
Segmentation 

함수가 어떻게 작동하고 분류하는지, 마스크가 background 와 person을 어떻게 잘 분리할 수 있는지에 대해서 정확하게 알고있습니다.
마지막에 코드를 병합하여 짤 때 조금더 범용성 있게 짜면 다른 사진이나 추가적인 함수를 짤때 더 가변성있게 대처할 수 있으므로 조금더 범용성 있게 짜보는건 어떨까요?

오늘도 화이팅!

## 1,2,3,4pass (O) , 5 : Not bad ( △)
-----------------------------------------------------------------------
