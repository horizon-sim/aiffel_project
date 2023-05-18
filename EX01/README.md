### Aiffel_peer_review(5/18)
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
## Project 1
- Dataset : diabetes Datset
- Problem : Linear regression for Given Target data and Visualization for compare between predict and Ground truth

```python
import pandas as pd
from sklearn.datasets import load_diabetes
df = load_diabetes()
df
# 어느하나 뺄게 없이 모두 중요하다
```
"Wouldn't it be nice to analyze features through basic EDA and choose more effective features for learning?"

```python
# (7) 기울기를 구하는 gradient 함수 구현하기

def gradient(x, w, b, y):
    # N은 데이터 포인트의 개수
    N = len(y)
    
    # y_pred 준비
    y_pred = model(x, w, b)
    
    # 공식에 맞게 gradient 계산
    dw = 1/N * 2 * x.T.dot(y_pred - y)
        
    # b의 gradient 계산
    db = 2 * (y_pred - y).mean()
    return dw, db
```
"The notes described in between codes are very legible => Good!"
```python
# (9) 모델 학습하기

w = np.random.rand(10)
b = np.random.rand()

w,b
```
"It would be easier to read if we explain the size when we initialize W?"
```python
losses = []

for i in range(1, 5001):
    dw, db = gradient(X_train, w, b, y_train)
    w -= LEARNING_RATE * dw
    b -= LEARNING_RATE * db
    L = loss(X_train, w, b, y_train)
    losses.append(L)
    if i % 100 == 0:
        print('Iteration %d : Loss %0.4f, ' % (i, L))
        prediction = model(X_test, w, b)
        mse = MSE(prediction, y_test)
        print(mse)
  ```
"Simple and concise => Looks like good"
"Wouldn't it be better to implement it using Class so that you can adjust lr or epoch when you want?"
```python
# (11) 정답 데이터와 예측한 데이터 시각화하기
# plt.plot(prediction, c="r")

plt.scatter(X_test[:, 0], y_test)
plt.scatter(X_test[:, 0], prediction)
plt.show()
```
- Visualization outputs


![image](https://github.com/horizon-sim/aiffel_project/assets/91248817/ecfa1e82-89f4-439b-9f23-ab375aaf935d)

"How about plt.legend() method for better understaning meaing of each point's class?"

## Conclusion
전체적으로 코드가 어떻게 진행되는지에 대해서 잘 알고 있다. 각 step에서 적절한 설명을 같이 첨부하여 코드를 실행하기 전에 어떠한 논리로 해당 코드를 짯는지 이해하기 용이했다.
각 Feature들의 추가적인 EDA를 통해서 어떤 feature들이 더 좋은지 feature selection을 같이 한다면 더욱 좋을 것 같다.
Visualization도 물론 잘 하였지만 plt.legend() 를 통해서 실제 구한 prediction point가 무엇이고 Ground truth가 무었인지 보여주는 표를 추가하면 좋겠습니다.

## 1,2,3,4,5 pass (O)
 
-----------------------------------------------------------------------
## Project 2
- Dataset : Bike Datset
- Problem : Linear regression for Given Target data and Visualization for compare between predict and Ground truth on given feature
```python
# (4) X, y train/test
# casual, registered 컬럼은 이미 제거했고 분 초도 계속 0으로 나오니 그냥 제거하도록 하겠습니다.
df.drop(["datetime", "minute", "second"], axis=1, inplace=True)
df
```
"Inplace= True로 쓰는 방법이 조금 더 간결하하고 좋은것 같습니다. Good!"
```python
# (6) 학습된 모델로 X_test에 대한 예측값 출력 및 손실함수값 계산
from sklearn.metrics import mean_squared_error

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mse
rmse = np.sqrt(mse)

print(f"MSE : {mse}")
print(f"RMSE : {rmse}")
```
"각주 설명도 적절했고 무엇을 구하고 평가하는지에 대해서 잘 알고 있습니다."

## 1,2,3,4,5 pass(O)
