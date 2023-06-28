# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 심재형
- 리뷰어 : 김창완


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [o] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

  - 코드는 정상적으로 전부 동작 했습니다

    ```python
    plt.plot(history_resnet50_no_aug.history['val_accuracy'], 'r')
    plt.plot(history_resnet50_aug.history['val_accuracy'], 'b')
    plt.plot(history_resnet50_cutmix.history['val_accuracy'], 'g')
    plt.title('Model validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['No Augmentation', 'With Augmentation', "With Cutmix"], loc='upper left')
    plt.show()
    ```

    위 코드의 결과가 

    ![image-20230628170533286](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230628170533286.png)

    로 잘 나왔습니다

- [o] 2.주석을 보고 작성자의 코드가 이해되었나요?
  
  - 네 특히 함수 지정 부분에서의 코드가 잘 이해되었습니다
  
    ```python
    # mix two images
    def mix_2_images(image_a, image_b, x_min, y_min, x_max, y_max):
        image_size_x = image_a.shape[1]
        image_size_y = image_a.shape[0]
        middle_left = image_a[y_min:y_max, 0:x_min, :] # image_b의 왼쪽 바깥 영역
        middle_center = image_b[y_min:y_max, x_min:x_max, :]  # image_b의 안쪽 영역
        middle_right = image_a[y_min:y_max, x_max:image_size_x, :] # image_b의 오른쪽 바깥 영역
        middle = tf.concat([middle_left,middle_center,middle_right], axis=1)
        top = image_a[0:y_min, :, :]
        bottom = image_a[y_max:image_size_y, :, :]
        mixed_img = tf.concat([top, middle, bottom],axis=0)
    
        return mixed_img
    ```
  
    

- [ ] 3.코드가 에러를 유발할 가능성이 있나요?
  - 에러를 유발할 부분은 없어 보입니다.


- [o] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  - 네 한 20분 붙잡고 전부 물어 봤는데 잘 대답해주셨습니다

- [o] 5.코드가 간결한가요?
  - 코드는 간결합니다 제가 더 보완해야 할 곳을 보지는 못했습니다



### Suggestion

- Optimizer를 SGD대신 ADAM을 써보는건 어땠을까 생각합니다.
- 아무래도 메모리 한계때문에 Epoch을 3번밖에 못돌린건 아쉽긴 합니다 그래도 그래프에서 곡선이 가장 가파른게 cutmix이다 보니 어느정도의 성과는 확실히 거둔것 같습니다
