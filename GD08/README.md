# 제목 : 

# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 심재형
- 리뷰어 : 이효준

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
> 정상적으로 동작하는 코드였습니다.
  
- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
```bash
!wget https://civitai.com/api/download/models/116417 -O lora_example.safetensors
```
> ``https://civitai.com/models/83636?modelVersionId=116417``에서 LoRA 모델 받고
```python
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "digiplay/hellofantasytime_v1.22"

unet = UNet2DConditionModel.from_pretrained("/content/diffusers_git/examples/dreambooth/data_1/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("/content/diffusers_git/examples/dreambooth/data_1/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
pipeline.to("cuda")

# load lora weight
# pipeline.load_lora_weights("./lora_example.safetensors")

# Perform inference, Stableor save, or push to the hub
#pipeline.save_pretrained("dreambooth-pipeline")
```  
>  사용자 Diffusion pipeline을 불러와 사용하는 방법 잘 보았습니다.

- [❌] 3.코드가 에러를 유발할 가능성이 있나요?
>  아니요

- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```python
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from PIL import Image

# Instantiate the Stable Diffusion model
model = keras_cv.models.StableDiffusion(jit_compile=True)

noise = tf.random.normal((512 // 8, 512 // 8, 4), seed=seed)

images = model.generate_image(
    interpolated_encodings,
    batch_size=interpolation_steps,
    diffusion_noise=noise,
)
```
![](https://camo.githubusercontent.com/48a05f234bd22378d97af41c6af54e95703866fc8a2c84aaface645127a620f3/68747470733a2f2f692e696d6775722e636f6d2f345a43785a59342e676966)
> 귀여운 골댕이와 맛있는 과일을 만드는 내용 재미있었습니다.

- [⭕] 5.코드가 간결한가요?
```python
export_as_gif("doggo-and-fruit-150.gif", images, rubber_band=True)
```
> StableDiffusion model이 생성한 이미지들을 움직이는 gif로 간편하게 만드는 코드가 추후에 다른 실험에서도 많은 도움이 될것 같아요!
