## A service that presents the optimal laundry course by detecting laundry contamination using YOLOv3

### 2021-1 Capstone Design2

> 2021.03 - 2021.06  

<br> 

**:book: Contents**
1. [Darknet](#1-darknet)
2. [Dataset](#2-dataset)
3. [Config](#3-config)
4. [Train](#4-train)
5. [Web](#5-web)

---
### 1. Darknet
* [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) 
  > mAP, loss, iteration 등을 추적하면서 연구하기 편해서 사용했다. 

### 2. Dataset
* [Google Open Images Dataset V6(Clothes, Button data)](https://storage.googleapis.com/openimages/web/index.html)
* [Tilda Textile Texture-Database(Stain data)](https://lmb.informatik.uni-freiburg.de/resources/datasets/tilda.en.html)
* [Kaggle(Stain data)](https://www.kaggle.com/priemshpathirana/fabric-stain-dataset)
* Google Image(Stain, Button data)
<br>

* [Yolo Mark](https://github.com/AlexeyAB/Yolo_mark)
  > Marking bounded boxes of objects in images for training Yolo v3   
<br>

### 3. Config
📂 **Darknet**  
┣ yolov3-spp_custom.cfg   
┃  
┣ obj.data  
┃ ┣ train.txt  
┃ ┗ valid.txt  
┃  
┗ obj.names  
<br>

### 4. Train
📂 **jupyterlab**   
┗ YOLO_MODEL.ipynb  
<br>

### 5. Web
```
python detect.py
```
---

### Reference
* [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) 

