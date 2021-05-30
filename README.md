# 딥러닝(YOLO) 기반 세탁물을 분류/예측하고 분류된 세탁물에 적합한 세탁 정보를 제공하는 인공지능 세탁기

### 2021-1 Capstone Design2 종합 설계 Repository 입니다.
> :star: YOLOv3 이용

#
**:book: Contents**
1. [Darknet](#1-darknet)
2. [Dataset](#2-dataset)
3. [Config](#3-config)
4. [Train](#4-train)


---
## 1. Darknet
* [AlexeyAB 의 darknet](https://github.com/AlexeyAB/darknet) 
  * mAP, loss, iteration 등을 추적하면서 연구하기 편해서 사용했다. 

## 2. Dataset
* Google Open Images Dataset V6(Clothes, Button data)
* Tilda(Stain data)
* Kaggle(Stain data)
* Google 이미지(Stain, Button data)

[Yolo Mark](https://github.com/AlexeyAB/Yolo_mark)를 이용해 라벨링

## 3. Config
* obj.cfg 
* obj.data
  * train.txt
  * valid.txt
* obj.names

## 4. Train

---

# Reference
* [jojoldu님의 junior-recruit-scheduler](https://github.com/jojoldu/junior-recruit-scheduler/blob/master/README.md)
