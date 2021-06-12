darknet/data/obj.data
darknet/data/obj.name
darknet/data/train.txt
darknet/data/validation.txt

darknet/cfg/yolov3_custom.cfg
darknet/backup/yolov3_custom_last.weight
darknet/backup/yolov3_custom_best.weight

data/train // train dataset - class(8) : Jacket / Shirt / Trousers / Skirt / Swimwear / Towel / stain / Button

generate_train.py // train.txt 생성
convert.py  // label class id 변경
make_validation.ipynb  // train:validation = 8:2로 나눔. train_suffle - 나누기 전 suffle된 상태