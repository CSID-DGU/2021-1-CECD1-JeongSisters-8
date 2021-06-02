import numpy as np
from collections import Counter
import argparse
import time, math
import cv2
import os
from flask import Flask, request, Response, jsonify, render_template
import jsonpickle
#import binascii
import io as StringIO
import base64
from io import BytesIO
import io
import json
from PIL import Image
from flask_bootstrap import Bootstrap

# construct the argument parse and parse the arguments

confthres = 0.01
nmsthres = 0.1
yolo_path = './'


def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
    lpath = os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    return COLORS


def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath


def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath


def load_model(configpath, weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def image_to_byte_array(image: Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def get_predection(image, net, LABELS, COLORS):
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print("class"+str(classID))
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # ensure at least one detection exists
    labels = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten(): #다차원 배열(array)을 1차원 배열로 평평하게 펴주는 함수
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1]) #시작지점
            (w, h) = (boxes[i][2], boxes[i][3]) #가로 세로 길이

            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            color=[0,0,0]

            # w = math.ceil(w*1.8)
            # h = math.ceil(h*1.8)
            # if w >h:
            #     w = math.ceil(w*1.5)
            # else:
            #     h = math.ceil(h*1.5)
            # y *=2
            # get dimensions of image
            labels.append(LABELS[classIDs[i]])
            dimensions = image.shape
            
            # height, width, number of channels in image
            height = dimensions[0]
            width = dimensions[1]

            if x<10 or y<10:
                print(x,y)
                y = 25
            if height<h:
                h = height-2*y
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            text = "{}: {:.1f}".format(LABELS[classIDs[i]], confidences[i])
            if LABELS[classIDs[i]] == 'stain':
                #putText함수는 \n을 구분해내지 못하기 때문에 수동적으로 처리해야한다. 
                stain_size = w*h
                # print(stain_size)
                cv2.putText(image, str(stain_size), (x + w, y + h), #이미지, 출력문자, 출력위치
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2) #폰트, 폰트크기, 색상, 두께
                
            else:
                if LABELS[classIDs[i]] != 'Button': #옷 종류들을 담는다. 
                    global_clothes.append(LABELS[classIDs[i]])
                # print(boxes)
                print(LABELS[classIDs[i]],confidences[i])
                # print(classIDs)
                # print(confidences[i])
            
            
            # channels = img.shape[2]
            # print(width,height)
            cv2.putText(image, text, (x, y-5), #이미지, 출력문자, 출력위치
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2) #폰트, 폰트크기, 색상, 두께
    print(labels)
    return image,labels

# labelsPath = "/"
labelsPath = "data/obj.names"
cfgpath = "data/prior/yolov3-spp_custom.cfg"
wpath = "data/yolov3-spp_custom_best.weights"
Lables = get_labels(labelsPath)
CFG = get_config(cfgpath)
Weights = get_weights(wpath)
nets = load_model(CFG, Weights)
Colors = get_colors(Lables)
# Initialize the Flask application
app = Flask(__name__)
# Bootstrap(app)


# ajax 통신 변수
tem_message = "temporary"
final_message = "prediction result"
global_clothes = []
global_stains = [] 
const_sec = 30
# route http posts to this method
@app.route('/test_files', methods=['POST'])
def test_files():
    # load our input image and grab its spatial dimensions
    #image = cv2.imread("./test1.jpg")
    output = str(request.files["output"].read().decode("utf-8"))
    
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg = np.array(img)
    image = npimg.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res,labels = get_predection(image, nets, Lables, Colors)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # show the output image
    # cv2.imshow(request.files["output"].read(), image)
    # cv2.waitKey()
    # image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    os.chdir('./test_results')
    cv2.imwrite(str(output), image)
    os.chdir('../')
    np_img = Image.fromarray(image)
    img_encoded = image_to_byte_array(np_img)
    return Response(response=img_encoded, status=200, mimetype="image/jpeg")

@app.route('/api/test', methods=['POST'])
def main():
    # load our input image and grab its spatial dimensions
    #image = cv2.imread("./test1.jpg")

    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg = np.array(img)
    image = npimg.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res,labels = get_predection(image, nets, Lables, Colors)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # show the output image
    # cv2.imshow(request.files["output"].read(), image)
    # cv2.waitKey()
    image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    np_img = Image.fromarray(image)
    img_encoded = image_to_byte_array(np_img)
    return Response(response=img_encoded, status=200, mimetype="image/jpeg")

@app.route('/')
def index():
    """Video streaming ."""
    return render_template('index.html',resultReceived=sendResult())

@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    global tem_message
    """Video streaming generator function."""
    if not camera.isOpened():
        raise RuntimeError("Could not start camera")

    # vc = cv2.VideoCapture(0)
    count = 0
    i=0
    second = 60#15초
    while second >= 0:
        success, img = camera.read()
        # cv2.imshow("webcam",frame)
        # if success:
        pic_name = 'pic{}.jpg'.format(i)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res, labels = get_predection(image, nets, Lables, Colors)
        image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        # if second%const_sec==0: #1초마다 한 번씩
        cv2.imwrite(pic_name, image) #변환된 이미지나 동영상의 특정 프레임을 저장
        # cv2.imshow('output', image) #읽어들인 이미지 파일을 윈도우창에 보여줌
        tem_message = ', '.join(labels)
        # np_img = Image.fromarray(image)
        # img_encoded = image_to_byte_array(np_img)
        '''imwrite랑 yield가 같이 있어야 화면에서 보임.'''
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + open(pic_name, 'rb').read() + b'\r\n')
        # time.sleep(3)
        i+=1
        second -=1 
        
        if img is None:
            print("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break
    print("global clothes")
    print(global_clothes)
        
# ajax 통신 함수
@app.route("/sendResult")
def sendResult():
    global tem_message, final_message

    if tem_message == "temporary":
        final_message = "no prediction yet"

    else:
        final_message = tem_message

    return final_message

@app.route("/totalResult")
def totalResult():
    global global_clothes
    if not global_clothes:
        final_clothes = "?"

    else:
        clothes = Counter(global_clothes)
        final_clothes = clothes.most_common()[0][0]

    return final_clothes
@app.route('/test')
def test():
    """Video streaming ."""
    return render_template('test.html')

@app.route('/totalResult2')
def totalResult2():
    global global_clothes
    summaries = {
    "clothes": "Shirt",
    "Stain": 3,
    "area": "150"
    }
    
    str2 = jsonify(summaries)

    return str2

    # start flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
