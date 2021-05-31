import numpy as np
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
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
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
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            if LABELS[classIDs[i]] == 'stain':
                #putText함수는 \n을 구분해내지 못하기 때문에 수동적으로 처리해야한다. 
                stain_size = "\n{}*{} = {}".format(w,h,w*h)
                # print(stain_size)
                text += stain_size
                print(text)
            else:
                # print(boxes)
                print(LABELS[classIDs[i]],confidences[i])
                # print(classIDs)
                # print(confidences[i])
            # get dimensions of image
            dimensions = image.shape
            
            # height, width, number of channels in image
            height = dimensions[0]
            width = dimensions[1]
            # channels = img.shape[2]
            # print(width,height)
            cv2.putText(image, text, (x, y-5), #이미지, 출력문자, 출력위치
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) #폰트, 폰트크기, 색상, 두께
    return image

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
    res = get_predection(image, nets, Lables, Colors)
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
    res = get_predection(image, nets, Lables, Colors)
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
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen():
    """Video streaming generator function."""
    # while True:
    #     rval, frame = vc.read()
    #     cv2.imwrite('pic.jpg', frame)
    #     yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + open('pic.jpg', 'rb').read() + b'\r\n')

    vc = cv2.VideoCapture(0)
    count = 0
    i=0
    while True:
        rval, img = vc.read()
        # cv2.imshow("webcam",frame)
        pic_name = 'pic{}.jpg'.format(i)


        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = get_predection(image, nets, Lables, Colors)
        image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        cv2.imwrite(pic_name, image)
        cv2.imshow('output', image)
        # np_img = Image.fromarray(image)
        # img_encoded = image_to_byte_array(np_img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open(pic_name, 'rb').read() + b'\r\n')
        time.sleep(3)
        i+=1
        if img is None:
            print("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break
        

    # start flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
