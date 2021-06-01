from flask import Flask, url_for, render_template, Response, request, redirect, g, session
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import cv2
import socket
import io
import numpy as np
import os
import time
from datetime import datetime


app = Flask(__name__)

# model = load_model('model_data/yolo.h5')
# crop_img 엉키지 않게


class Microsecond(object):
    def __init__(self):
        dt = datetime.now()
        self.microsecond = dt.microsecond

    def get_path_name(self):
        return 'model/' + str(self.microsecond)


crop_img_origin_path = Microsecond()
default_img = cv2.imread('model/crop_img.jpg')

# 새로운 폴더 만들기!
try:
    if not(os.path.isdir(crop_img_origin_path.get_path_name())):
        os.makedirs(os.path.join(crop_img_origin_path.get_path_name()))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise


@app.before_request
def before_request():
    g.total_q = 10

# def gen():
#     """Video streaming generator function."""
#     while True:
#         rval, frame = vc.read()
#         cv2.imwrite('pic.jpg', frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + open('pic.jpg', 'rb').read() + b'\r\n')


def gen(camera):
    if not camera.isOpened():
        raise RuntimeError("Could not start camera")
    model = load_model('model_data/yolo.h5')

    while True:
        success, img = camera.read()
        if success:
            try:
                cv2.rectangle(img, (250, 250), (600, 600), (000, 51, 51), 2)

                crop_img = img[250:600, 250:600]
                crop_img_path = crop_img_origin_path.get_path_name() + '/crop_img.jpg'
                cv2.imwrite(crop_img_path, crop_img)
                # # print(crop_img)
                # result = model_predict()
                image = load_img(crop_img_path, target_size=(64, 64))
                image = img_to_array(image)
                image = image.reshape(
                    (1, image.shape[0], image.shape[1], image.shape[2]))

                prediction = model.predict(image)

                target_idx_for_predict = target_idx.get_idx()
                print("타겟예측: ", prediction[0][target_idx_for_predict])

                if np.argmax(prediction[0]) == 1:
                    result = get_label(np.argmax(prediction[0]))

                elif prediction[0][target_idx_for_predict] > 0:
                    result = get_label(target_idx_for_predict)
                else:
                    result = ''

                predict_label.set_label(result)

                ret, jpeg = cv2.imencode('.jpg', crop_img)
                frame = jpeg.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            except:
                print("An exception occurred")

        else:
            print("Status of camera.read()\n",
                  success, "\n=======================")


@app.route('/')  # http://127.0.0.1:5000
def index():
    # return 'c'
    # return render_template('base.html')
    return render_template('index.html')
# video streaming


@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')





def gen(camera):
    sess = tf.Session()

    with sess.as_default():

        while True:

            success, img = camera.read()

            if success:
                    results = tfnet.return_predict(img)


                    for result in results:
                        #tl = (result["topleft"]['x'], result['topleft']['y'])
                        #br = (result['bottomright']['x'], result['bottomright']['y'])
                        label = result["label"]
                        print(label)
                        cv2.rectangle(img,
                                    (result["topleft"]["x"], result["topleft"]["y"]),
                                    (result["bottomright"]["x"], result["bottomright"]["y"]),
                                    (255, 0, 0), 4)
                        text_x, text_y = result["topleft"]["x"] - 10, result["topleft"]["y"] - 10
                        cv2.putText(img, result["label"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('frame',img)

                    global tem_message
                    tem_message = label
                        
                        # log 체크
                    print("result: ", label, "| confidence: ", confidence)

                    #cv2.imshow('frame',img)

    
                    ret, jpeg = cv2.imencode('.jpg', img)
                    frame = jpeg.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                print("Status of camera.read()\n", success, img, "\n=======================")

@app.route('/video_feed')
def video_feed():
    cam = cv2.VideoCapture(0)
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    #if cam.isOpened():
    #    print('opended')
    return Response(gen(cam),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def webcam():
    return render_template('webcam.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)





if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)
