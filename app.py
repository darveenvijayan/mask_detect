import dash
import dash_core_components as dcc
import dash_html_components as html

from flask import Flask, Response
import cv2

from pathlib import Path

import click
import cv2
import torch

from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

# from common.facedetector import FaceDetector
from train import MaskDetector



model = MaskDetector()
model.load_state_dict(torch.load('/media/darveen/DATADRIVE1/Projects/KLCCUH/MaskDetection/covid-mask-detector/covid-mask-detector_dash/models/face_mask.ckpt')['state_dict'], strict=False)
# print('model loaded')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# faceDetector = FaceDetector(
#     prototype='models/deploy.prototxt.txt',
#     model='models/res10_300x300_ssd_iter_140000.caffemodel',
# )

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


transformations = Compose([
    ToPILImage(),
    Resize((100, 100)),
    ToTensor(),
])


font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.namedWindow('main', cv2.WINDOW_NORMAL)
labels = ['No Mask', 'Mask']
labelColor = [(10, 0, 255), (10, 255, 0)]






class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):

        success, image = self.video.read()

        if success is True:
            jpeg = image



#             faces = faceDetector.detect(jpeg)
            faces = face_cascade.detectMultiScale(jpeg, 1.1, 4)

            for face in faces:
                xStart, yStart, width, height = face
                
                # clamp coordinates that are outside of the image
                xStart, yStart = max(xStart, 0), max(yStart, 0)
                
                # predict mask label on extracted face
                faceImg = jpeg[yStart:yStart+height, xStart:xStart+width]
                output = model(transformations(faceImg).unsqueeze(0).to(device))
                _, predicted = torch.max(output.data, 1)
                
                # draw face frame
                cv2.rectangle(jpeg,
                            (xStart, yStart),
                            (xStart + width, yStart + height),
                            (126, 65, 64),
                            thickness=2)
                
                # center text according to the face frame
                textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
                textX = xStart + width // 2 - textSize[0] // 2
                
                # draw prediction label
                cv2.putText(jpeg,
                            labels[predicted],
                            (textX, yStart-20),
                            font, 1, labelColor[predicted], 2) 

                ret, jpeg = cv2.imencode('.jpg', jpeg)

            else:
                ret, jpeg = cv2.imencode('.jpg', image)


        return jpeg.tobytes()


def gen(camera):
    # modelpath = '/media/darveen/DATADRIVE1/Projects/KLCCUH/MaskDetection/covid-mask-detector/covid-mask-detector_dash/models/face_mask.ckpt'
    # model = MaskDetector()
    # model.load_state_dict(torch.load('/media/darveen/DATADRIVE1/Projects/KLCCUH/MaskDetection/covid-mask-detector/covid-mask-detector_dash/models/face_mask.ckpt')['state_dict'], strict=False)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model.eval()

    # faceDetector = FaceDetector(
    #     prototype='covid-mask-detector/models/deploy.prototxt.txt',
    #     model='covid-mask-detector/models/res10_300x300_ssd_iter_140000.caffemodel',
    # )

    # transformations = Compose([
    #     ToPILImage(),
    #     Resize((100, 100)),
    #     ToTensor(),
    # ])


    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    # labels = ['No Mask', 'Mask']
    # labelColor = [(10, 0, 255), (10, 255, 0)]


    while True:
        frame = camera.get_frame()

        # faces = faceDetector.detect(frame)
        # for face in faces:
        #     xStart, yStart, width, height = face
            
        #     # clamp coordinates that are outside of the image
        #     xStart, yStart = max(xStart, 0), max(yStart, 0)
            
        #     # predict mask label on extracted face
        #     faceImg = frame[yStart:yStart+height, xStart:xStart+width]
        #     output = model(transformations(faceImg).unsqueeze(0).to(device))
        #     _, predicted = torch.max(output.data, 1)
            
        #     # draw face frame
        #     cv2.rectangle(frame,
        #                   (xStart, yStart),
        #                   (xStart + width, yStart + height),
        #                   (126, 65, 64),
        #                   thickness=2)
            
        #     # center text according to the face frame
        #     textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
        #     textX = xStart + width // 2 - textSize[0] // 2
            
        #     # draw prediction label
        #     cv2.putText(frame,
        #                 labels[predicted],
        #                 (textX, yStart-20),
        #                 font, 1, labelColor[predicted], 2) 


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# app = dash.Dash(__name__)
# server = app.server
app.title='Dashboard'

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div([
    html.H1("Webcam Test"),
    html.Img(src="/video_feed")
])

if __name__ == '__main__':
    app.run_server()
