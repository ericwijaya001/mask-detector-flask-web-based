import os
import cv2
import imutils
import time
# import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
from flask import Flask, render_template, request, send_from_directory, Response, redirect, url_for

from camera import VideoCamera


app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
CLASSIFY_FOLDER = "templates\classified_img"
CLASSIFIED_FOLDER = "classified_img"

print("[INFO] loading face detector model...")
prototxtPath = r'static\face_detector\deploy.prototxt'
weightsPath = r'static\face_detector\res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
model = load_model('static\mask_detector.model')

# buat dewe
def classify(model, img_path,file):
    image = cv2.imread(img_path)
    orig = image.copy()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))
    
    print("[INFO] computing face detections...")
    faceNet.setInput(blob)
    detections = faceNet.forward()

    label = ''
    prob = ''

    # keterangan kek di camera.py
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > 0.5 :
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]

            label = "Hasil Prediksi = Bermasker" if mask > withoutMask else "Hasil Prediksi = Tidak Bermasker"
            color = (0, 255, 0) if label == "Hasil Prediksi = Bermasker" else (0, 0, 255)

            prob = "{} {:.2f}%".format('Probabilitas = ',max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    save_processed_img = os.path.join(STATIC_FOLDER, file.filename)
    cv2.imwrite(save_processed_img,image)

    if label == '' or prob == '':
        label = 'Wajah Manusia Tidak Terdeteksi'
        prob = 'Error'
    return label, prob


# home page
@app.route("/")
def index():
    return render_template("home.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/classificationbyimage")
def classificationbyimage():
    return render_template("classificationbyimage.html")

@app.route("/classified", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("classificationbyimage.html")

    else:
        file = request.files["image"]

        # cek ada file gambar ga
        if file.filename == '':
            return redirect(url_for('classificationbyimage'))
        # print('file.filename == ',file.filename)

        # upload gambar
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(model, upload_image_path,file)
        print('label == ',label,', prob == ',prob)
    return render_template(
        "classify.html", image_name=file.filename, label=label, prob=prob
    )

@app.route('/live_video')
def live_video():
    return render_template('live_video.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
    app.debug = True
