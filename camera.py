from imutils.video import VideoStream
import time
import imutils
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


prototxtPath = r'static\face_detector\deploy.prototxt'
weightsPath = r'static\face_detector\res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model('static\mask_detector.model')


def detect_and_predict_mask(frame, faceNet, maskNet):
    # buat blob dr frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

	# deteksi wajah dr blob
    faceNet.setInput(blob)
    detections = faceNet.forward()

	# init list wajah, lokasi, n prediksi dr wajah yg terdeteksi
    faces = []
    locs = []
    preds = []

    # selama terdeteksi
    for i in range(0, detections.shape[2]):

        # ekstrak probabilitas nilai confidence
        confidence = detections[0, 0, i, 2]

        # pastiin nilai confidence kebih besar dr minimal nilai confidence
        # 0.5 sebagai threshold
        if confidence > 0.5:
            # nentukan koordinat x & y bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

			# mastiin bounding box dlm frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # ekstrak ROI wajah dan convert dr BGR ke RGB, 
            # resize 224x224px n preproses
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            # face = np.expand_dims(face, axis=0)

			# masukin wajah dan bounding box ke list
            faces.append(face)
            locs.append((startX, startY, endX, endY))

	# prediksi cm klo ada min 1 wajah terdeteksi
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces)

    return (locs, preds)


class VideoCamera(object):
    def __init__(self):
        print("[INFO] starting video stream...")
        self.video = VideoStream(src=0).start()
        time.sleep(2.0)

    def __del__(self):
        self.video.stream.release()



    def get_frame(self):
        # loop selama video msh jalan
        while True:
            # baca per frame dan resize max width 400px
            frame = self.video.read()
            frame = imutils.resize(frame, width=400)

            # deteksi wajah2 dlm frame 
            # n nentuin pake masker ga
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop selama lokasi muka terdeteksi
            for (box, pred) in zip(locs, preds):

                # bounding box
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # nentuin label class n warna bounding box n textnya
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        		# probabilitas label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display label n bounding box ke output frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            ret, jpeg = cv2.imencode('.jpg',frame)
            return jpeg.tobytes()





