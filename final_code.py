# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from pyzbar.pyzbar import decode
from imutils.video import VideoStream
import numpy as np
import board
import adafruit_mlx90614
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)
	print(".")

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) == 1:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
s=1
z=0
v=0
o=0
a=0
b=0
print("[INFO] starting video stream...")
#vs = VideoStream(src='http://192.168.0.160:8080/video').start()
vs= VideoStream(src=0).start()
# loop over the frames from the video stream
while True:
    frame = vs.read()
    #dim=(frame.shape[0],200)
    
    if s==1:
        frame = imutils.resize(frame, width=200)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            
            
            
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
    
            
            
            if mask>withoutMask:
                z=z+1
            if mask<withoutMask:
                z=0
            if z==2:
                s=2;
                break
            
            
    if s==2:
        i2c = board.I2C()
        mlx = adafruit_mlx90614.MLX90614(i2c)

    
        temp=str(mlx.object_temperature)
        
        if abs(mlx.object_temperature - mlx.ambient_temperature) <1.5:
            text="Put your hand in front of sensor"
            cv2.putText(frame,text, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1,color, 2)
            
            
        elif mlx.object_temperature > mlx.ambient_temperature and mlx.object_temperature < 37:
            text=("Normal Body Temperature : "+ str(int(mlx.object_temperature)))
            cv2.putText(frame,text, (50,50 ),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            a=a+1
        
        elif mlx.object_temperature > 37:
            text=("***HIGH BODY TEMPERATURE*** : "+ str(int(mlx.object_temperature)))
            cv2.putText(frame,text, (50,50 ),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            b=b+1            
        if a==25:
            s=3
            a=0
        if b==25:
            s=1
            b=0
    
    
    if s==3:
        frame = imutils.resize(frame, width=400)
        for barcode in decode(frame):
            myData = barcode.data.decode('utf-8')
            print(myData)
            verified=str(myData)
            pts = np.array([barcode.polygon],np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(frame,[pts],True,(255,0,0),5)
            pts2 = barcode.rect
            #cv2.putText(frame,myData,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,255),2)

            if verified[0:45]=="https://block.dmrr.mahaonline.gov.in/uni-pass":
                text="Fully Vaccinated , ACCESS GRANTED !!"
                cv2.putText(frame,text, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
                o=o+1
            else:
                text="Not Fully Vaccinated , ACCESS DENIED !!"
                cv2.putText(frame,text, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)
                v=v+1
            if o==15 or v==15:
                s=1
                o=0
                v=0
                break

    frame = imutils.resize(frame, width=400)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
