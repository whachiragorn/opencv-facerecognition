# USAGE
# python recognize_video.py --detector face_detection_model 
# --embedding-model openface_nn4.small2.v1.t7  
# --input videos/45deg.mov 
# --output output/45deg_output.mp4 
# --display 0 --recognizer output/recognizer.pickle 
# --le output/le.pickle
# python recognize_video.py --detection-method hog --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --input videos/0deg.mp4 --output output/0deg_output_04.avi --display 0 --recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import face_recognition
from json import JSONEncoder
import json
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")

ap.add_argument("-i", "--input", default=1,
	help="path to input video")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-dm", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")

ap.add_argument("-f", "--folder", type=str,
	help="folder to create file")

ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
# detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
# recognizer = pickle.loads(open(args["recognizer"], "rb").read())
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())


for  prob_i in [0.06,0.07]:
# for  prob_i in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]:
	op=args["folder"]+'/ouput_prob'+str(prob_i)+'/'
	if not os.path.exists(op) and not os.path.exists(op+"_frame"):
            os.makedirs(op)  
            print("Directory " , op,  " Created ")  
	else:
            print("Directory " , op,  " already exists")  
	f = open(op+'/output_'+str(prob_i)+'.txt', "w+")

	# initialize the video stream, then allow the camera sensor to warm up
	print("[INFO] processing video...")
	# vs = VideoStream(src=0).start()
	vs = cv2.VideoCapture(args["input"])
	# print(confidence_i)

	writer = None

	# boxjson=json.loads(open(args["all.txt","rb"]).read())

	# start the FPS throughput estimator
	# fps = FPS().start()
	framearray=[]
	framecount=0
	namearray=[]
	boxarray=[]
	

	# loop over frames from the video file stream

	while True:
		# grab the frame from the threaded video stream
		#frame = vs.read()
		(grabbed, frame) = vs.read()

		# if the frame was not grabbed, then we have reach the 
		# end of the stream
		if not grabbed:
			break

		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# frame = imutils.resize(frame, width=1000)
		# (h, w) = frame.shape[:2]

		# construct a blob from the image
		
		# imageBlob = cv2.dnn.blobFromImage(
		# 	cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		# 	(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image

		# detector.setInput(imageBlob)
		# detections = detector.forward()

		# loop over the detections
		# for i in range(0, detections.shape[2]):
		# 	# extract the confidence (i.e., probability) associated with
		# 	# the prediction
		# 	confidence = detections[0, 0, i, 2]

		# 	# filter out weak detections
		# 	if confidence > args["confidence"]:
		# 		# compute the (x, y)-coordinates of the bounding box for
		# 		# the face
		# 		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		# 		(startX, startY, endX, endY) = box.astype("int")
		
		boxes = face_recognition.face_locations(frame,
			model=args["detection_method"])
		for box  in boxes:
			(startY, endX, endY ,startX)=box
		# rescale the face coordinates
			# top = int(top * r)
			# right = int(right * r)
			# bottom = int(bottom * r)
			# left = int(left * r)
			
			# (startX, startY, endX, endY) = boxes.astype("int")
			# print (boxes)

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			# (fH, fW) = face.shape[:2]

	# 		# ensure the face width and height are sufficiently large
			# if fW < 20 or fH < 20:
			# 	continue

	# 		# construct a blob for the face ROI, then pass the blob
	# 		# through our face embedding model to obtain the 128-d
	# 		# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

					# blobFromImage(image, scalefactor, size, mean, swapRB, crop)

			# perform classification to recognize the face

			predict_proba = max(vec[0])
			preds = recognizer.predict_proba(vec)[0]

			# print(vec[0])
			
			j = np.argmax(preds)
			proba = preds[j]
			if proba>prob_i:
				name = le.classes_[j]
			else:
				name='Unknown'
			index=0
			# cv2.imwrite(op+str(framecount)+"_"+name+"_"+str(np.round(box,0))+".jpg",  frame[startY:endY,startX:endX])

			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			
			framearray.append(framecount)
			namearray.append(name)
			boxarray.append(np.round(box,0))

		framecount+=1
		if writer is None and args["output"] is not None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 24,
				(frame.shape[1], frame.shape[0]), True)

		if writer is not None:
			writer.write(frame)
			# update the FPS counter
			# fps.update()
		if args["display"] > 0:
			cv2.imshow("Frame", frame)
			# show the output frame
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

	# stop the timer and display FPS information
	# fps.stop()
	# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	vs.release()
# do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()
	class NumpyArrayEncoder(JSONEncoder):
		def default(self, obj):
			if isinstance(obj, np.ndarray):
				return obj.tolist()
			return JSONEncoder.default(self, obj)

	if writer is not None:
		writer.release()

	data = {"count": framearray, "names": namearray,"boxes":boxarray}
	f.write(json.dumps(data,cls=NumpyArrayEncoder))
	f.close()
	
# crop หน้าเซฟใส่โฟล์เดอร์ cv2.imwrite
# array.append {frame,name,boxes} write file