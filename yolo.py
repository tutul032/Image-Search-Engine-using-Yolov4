# run command python yolo.py --image image/3.jpg

import numpy as np
import _pickle as cpickle
import pickle
import time
from search import RGBHistogram
from search import Searcher
import cv2
import os


# load the COCO class labels our YOLO model was trained on
labelsPath = 'yolo-coco\coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = 'yolo-coco\yolo-obj_5000.weights'
configPath = 'yolo-coco\yolo-obj.cfg'

# load YOLO object detector trained 3 classes (bmw, audi, mercedez)
print("[INFO] loading YOLO.....")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load input image and grab its spatial dimensions
image = cv2.imread("given_S/s2.jpg")
image = cv2.resize(image, (400, 200)) 
(H, W) = image.shape[:2]

# seach similar Image
desc = RGBHistogram()
desc.feature_extraction("obj" + "/*.jpg")
queryImage = image
queryFeatures = desc.describe(queryImage)
# load the index perform the search
index = pickle.load( open( "index", "rb" ) ) 
searcher = Searcher(index)
results = searcher.search(queryFeatures)
(score, imageName) = results[0]
path = "obj" + "/%s" % (imageName)
result = cv2.imread(path)
result = cv2.resize(result, (400, 200)) 
print ("Similar Image:",imageName, " " + "Score", score)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw bounding boxes and associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize the lists of detected bounding boxes, confidences, and
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
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > 0.5:
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

			# update the list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.6f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# show the output image
cv2.imshow("Detected Image", image)
cv2.imshow("Similar Image", result)
cv2.waitKey(0)