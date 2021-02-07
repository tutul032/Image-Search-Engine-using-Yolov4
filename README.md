# Yolov4-Image-Search-Engine-for-car-brand
Dependencies:
1. scipy
2. OpenCV 4.5.1
3. Numpy

Please install the dependencies and run the yolo.py from the command line which will detect the car brand and show the closest car image of the detected car image from the training dataset of yolo weights. search.py methods computes the features of all images from the training dataset and also finds and sorts the euclidian distance between the detected car image feature and all image feature of training dataset and displays the closest car image from training dataset. The weights file can be downloaded from the following link: 
https://drive.google.com/file/d/1KcQ3r5IOzUBKiQ8Pg0BlwFhmlJ6GMJnS/view?usp=sharing

Please download the weights "yolo-obj_5000.weights" and place it into the yolo-coco folder then run the yolo.py. If you want to detect other images please change the given_S folder images and line 31 from yolo.py.
# Result
![image](https://github.com/tutul032/Image-search-engine-using-yolov4/blob/main/yolov4.jpg)
