import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
import cv2
import argparse
from collections import Counter
from centroidtracker import CentroidTracker
sys.path.append("..")
from utils import label_map_util
from skimage.color import rgb2lab, deltaE_cie76
from utils import visualization_utils as vis_util
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
args = vars(ap.parse_args())
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map2.pbtxt')

NUM_CLASSES = 1
ct = CentroidTracker()
writer = None
cap = cv2.VideoCapture('Video/8.mp4')
image2 = cv2.imread('black.jpg')
# cap = cv2.VideoCapture(0)
image_np = cap.read()
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

im_height = 0
im_width = 0
tableNo = ''
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colors(image, number_of_colors, show_chart):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    
    return rgb_colors

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            image2 = cv2.imread('black.jpg')
            image2 = cv2.resize(image2,(30,240))
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
           
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            rects = []
            (boxes, scores, classes, num_detections) = sess.run(
                                                                [boxes, scores, classes, num_detections],
                                                                feed_dict={image_tensor: image_np_expanded})
            im_height, im_width = image_np.shape[:2]
            person_classes = np.squeeze(classes)
            final_score = np.squeeze(scores)
            
            #detect person in each table frame in video
            for i in range(len(boxes[0])):
                if person_classes[i] == 1 and final_score[i] > 0.5:
                    position = boxes[0][i]
                    #get object's bound
                    (xmin, xmax, ymin, ymax) = (position[1]*im_width, position[3]*im_width,position[0]*im_height, position[2]*im_height)
                    if xmax - xmin < frame_width * 0.2:
                        crop_img = image_np[int(ymin):int(ymax), int(xmin):int(xmax)]
                        # cv2.rectangle(image_np,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)
                        
                        colots = get_colors(crop_img, 8, True)
                        # print(colots)
                        cv2.rectangle(image2,(0,0),(30,30),(colots[0]),-1)
                        cv2.rectangle(image2,(0,30),(30,60),(colots[1]),-1)
                        cv2.rectangle(image2,(0,60),(30,90),(colots[2]),-1)
                        cv2.rectangle(image2,(0,90),(30,120),(colots[3]),-1)
                        cv2.rectangle(image2,(0,120),(30,150),(colots[4]),-1)
                        cv2.rectangle(image2,(0,150),(30,180),(colots[5]),-1)
                        cv2.rectangle(image2,(0,180),(30,210),(colots[6]),-1)
                        cv2.rectangle(image2,(0,210),(30,240),(colots[7]),-1)
                        cv2.imshow('colot',image2)
                        cv2.imshow("Display1", image_np)
                        box = [int(xmin),int(ymin),int(xmax),int(ymax)]
                        rects.append(box)
                        
            # if writer is None:
            #     writer = cv2.VideoWriter('faster_rcnn_resnet50_coco.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
            # writer.write(image_np)

            objects = ct.update(rects)
            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                cv2.putText(image_np, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(image_np, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            # cv2.imshow("Display1", image_np)
            ank = cv2.waitKey(1) & 0xFF #key pressing detect
            if ank == ord('q') or ank == 27: #Close window
                cv2.destroyAllWindows()
                # writer.release()
                break

