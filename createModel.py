import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import tensorflow as tf
from utils import label_map_util
import numpy as np
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    NUM_CLASSES = 1
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

    # PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map2.pbtxt')
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

    # Loop through each person in the training set
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for class_dir in os.listdir(train_dir):
                if not os.path.isdir(os.path.join(train_dir, class_dir)):
                    continue

                # Loop through each training image for the current person
                for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                    print(img_path)
                    image_np = face_recognition.load_image_file(img_path)
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
                    for i in range(len(boxes[0])):
                        if person_classes[i] == 1 and final_score[i] > 0.5:
                            position = boxes[0][i]
                            face_bounding_boxes = (int(position[0]*im_height), int(position[3]*im_width), int(position[2]*im_height), int(position[1]*im_width))
                            xx = []
                            xx.append(face_bounding_boxes)
                            # print(xx)
                            # Add face encoding for current image to the training set
                            X.append(face_recognition.face_encodings(image_np, known_face_locations=xx)[0])
                            y.append(class_dir)
                           

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def recount():
    global detect_delay, detect
    if detect_delay != 0:
        detect_delay -= 1
    else:
        detect = True

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    
    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()

detect = True
detect_delay = 10

if __name__ == "__main__":
    video_capture = cv2.VideoCapture("video/8.mp4")
    # video_capture = cv2.VideoCapture(0)
    print("Training KNN classifier...")
    classifier = train("data/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    knn_clf=None
    model_path="trained_knn_model.clf"
    distance_threshold=0.45
    NUM_CLASSES = 1
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

    # PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map2.pbtxt')
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
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                
                ret,X_img = video_capture.read()
                cv2.imshow('Display',X_img)

                ret,X_img = video_capture.read()
                if knn_clf is None and model_path is None:
                    raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
                knn_clf=None
                # Load a trained KNN model (if one was passed in)
                if knn_clf is None:
                    with open(model_path, 'rb') as f:
                        knn_clf = pickle.load(f)

                # image_np = face_recognition.load_image_file(img_path)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(X_img, axis=0)
            
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
                im_height, im_width = X_img.shape[:2]
                person_classes = np.squeeze(classes)
                final_score = np.squeeze(scores)                                                    
                for i in range(len(boxes[0])):
                    if person_classes[i] == 1 and final_score[i] > 0.5:
                        position = boxes[0][i]
                        face_bounding_boxes = (int(position[0]*im_height), int(position[3]*im_width), int(position[2]*im_height), int(position[1]*im_width))
                        # (top, right, bottom, left) = (int(position[0]*im_height), int(position[3]*im_width), int(position[2]*im_height), int(position[1]*im_height))
                        # (xmin, xmax, ymin, ymax) = (position[1]*im_width, position[3]*im_width,position[0]*im_height, position[2]*im_height)
                        # cv2.rectangle(X_img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)
                        X_face_locations = []
                        X_face_locations.append(face_bounding_boxes)
                        faces_encodings = face_recognition.face_encodings(X_img,known_face_locations=X_face_locations)
                        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=4)
                        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
                        predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
                        for name, (top, right, bottom, left) in predictions:
                            if name != "unknow" and detect:
                                detect = False
                                detect_delay = 5
                                
                            elif name == "unknow":
                                recount()
                            cv2.rectangle(X_img,(left, top),(right, bottom),(0,255,0),3)
                            cv2.rectangle(X_img,(left, bottom - 25),(right, bottom),(0,255,0),3)
                            cv2.putText(X_img,name,( left + 6, bottom - 8  ) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA )
                    else:
                        recount()

                # X_face_locations = face_recognition.face_locations(X_img)
                # if len(X_face_locations) != 0:
                #     faces_encodings = face_recognition.face_encodings(X_img,known_face_locations=X_face_locations)
                #     closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
                #     are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
                #     predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
                #     for name, (top, right, bottom, left) in predictions:
                #         if name != "unknow" and detect:
                #             detect = False
                #             detect_delay = 5
                #             print("beep")
                #         elif name == "unknow":
                #             recount()
                #         cv2.rectangle(X_img,(left, top),(right, bottom),(0,255,0),3)
                #         cv2.rectangle(X_img,(left, bottom - 25),(right, bottom),(0,255,0),3)
                #         cv2.putText(X_img,name,( left + 6, bottom - 8  ) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA )
                # else:
                #     recount()
                cv2.imshow('Display',X_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

