from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import timeit

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
    # hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    # if (show_chart):
    #     plt.figure(figsize = (8, 6))
    #     plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    
    return rgb_colors

def compare_color(color1, color2):
    threshold = 10
    count = 0
    for i in range(len(color1)):
        selected_color = rgb2lab(np.uint8(np.asarray([[color1[i]]])))
        for j in range(len(color2)):
            curr_color = rgb2lab(np.uint8(np.asarray([[color2[j]]])))
            diff = deltaE_cie76(selected_color, curr_color)
            if (diff < threshold):
                count += 1
                break

    if(count < 9):
        #not same
       return False
    else:
        #same
        return True


if __name__ == "__main__":
    #load first image
    image1 = cv2.imread('1.jpg')
    

    #load the second image 
    image2 = cv2.imread('4.jpg')

    start = timeit.default_timer()
    color1 = get_colors(image1, 10, True)
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    color2 = get_colors(image2, 10, True)
    threshold = 10
    for i in range(len(color1)):
        cv2.rectangle(image1,(0,i*30),(30,(i+1)*30),(color1[i]),-1)
        cv2.rectangle(image2,(0,i*30),(30,(i+1)*30),(color2[i]),-1)
    count = 0
    for i in range(len(color1)):
        selected_color = rgb2lab(np.uint8(np.asarray([[color1[i]]])))
        for j in range(len(color2)):
            curr_color = rgb2lab(np.uint8(np.asarray([[color2[j]]])))
            diff = deltaE_cie76(selected_color, curr_color)

            
            if (diff < threshold):
               
                count += 1
                break

    if(count < 10):
        print('No')
    else:
        print('Yes')

    cv2.imshow('image1', image1)
    cv2.imshow('image2', image2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()