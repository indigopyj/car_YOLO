import cv2
import numpy as np
import math

def average_hash(image, hashSize=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize, hashSize))
    mean = np.mean(resized)
    bin_img = np.zeros((hashSize, hashSize))
    for i in range(hashSize):
        for j in range(hashSize):
            if (resized[i][j] >= mean):  # 평균보다 크면 1, 평균보다 작으면 0
                bin_img[i][j] = 1

    return bin_img.flatten()


def hamming_distance(x, y):
    aa = x.reshape(1, -1)
    bb = y.reshape(1,-1)
    dist = (aa != bb).sum()
    return dist

def find_match(before_boxes, after_boxes, threshold = 0.09):
    '''

    :param before_boxes: (label, cropped_img_array)
    :param after_boxes: (label, cropped_img_array)
    :param threshold: threshold for hamming distance
    :return:
    '''
    match_boxes = []
    #diff_set = []
    same_set = []
    abs_threshold = 10

    for i, after in enumerate(after_boxes):
        j = 0
        for before in before_boxes:
            hash1 = average_hash(before[1])
            hash2 = average_hash(after[1])
            result = hamming_distance(hash1, hash2) / 256
            print(result, i, j)
            if (result < threshold):  # different defect
                same_set.append(i)

            j+=1
    print()
    after_set = set(range(len(after_boxes)))
    print(after_set)
    same_set = set(same_set)
    print(same_set)
    after_set = after_set.difference(same_set) # get only different defects
    print(after_set)
    #diff_set = set(diff_set)
    for i in after_set:
        match_boxes.append(after_boxes[i])

    return match_boxes




