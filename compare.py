import cv2
import numpy as np
import os
import argparse
from matching import find_match

def main():
    '''
            arguemt example:
                python3 compare.py filename
            filename : ex. 1234_profile.jpg
                    file name of before and after images. They have same name but in different directories.
    '''

    parser = argparse.ArgumentParser(description='Arguments for predicting yolo result')
    parser.add_argument('file_name', type=str, help="target file name for yolo")
    args = parser.parse_args()
    file_name = args.file_name

    before_path = "photo/raw/before/"
    after_path = "photo/raw/after/"
    before_yolo_path = "photo/yolo/before/"
    after_yolo_path = "photo/yolo/after/"
    template_path = "photo/template/"

    before_box_txt = before_yolo_path + os.path.splitext(file_name)[0]+'_box.txt'
    after_box_txt = after_yolo_path + os.path.splitext(file_name)[0] + '_box.txt'
    before_box_file = open(before_box_txt, 'r')
    after_box_file = open(after_box_txt, 'r')

    before_img = cv2.imread(before_path + file_name)
    after_img = cv2.imread(after_path + file_name)

    before_boxes = []   # save label and img array of bounding box
    after_boxes = []    # save label and img array of bounding box

    for line in before_box_file.readlines(): # get box coordinates from txt file
        word = line[:-1].split(" ") #remove \n
        before_boxes.append((word[0],
                             before_img[int(word[2]):int(word[4]),
                                        int(word[1]):int(word[3])]))

    for line in after_box_file.readlines(): # get box coordinates from txt file
        word = line[:-1].split(" ")  # remove \n
        after_boxes.append((word[0],
                            after_img[int(word[2]):int(word[4]),
                            int(word[1]):int(word[3])]))

    match_boxes = find_match(before_boxes, after_boxes)

    for i, box in enumerate(match_boxes):
        new_defect_labels = {'scratch': 0, 'dent': 0, 'glass': 0}
        new_defect_name = os.path.splitext(file_name)[0] + '_' + str(i + 1)
        new_defect_path = template_path + new_defect_name + '.jpg'
        new_defect_txt = template_path + new_defect_name + '.txt'
        if (box[0] == 'scratch'):
            new_defect_labels['scratch'] = 1
        elif (box[0] == 'dent'):
            new_defect_labels['dent'] = 1
        elif (box[0] == 'glass'):
            new_defect_labels['glass'] = 1
        cv2.imwrite(new_defect_path, box[1])
        new_defect_txt_file = open(new_defect_txt, 'w')
        new_defect_txt_file.write(str(new_defect_labels))
        new_defect_txt_file.close()


if __name__ == '__main__':
    main()