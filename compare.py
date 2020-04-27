import cv2
import numpy as np
import os
import argparse
from matching import find_match
import json

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

    before_box_json = before_yolo_path + os.path.splitext(file_name)[0]+'.json'
    after_box_json = after_yolo_path + os.path.splitext(file_name)[0] + '.json'
    with open(before_box_json, 'r') as f:
        before_box = json.load(f)
    with open(after_box_json, 'r') as f:
        after_box = json.load(f)


    before_img = cv2.imread(before_path + file_name)
    after_img = cv2.imread(after_path + file_name)

    before_boxes = []   # save label and coordinate of bounding box
    after_boxes = []    # save label and coordinate of bounding box

    # get box coordinates from json file
    for box in before_box["predictions"]:
        before_boxes.append((box["label"], int(box["topx"]), int(box["topy"]), int(box["btmx"]), int(box["btmy"])))

    for box in after_box["predictions"]:
        after_boxes.append((box["label"], int(box["topx"]), int(box["topy"]), int(box["btmx"]), int(box["btmy"])))

    match_boxes = find_match(before_img, before_boxes, after_img, after_boxes)

    new_defect_path = template_path + os.path.splitext(file_name)[0] + '.json'
    new_defects = []
    new_defect_dict = dict()
    for box in match_boxes:
        defect = dict()
        defect["label"] = box[0]
        defect["topx"] = box[1]
        defect["btmx"] = box[3]
        defect["btmy"] = box[4]

        new_defects.append(defect)

    new_defect_dict["new_defects"] = new_defects

    with open(new_defect_path, 'w', encoding='utf-8') as make_file:
        json.dump(new_defect_dict, make_file, indent='\t')
    make_file.close()




if __name__ == '__main__':
    main()