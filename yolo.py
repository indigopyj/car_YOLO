from darkflow.net.build import TFNet
import argparse
import cv2
import numpy as np
import os
from matching import find_match
from PIL import Image, ImageStat, ImageEnhance
import json

# helper function for adjust brightness of input image
def adjust_brightness(image):
    temp = image.convert('L')
    stat = ImageStat.Stat(temp)
    brightness = (stat.mean[0] / 255)

    brightness_enhancer = ImageEnhance.Brightness(image)

    # change brightness for dark image
    if brightness < 0.3:
        image = brightness_enhancer.enhance(1.8 - brightness)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return image

def cropped_boxes(img, result):
    h, w, _ = img.shape
    boxes = []
    for res in result:
        label = res['label']
        conf = res['confidence']
        top_x = res['topleft']['x']
        top_y = res['topleft']['y']
        btm_x = res['bottomright']['x']
        btm_y = res['bottomright']['y']


        box = [label, conf, img[top_y : btm_y ,top_x : btm_x]]
        boxes.append(box)

    return boxes

def iou(box1, box2):
  """Compute the Intersection-Over-Union of two given boxes.
  Args:
    box1: array of 4 elements [cx, cy, width, height].
    box2: same as above
  Returns:
    iou: a float number in range [0, 1]. iou of the two boxes.
  """

  lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
      max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
  if lr > 0:
    tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
        max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    if tb > 0:
      intersection = tb*lr
      union = box1[2]*box1[3]+box2[2]*box2[3]-intersection

      return intersection/union

  return 0

def batch_iou(boxes, box):
  """Compute the Intersection-Over-Union of a batch of boxes with another
  box.
  Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
  Returns:
    ious: array of a float number in range [0, 1].
  """
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0
  )
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0
  )
  inter = lr*tb
  union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
  return inter/union


def nms(boxes, probs, threshold):
    probs = np.asarray(probs)
    order = probs.argsort()[::-1]
    boxes = np.asarray(boxes)
    boxes = np.reshape(boxes, (-1,4))
    keep = [True] * len(order)
    if(len(boxes) >= 4):
        for i in range(len(order) - 1):
            ovps = batch_iou(boxes[order[i + 1:]], boxes[order[i]])
            for j, ov in enumerate(ovps):
                if ov > threshold:
                    keep[order[j + i + 1]] = False
    return keep




def main():
    '''
        arguemt example:
            python3 yolo.py filename
        filename : ex. 1234_profile.jpg
                    file name of before and after images. They have same name but in different directories.
    '''
    parser = argparse.ArgumentParser(description='Arguments for predicting yolo result')
    parser.add_argument('file_name', type=str, help="target file name for yolo")
    args = parser.parse_args()
    file_name = args.file_name
    options = {"model" : "cfg/yolo-voc-3c-aug.cfg", "load":5068, "threshold":0.35, "gpu" : 1.0}

    tfnet = TFNet(options)

    before_path = "photo/raw/before/"
    after_path = "photo/raw/after/"
    before_yolo_path = "photo/yolo/before/"
    after_yolo_path = "photo/yolo/after/"

    # before_img = cv2.imread(before_path+file_name)
    # after_img = cv2.imread(after_path + file_name)

    before_img = Image.open(before_path+file_name)
    after_img = Image.open(after_path+file_name)

    before_img = adjust_brightness(before_img)
    after_img = adjust_brightness(after_img)

    img_set = [before_img, after_img]


    for index, img in enumerate(img_set):
        yolo_img = img.copy()
        h, w, _ = yolo_img.shape
        # detection with sliding windows method
        # slide img into four parts
        img1 = yolo_img[:int(2*h/3), : int(2*w/3)]
        img2 = yolo_img[:int(2*h/3), int(w/3):]
        img3 = yolo_img[int(h / 3):, : int(2 * w / 3)]
        img4 = yolo_img[int(h / 3):, int(w / 3):]

        result1 = tfnet.return_predict(img1)
        result2 = tfnet.return_predict(img2)
        result3 = tfnet.return_predict(img3)
        result4 = tfnet.return_predict(img4)

        results = [result1, result2, result3, result4]

        # set color of class randomly
        color1 = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        color2 = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        color3 = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        # lists for nms
        probs = []
        boxes = []
        boxes_iou = []
        label_list = []

        for i, result in enumerate(results):
            for res in result: # draw bounding boxes
                label = res['label']


                conf = res['confidence']
                # bias for finding original coordinates
                bias_x = 0
                bias_y = 0
                if(i == 1 or i == 3):
                    bias_x = w/3
                if(i == 2 or i ==3):
                    bias_y = h/3


                top_x = int(res['topleft']['x'] + bias_x)
                top_y = int(res['topleft']['y'] + bias_y)
                btm_x = int(res['bottomright']['x'] + bias_x)
                btm_y = int(res['bottomright']['y'] + bias_y)
                cx = int((top_x + btm_x)/2)
                cy = int((top_y + btm_y)/2)
                box_h = btm_y - top_y
                box_w = btm_x - top_x

                probs.append(conf)
                label_list.append(label)
                boxes_iou.append([cx,cy,box_w,box_h])
                boxes.append([(top_x,top_y), (btm_x, btm_y)])



        # Non-maximum suppression : remove duplicated boxes
        nms_res = nms(boxes, probs, 0.6)
        count = 0
        predictions = []
        predict_dict = dict()
        for i in range(len(nms_res)):
            if nms_res[i]:
                defect = dict()
                if (label_list[i] == 'scratch'):
                    color = color1
                if (label_list[i] == 'dent'):
                    color = color2
                if (label_list[i] == 'glass'):
                    color = color3

                topxy = boxes[i][0]
                btmxy = boxes[i][1]

                defect["label"] = label_list[i]
                defect["topx"] = topxy[0]
                defect["topy"] = topxy[1]
                defect["btmx"] = btmxy[0]
                defect["btmy"] = btmxy[1]
                cv2.rectangle(yolo_img, topxy, btmxy, color, 4)
                text_x, text_y = boxes[i][0][0] - 10, boxes[i][0][1] - 10
                count += 1
                cv2.putText(yolo_img, label_list[i] + str(count), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            color, 2, cv2.LINE_AA)

                predictions.append(defect)

        predict_dict["predictions"] = predictions

        if(index == 0): # before img
            save_img_path = before_yolo_path+file_name
            save_json_path = before_yolo_path+os.path.splitext(file_name)[0]+'.json'

        elif(index == 1):   # after img
            save_img_path = after_yolo_path + file_name
            save_json_path = after_yolo_path + os.path.splitext(file_name)[0] + '.json'

        cv2.imwrite(save_img_path, yolo_img)     # save img
        with open(save_json_path, 'w', encoding='utf-8') as make_file:
            json.dump(predict_dict, make_file, indent='\t')
        make_file.close()


if __name__== "__main__":
    main()
