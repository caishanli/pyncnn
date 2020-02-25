import sys
import cv2
import numpy as np
import ncnn
from ncnn.model_zoo import get_model

use_gpu = False
if ncnn.build_with_gpu():
    use_gpu = True

def draw_objects(bgr, objects):
    class_names = ["background",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"]

    image = bgr

    for obj in objects:
        print("%d = %.5f at %.2f %.2f %.2f x %.2f\n"%(obj['label'], obj['prob'],
                obj['x'], obj['y'], obj['width'], obj['height']))

        cv2.rectangle(image, (int(obj['x']), int(obj['y'])), 
            (int(obj['x'] + obj['width']), int(obj['y'] + obj['height'])), (255, 0, 0))
        
        text = "%s %.1f%%"%(class_names[int(obj['label'])], obj['prob'] * 100)

        label_size, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        x = obj['x']
        y = obj['y'] - label_size[1] - baseLine
        if y < 0:
            y = 0
        if x + label_size[0] > image.shape[1]:
            x = image.shape[1] - label_size[0]

        cv2.rectangle(image, (int(x), int(y)), (int(x + label_size[0]), int(y + label_size[1] + baseLine)),
                      (255, 255, 255), -1)

        cv2.putText(image, text, (int(x), int(y + label_size[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow("image", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s [imagepath]\n"%(sys.argv[0]))
        sys.exit(0)

    imagepath = sys.argv[1]

    m = cv2.imread(imagepath)
    if m is None:
        print("cv2.imread %s failed\n"%(imagepath))
        sys.exit(0)

    net = get_model('mobilenet_yolov2', num_threads=4, use_gpu=use_gpu)

    objects = net(m)

    draw_objects(m, objects)