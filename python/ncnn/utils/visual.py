import numpy as np
import cv2

def draw_detection_objects(image, class_names, objects, min_prob=0.0):
    for obj in objects:
        if obj['prob'] < min_prob:
            continue
        
        print("%d = %.5f at %.2f %.2f %.2f x %.2f\n"%(obj['label'], obj['prob'],
                obj['x'], obj['y'], obj['w'], obj['h']))

        cv2.rectangle(image, (int(obj['x']), int(obj['y'])), 
            (int(obj['x'] + obj['w']), int(obj['y'] + obj['h'])), (255, 0, 0))
        
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

def print_topk(cls_scores, topk):
    indexes = np.argsort(cls_scores)[::-1][0:topk]
    scores = cls_scores[indexes]

    for index, score in zip(indexes, scores):
        print("%d=%f"%(index, score))