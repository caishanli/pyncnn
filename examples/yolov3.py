import sys
import cv2
import numpy as np
import ncnn

use_gpu = False
if ncnn.build_with_gpu():
    use_gpu = True

def detect_yolov3(bgr):
    yolov3 = ncnn.Net()
    yolov3.opt.use_vulkan_compute = use_gpu

    # original pretrained model from https://github.com/eric612/MobileNet-YOLO
    # param : https://drive.google.com/open?id=1V9oKHP6G6XvXZqhZbzNKL6FI_clRWdC-
    # bin : https://drive.google.com/open?id=1DBcuFCr-856z3FRQznWL_S5h-Aj3RawA
    # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    yolov3.load_param("mobilenetv2_yolov3.param")
    yolov3.load_model("mobilenetv2_yolov3.bin")

    target_size = 352

    img_w = bgr.shape[1]
    img_h = bgr.shape[0]

    mat_in = ncnn.Mat.from_pixels_resize(bgr, ncnn.Mat.PixelType.PIXEL_BGR, bgr.shape[1], bgr.shape[0], target_size, target_size)

    mean_vals = [127.5, 127.5, 127.5]
    norm_vals = [0.007843, 0.007843, 0.007843]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)

    ex = yolov3.create_extractor()
    ex.set_num_threads(4)

    ex.input("data", mat_in)

    mat_out = ncnn.Mat()
    ex.extract("detection_out", mat_out)

    objects = []

    #printf("%d %d %d\n", mat_out.w, mat_out.h, mat_out.c)
    
    #method 1, use ncnn.Mat.row to get the result, no memory copy
    for i in range(mat_out.h):
        values = mat_out.row(i)

        obj = {}
        obj['label'] = values[0]
        obj['prob'] = values[1]
        obj['x'] = values[2] * img_w
        obj['y'] = values[3] * img_h
        obj['width'] = values[4] * img_w - obj['x']
        obj['height'] = values[5] * img_h - obj['y']

        objects.append(obj)
    
    '''
    #method 2, use ncnn.Mat->numpy.array to get the result, no memory copy too
    out = np.array(mat_out)
    for i in range(len(out)):
        values = out[i]

        obj = {}
        obj['label'] = values[0]
        obj['prob'] = values[1]
        obj['x'] = values[2] * img_w
        obj['y'] = values[3] * img_h
        obj['width'] = values[4] * img_w - obj['x']
        obj['height'] = values[5] * img_h - obj['y']

        objects.append(obj)
    '''
    
    # extractor need relese manually when build ncnn with vuklan,
    # due to python relese ex after net, but in extractor.destruction use net
    ex = None

    return objects

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

def main():
    #if len(sys.argv) != 2:
    #   print("Usage: %s [imagepath]\n"%(sys.argv[0]))
    #    return -1

    #imagepath = sys.argv[1]
    imagepath = "dog.jpg"

    m = cv2.imread(imagepath)
    if m is None:
        print("cv2.imread %s failed\n"%(imagepath))
        return -1

    if use_gpu:
        ncnn.create_gpu_instance()

    objects = detect_yolov3(m)

    if use_gpu:
        ncnn.destroy_gpu_instance()

    draw_objects(m, objects)

    return 0

if __name__ == "__main__":
    main()