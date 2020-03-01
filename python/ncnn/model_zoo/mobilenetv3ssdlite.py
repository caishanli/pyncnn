import numpy as np
import ncnn
from .model_store import get_model_file

def clamp(v, lo, hi):
    if v < lo:
        return lo
    elif hi < v:
        return hi
    else:
        return v

class MobileNetV3_SSDLite:
    def __init__(self, img_width=300, img_height=300, num_threads=1, use_gpu=False):
        self.img_width = img_width
        self.img_height = img_height
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = [123.675, 116.28, 103.53]
        self.norm_vals = [1.0, 1.0, 1.0]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu

        # converted ncnn model from https://github.com/ujsyehao/mobilenetv3-ssd
        # the ncnn model https://github.com/caishanli/pyncnn-assets/tree/master/models
        self.net.load_param(get_model_file("mobilenetv3_ssdlite_voc.param"))
        self.net.load_model(get_model_file("mobilenetv3_ssdlite_voc.bin"))

        self.class_names = ["background",
            "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"]

    def __del__(self):
        self.net = None

    def __call__(self, img):
        img_h = img.shape[0]
        img_w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img.shape[1], img.shape[0], self.img_width, self.img_height)
        mat_in.substract_mean_normalize([], self.norm_vals)
        mat_in.substract_mean_normalize(self.mean_vals, [])

        ex = self.net.create_extractor()
        ex.set_light_mode(True)
        ex.set_num_threads(self.num_threads)

        ex.input("input", mat_in)

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

            x1 = clamp(values[2] * self.img_width, 0.0, float(self.img_width - 1)) / self.img_width * img_w
            y1 = clamp(values[3] * self.img_height, 0.0, float(self.img_height - 1)) / self.img_height * img_h
            x2 = clamp(values[4] * self.img_width, 0.0, float(self.img_width - 1)) / self.img_width * img_w
            y2 = clamp(values[5] * self.img_height, 0.0, float(self.img_height - 1)) / self.img_height * img_h

            if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                continue

            obj['x'] = x1
            obj['y'] = y1
            obj['w'] = x2 - x1
            obj['h'] = y2 - y1

            objects.append(obj)
            
        '''
        #method 2, use ncnn.Mat->numpy.array to get the result, no memory copy too
        out = np.array(mat_out)
        for i in range(len(out)):
            values = out[i]
            obj = {}
            obj['label'] = values[0]
            obj['prob'] = values[1]

            x1 = clamp(values[2] * self.img_width, 0.0, float(self.img_width - 1)) / self.img_width * img_w
            y1 = clamp(values[3] * self.img_height, 0.0, float(self.img_height - 1)) / self.img_height * img_h
            x2 = clamp(values[4] * self.img_width, 0.0, float(self.img_width - 1)) / self.img_width * img_w
            y2 = clamp(values[5] * self.img_height, 0.0, float(self.img_height - 1)) / self.img_height * img_h

            obj['x'] = x1
            obj['y'] = y1
            obj['w'] = x2 - x1
            obj['h'] = y2 - y1

            objects.append(obj)
        '''

        return objects
