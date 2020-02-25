import ncnn
from .model_store import get_model_file

class MobileNetV2_YoloV3:
    def __init__(self, img_width=352, img_height=352, num_threads=1, use_gpu=False):
        self.img_width = img_width
        self.img_height = img_height
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = [127.5, 127.5, 127.5]
        self.norm_vals = [0.007843, 0.007843, 0.007843]

        if self.use_gpu:
            ncnn.create_gpu_instance()

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.load_param(get_model_file("mobilenetv2_yolov3.param"))
        self.net.load_model(get_model_file("mobilenetv2_yolov3.bin"))

        self.ex = self.net.create_extractor()
        self.ex.set_num_threads(self.num_threads)

    def __del__(self):
        self.ex = None
        self.net = None

        if self.use_gpu:
            ncnn.destroy_gpu_instance()

    def __call__(self, img):
        img_h = img.shape[0]
        img_w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR, img.shape[1], img.shape[0], self.img_width, self.img_height)
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        self.ex.input("data", mat_in)

        mat_out = ncnn.Mat()
        self.ex.extract("detection_out", mat_out)

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

        return objects
