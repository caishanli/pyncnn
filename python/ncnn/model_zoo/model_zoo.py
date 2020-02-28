from .yolov2 import MobileNet_YoloV2
from .yolov3 import MobileNetV2_YoloV3
from .mobilenetssd import MobileNet_SSD
from .mobilenetv2ssdlite import MobileNetV2_SSDLite

__all__ = ['get_model', 'get_model_list']

_models = { 
            'mobilenet_yolov2': MobileNet_YoloV2,
            'mobilenetv2_yolov3': MobileNetV2_YoloV3, 
            'mobilenet_ssd': MobileNet_SSD, 
            'mobilenetv2_ssdlite': MobileNetV2_SSDLite, 
        }

def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net

def get_model_list():
    return list(_models.keys())