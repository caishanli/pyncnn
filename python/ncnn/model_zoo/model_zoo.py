from .yolov3 import MobileNetV2_YoloV3

__all__ = ['get_model', 'get_model_list']

_models = { 'mobilenetv2_yolov3': MobileNetV2_YoloV3 }

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