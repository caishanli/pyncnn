import pyncnn as ncnn

        
net = ncnn.Net()
net.load_param("E:\\work\\3rdparty\\ncnn\\benchmark\\alexnet.param")

dr = ncnn.DataReaderFromEmpty()
net.load_model(dr)
