import time
import pyncnn as ncnn

dr = ncnn.DataReaderFromEmpty()

net = ncnn.Net()
net.load_param("test.param")
net.load_model(dr)

in_mat = ncnn.Mat(227, 227, 3)
out_mat = ncnn.Mat()

start = time.time()

ex = net.create_extractor()
ex.input("data", in_mat)
ex.extract("output", out_mat)

end = time.time()
print("timespan = ", end - start)
