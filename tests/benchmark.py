import sys
import time
import pyncnn as ncnn

g_warmup_loop_count = 8
g_loop_count = 4

g_blob_pool_allocator = ncnn.UnlockedPoolAllocator()
g_workspace_pool_allocator = ncnn.PoolAllocator()

def benchmark(comment, _in, opt):
    #_in.fill(0.01)

    net = ncnn.Net()
    net.opt = opt

    net.load_param(comment + ".param")

    dr = ncnn.DataReaderFromEmpty()
    net.load_model(dr)

    out = ncnn.Mat()

    #warm up
    for i in range(g_warmup_loop_count):
        ex = net.create_extractor()
        ex.input("data", _in)
        ex.extract("output", out)

    time_min = sys.float_info.max
    time_max = -sys.float_info.max
    time_avg = 0.0

    for i in range(g_loop_count):
        start = time.time()

        ex = net.create_extractor()
        ex.input("data", _in)
        ex.extract("output", out)

        end = time.time()

        timespan = end - start

        time_min = timespan if timespan < time_min else time_min
        time_max = timespan if timespan > time_max else time_max
        time_avg += timespan

    time_avg /= g_loop_count

    print("%20s  min = %7.2f  max = %7.2f  avg = %7.2f"%(comment, time_min, time_max, time_avg))

if __name__ == "__main__":
    loop_count = 4
    num_threads = ncnn.get_cpu_count()
    powersave = 0
    gpu_device = -1
    use_vulkan_compute = False

    g_loop_count = loop_count

    opt = ncnn.Option()
    opt.lightmode = False
    opt.num_threads = num_threads
    #opt.blob_allocator = g_blob_pool_allocator
    #opt.workspace_allocator = g_workspace_pool_allocator
    opt.use_winograd_convolution = True
    opt.use_sgemm_convolution = True
    opt.use_int8_inference = True
    opt.use_vulkan_compute = use_vulkan_compute
    opt.use_fp16_packed = True
    opt.use_fp16_storage = True
    opt.use_fp16_arithmetic = True
    opt.use_int8_storage = True
    opt.use_int8_arithmetic = True
    opt.use_packing_layout = True

    ncnn.set_cpu_powersave(powersave)
    ncnn.set_omp_dynamic(0)
    ncnn.set_omp_num_threads(num_threads)

    print("loop_count = %d"%(loop_count))
    print("num_threads = %d"%(num_threads))
    print("powersave = %d"%(ncnn.get_cpu_powersave()))
    print("gpu_device = %d"%(gpu_device))

    mat = ncnn.Mat(227, 227, 3)
    benchmark("ncnn", mat, opt)



