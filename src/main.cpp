#include <pybind11/pybind11.h>
#include <cpu.h>
#include <net.h>
#include <option.h>

#include "pybind11_datareader.h"
#include "pybind11_allocator.h"
using namespace ncnn;

namespace py = pybind11;

PYBIND11_MODULE(pyncnn, m) {
	py::class_<Allocator, PyAllocator<>>(m, "Allocator");
	py::class_<PoolAllocator, Allocator, PyAllocatorOther<PoolAllocator>>(m, "PoolAllocator")
		.def(py::init<>())
		.def("set_size_compare_ratio", &PoolAllocator::set_size_compare_ratio)
		.def("clear", &PoolAllocator::clear)
		.def("fastMalloc", &PoolAllocator::fastMalloc)
		.def("fastFree", &PoolAllocator::fastFree);
	py::class_<UnlockedPoolAllocator, Allocator, PyAllocatorOther<UnlockedPoolAllocator>>(m, "UnlockedPoolAllocator")
		.def(py::init<>())
		.def("set_size_compare_ratio", &UnlockedPoolAllocator::set_size_compare_ratio)
		.def("clear", &UnlockedPoolAllocator::clear)
		.def("fastMalloc", &UnlockedPoolAllocator::fastMalloc)
		.def("fastFree", &UnlockedPoolAllocator::fastFree);

	py::class_<DataReader, PyDataReader<>>(m, "DataReader")
		.def(py::init<>())
		.def("scan", &DataReader::scan)
		.def("read", &DataReader::read);
	py::class_<DataReaderFromEmpty, DataReader, PyDataReaderOther<DataReaderFromEmpty>>(m, "DataReaderFromEmpty")
		.def(py::init<>())
		.def("scan", &DataReaderFromEmpty::scan)
		.def("read", &DataReaderFromEmpty::read);

	py::class_<Option>(m, "Option")
		.def(py::init<>())
		.def_readwrite("lightmode", &Option::lightmode)
		.def_readwrite("num_threads", &Option::num_threads)
		.def_readwrite("blob_allocator", &Option::blob_allocator)
		.def_readwrite("workspace_allocator", &Option::workspace_allocator)
		.def_readwrite("use_winograd_convolution", &Option::use_winograd_convolution)
		.def_readwrite("use_sgemm_convolution", &Option::use_sgemm_convolution)
		.def_readwrite("use_int8_inference", &Option::use_int8_inference)
		.def_readwrite("use_vulkan_compute", &Option::use_vulkan_compute)
		.def_readwrite("use_fp16_packed", &Option::use_fp16_packed)
		.def_readwrite("use_fp16_storage", &Option::use_fp16_storage)
		.def_readwrite("use_fp16_arithmetic", &Option::use_fp16_arithmetic)
		.def_readwrite("use_int8_storage", &Option::use_int8_storage)
		.def_readwrite("use_int8_arithmetic", &Option::use_int8_arithmetic)
		.def_readwrite("use_packing_layout", &Option::use_packing_layout);

	py::class_<Mat>(m, "Mat")
		.def(py::init<>())
		.def(py::init<int>())
		.def(py::init<int, size_t>())
		.def(py::init<int, size_t, Allocator*>())
		.def(py::init<int, int>())
		.def(py::init<int, int, size_t>())
		.def(py::init<int, int, size_t, Allocator*>())
		.def(py::init<int, int, int>())
		.def(py::init<int, int, int, size_t>())
		.def(py::init<int, int, int, size_t, Allocator*>())
		.def(py::init<int, size_t, int>())
		.def(py::init<int, size_t, int, Allocator*>())
		.def(py::init<int, int, size_t, int>())
		.def(py::init<int, int, size_t, int, Allocator*>())
		.def(py::init<int, int, int, size_t, int>())
		.def(py::init<int, int, int, size_t, int, Allocator*>())
		.def(py::init<const Mat&>())
		.def(py::init<int, void*>())
		.def(py::init<int, void*, size_t>())
		.def(py::init<int, void*, size_t, Allocator*>())
		.def(py::init<int, int, void*>())
		.def(py::init<int, int, void*, size_t>())
		.def(py::init<int, int, void*, size_t, Allocator*>())
		.def(py::init<int, int, int, void*>())
		.def(py::init<int, int, int, void*, size_t>())
		.def(py::init<int, int, int, void*, size_t, Allocator*>())
		.def(py::init<int, void*, size_t, int>())
		.def(py::init<int, void*, size_t, int, Allocator*>())
		.def(py::init<int, int, void*, size_t, int>())
		.def(py::init<int, int, void*, size_t, int, Allocator*>())
		.def(py::init<int, int, int, void*, size_t, int>())
		.def(py::init<int, int, int, void*, size_t, int, Allocator*>())
		//todo assign
		//.def(py::self = py::self)
		//todo how to bind fill(int) with no template
		.def("fill", py::overload_cast<float>(&Mat::fill<float>))
		.def("fill", py::overload_cast<int>(&Mat::fill<int>))
		//todo overload functions with default args
		//.def("clone", &Mat::clone)
		//.def("reshape", &Mat::reshape)
		//.def("create", &Mat::create)
		//.def("create_like", &Mat::create_like)
		.def("addref", &Mat::addref)
		.def("release", &Mat::release)
		.def("empty", &Mat::empty)
		.def("total", &Mat::total)
		.def("channel", py::overload_cast<int>(&Mat::channel))
		.def("channel", py::overload_cast<int>(&Mat::channel, py::const_))
		.def("row", py::overload_cast<int>(&Mat::row<float>))
		.def("row", py::overload_cast<int>(&Mat::row<float>, py::const_))
		.def("channel_range", py::overload_cast<int, int>(&Mat::channel_range))
		.def("channel_range", py::overload_cast<int, int>(&Mat::channel_range, py::const_))
		.def("row_range", py::overload_cast<int, int>(&Mat::row_range))
		.def("row_range", py::overload_cast<int, int>(&Mat::row_range, py::const_))
		.def("range", py::overload_cast<int, int>(&Mat::range))
		.def("range", py::overload_cast<int, int>(&Mat::range, py::const_))
		.def("__getitem__", [](const Mat& m, size_t i) { return m[i]; })
		//todo convenient construct from pixel data
		.def("to_pixels", py::overload_cast<unsigned char*, int>(&Mat::to_pixels, py::const_))
		.def("to_pixels", py::overload_cast<unsigned char*, int, int>(&Mat::to_pixels, py::const_))
		.def("to_pixels_resize", py::overload_cast<unsigned char*, int, int, int>(&Mat::to_pixels_resize, py::const_))
		.def("to_pixels_resize", py::overload_cast<unsigned char*, int, int, int, int>(&Mat::to_pixels_resize, py::const_))
		.def("substract_mean_normalize", &Mat::substract_mean_normalize)
		.def_readwrite("data", &Mat::data)
		.def_readwrite("refcount", &Mat::refcount)
		.def_readwrite("elemsize", &Mat::elemsize)
		.def_readwrite("elempack", &Mat::elempack)
		.def_readwrite("allocator", &Mat::allocator)
		.def_readwrite("dims", &Mat::dims)
		.def_readwrite("w", &Mat::w)
		.def_readwrite("h", &Mat::h)
		.def_readwrite("c", &Mat::c)
		.def_readwrite("cstep", &Mat::cstep)
		;

	py::class_<ncnn::Extractor>(m, "Extractor")
		.def("set_light_mode", &Extractor::set_light_mode)
		.def("set_num_threads", &Extractor::set_num_threads)
#if NCNN_STRING
		.def("input", py::overload_cast<const char*, const Mat&>(&Extractor::input))
		.def("extract", py::overload_cast<const char*, Mat&>(&Extractor::extract))
#endif
		.def("input", py::overload_cast<int, const Mat&>(&Extractor::input))
		.def("extract", py::overload_cast<int, Mat&>(&Extractor::extract));

	py::class_<Net>(m, "Net")
		.def(py::init<>())
		.def_readwrite("opt", &Net::opt)
		.def("load_param", py::overload_cast<const DataReader&>(&Net::load_param))
		.def("load_model", py::overload_cast<const DataReader&>(&Net::load_model))
		//.def("load_model", py::overload_cast<const DataReaderFromEmpty&>(&Net::load_model))
#if NCNN_STRING
		.def("load_param", py::overload_cast<const char*>(&Net::load_param))
		.def("load_model", py::overload_cast<const char*>(&Net::load_model))
#endif
		.def("load_param", py::overload_cast<const unsigned char*>(&Net::load_param))
		.def("load_model", py::overload_cast<const unsigned char*>(&Net::load_model))
		.def("clear", &Net::clear)
		.def("create_extractor", &Net::create_extractor);

	m.def("cpu_support_arm_neon", &cpu_support_arm_neon);
	m.def("cpu_support_arm_vfpv4", &cpu_support_arm_vfpv4);
	m.def("cpu_support_arm_asimdhp", &cpu_support_arm_asimdhp);
	m.def("get_cpu_count", &get_cpu_count);
	m.def("get_cpu_powersave", &get_cpu_powersave);
	m.def("set_cpu_powersave", &set_cpu_powersave);
	m.def("get_omp_num_threads", &get_omp_num_threads);
	m.def("set_omp_num_threads", &set_omp_num_threads);
	m.def("get_omp_dynamic", &get_omp_dynamic);
	m.def("set_omp_dynamic", &set_omp_dynamic);

	m.doc() = R"pbdoc(
        ncnn python wrapper
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}