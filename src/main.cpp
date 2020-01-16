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
		//.def("fill", py::overload_cast<float>(&Mat::fill))
		//.def("fill", py::overload_cast<int>(&Mat::fill))
		.def("empty", &Mat::empty)
		.def("total", &Mat::total)
		.def("channel", py::overload_cast<int>(&Mat::channel))
		//.def("row", py::overload_cast<int>(&Mat::row))
		.def("channel_range", py::overload_cast<int, int>(&Mat::channel_range))
		.def("row_range", py::overload_cast<int, int>(&Mat::row_range))
		.def("range", py::overload_cast<int, int>(&Mat::range))
		.def("__getitem__", [](const Mat& m, size_t i) { return m[i]; })
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

	m.def("set_cpu_powersave", &set_cpu_powersave);
	m.def("set_cpu_powersave", &set_omp_dynamic);
	m.def("set_cpu_powersave", &set_omp_num_threads);

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