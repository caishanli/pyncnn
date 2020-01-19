#include <pybind11/pybind11.h>
#include <cpu.h>
#include <net.h>
#include <option.h>
#include <blob.h>
#include <paramdict.h>

#include "pybind11_datareader.h"
#include "pybind11_allocator.h"
#include "pybind11_modelbin.h"
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

	py::class_<Blob>(m, "Blob")
		.def(py::init<>())
#if NCNN_STRING
		.def_readwrite("name", &Blob::name)
#endif // NCNN_STRING
		.def_readwrite("producer", &Blob::producer)
		.def_readwrite("consumers", &Blob::consumers);

	py::class_<ModelBin, PyModelBin<>>(m, "ModelBin");
	py::class_<ModelBinFromDataReader, ModelBin, PyModelBinOther<ModelBinFromDataReader>>(m, "ModelBinFromDataReader")
		.def(py::init<const DataReader&>())
		.def("load", &ModelBinFromDataReader::load);
	py::class_<ModelBinFromMatArray, ModelBin, PyModelBinOther<ModelBinFromMatArray>>(m, "ModelBinFromMatArray")
		.def(py::init<const Mat*>())
		.def("load", &ModelBinFromMatArray::load);

	py::class_<ParamDict>(m, "ParamDict")
		.def(py::init<>())
		.def("get", (int(ParamDict::*)(int, int) const)&ParamDict::get)
		.def("get", (float(ParamDict::*)(int, float) const)&ParamDict::get)
		.def("get", (Mat(ParamDict::*)(int, const Mat&) const)&ParamDict::get)
		.def("get", (void(ParamDict::*)(int, int))&ParamDict::set)
		.def("get", (void(ParamDict::*)(int, float))&ParamDict::set)
		.def("get", (void(ParamDict::*)(int, const Mat&))&ParamDict::set);

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
		.def(py::init<int, size_t, Allocator*>(),
			py::arg("w") = 1, 
			py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
		.def(py::init<int, int, size_t, Allocator*>(),
			py::arg("w") = 1, py::arg("h") = 1,
			py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
		.def(py::init<int, int, int, size_t, Allocator*>(),
			py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
			py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
		.def(py::init<int, size_t, int, Allocator*>(),
			py::arg("w") = 1,
			py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
		.def(py::init<int, int, size_t, int, Allocator*>(),
			py::arg("w") = 1, py::arg("h") = 1,
			py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
		.def(py::init<int, int, int, size_t, int, Allocator*>(),
			py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
			py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
		.def(py::init<const Mat&>())
		.def(py::init<int, void*, size_t, Allocator*>(),
			py::arg("w") = 1, py::arg("data") = nullptr,
			py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
		.def(py::init<int, int, void*, size_t, Allocator*>(),
			py::arg("w") = 1, py::arg("h") = 1, py::arg("data") = nullptr,
			py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
		.def(py::init<int, int, int, void*, size_t, Allocator*>(),
			py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("data") = nullptr,
			py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
		.def(py::init<int, void*, size_t, int, Allocator*>(),
			py::arg("w") = 1, py::arg("data") = nullptr,
			py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
		.def(py::init<int, int, void*, size_t, int, Allocator*>(),
			py::arg("w") = 1, py::arg("h") = 1, py::arg("data") = nullptr,
			py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
		.def(py::init<int, int, int, void*, size_t, int, Allocator*>(),
			py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("data") = nullptr,
			py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
		//todo assign
		//.def(py::self=py::self)
		.def("fill", (void(Mat::*)(int))(&Mat::fill))
		.def("fill", (void(Mat::*)(float))(&Mat::fill))
		.def("clone", (Mat(Mat::*)(Allocator*))&Mat::clone, py::arg("allocator") = nullptr)
		.def("reshape", (Mat(Mat::*)(int, Allocator*) const)&Mat::reshape,
			py::arg("w") = 1, py::arg("allocator") = nullptr)
		.def("reshape", (Mat(Mat::*)(int, int, Allocator*) const)&Mat::reshape,
			py::arg("w") = 1, py::arg("h") = 1, py::arg("allocator") = nullptr)
		.def("reshape", (Mat(Mat::*)(int, int, int, Allocator*) const)&Mat::reshape,
			py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("allocator") = nullptr)
		.def("create", (void(Mat::*)(int, size_t, Allocator*))&Mat::create,
			py::arg("w") = 1, 
			py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
		.def("create", (void(Mat::*)(int, int, size_t, Allocator*))&Mat::create,
			py::arg("w") = 1, py::arg("h") = 1,
			py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
		.def("create", (void(Mat::*)(int, int, int, size_t, Allocator*))&Mat::create,
			py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
			py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
		.def("create", (void(Mat::*)(int, size_t, int, Allocator*))&Mat::create,
			py::arg("w") = 1,
			py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
		.def("create", (void(Mat::*)(int, int, size_t, int, Allocator*))&Mat::create,
			py::arg("w") = 1, py::arg("h") = 1,
			py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
		.def("create", (void(Mat::*)(int, int, int, size_t, int, Allocator*))&Mat::create,
			py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
			py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
		.def("create_like", (void(Mat::*)(const Mat&, Allocator*))&Mat::create_like,
			py::arg("m") = Mat(), py::arg("allocator") = nullptr)
		.def("addref", &Mat::addref)
		.def("release", &Mat::release)
		.def("empty", &Mat::empty)
		.def("total", &Mat::total)
		.def("channel", (Mat(Mat::*)(int))&Mat::channel)
		.def("channel", (const Mat(Mat::*)(int) const)&Mat::channel)
		.def("row", (float*(Mat::*)(int))&Mat::row)
		.def("row", (const float*(Mat::*)(int) const)&Mat::row)
		.def("channel_range", (Mat(Mat::*)(int, int))&Mat::channel_range)
		.def("channel_range", (const Mat(Mat::*)(int, int) const)&Mat::channel_range)
		.def("row_range", (Mat(Mat::*)(int, int))&Mat::row_range)
		.def("row_range", (const Mat(Mat::*)(int, int) const)&Mat::row_range)
		.def("range", (Mat(Mat::*)(int, int))&Mat::range)
		.def("range", (const Mat(Mat::*)(int, int) const)&Mat::range)
		//todo __getitem__ in python crashed
		//.def("__getitem__", [](const Mat& m, size_t i) { return m[i]; })
		//todo convenient construct from pixel data
		//.def("from_pixels", (Mat(Mat::*)(const unsigned char*, int, int, int, Allocator*))&Mat::from_pixels)
		.def("to_pixels", (void(Mat::*)(unsigned char*, int) const)&Mat::to_pixels)
		.def("to_pixels", (void(Mat::*)(unsigned char*, int, int) const)&Mat::to_pixels)
		.def("to_pixels_resize", (void(Mat::*)(unsigned char*, int, int, int) const)&Mat::to_pixels_resize)
		.def("to_pixels_resize", (void(Mat::*)(unsigned char*, int, int, int, int) const)&Mat::to_pixels_resize)
		.def("substract_mean_normalize", &Mat::substract_mean_normalize)
		.def("from_float16", &Mat::from_float16)
		.def_readwrite("data", &Mat::data)
		.def_readwrite("refcount", &Mat::refcount)
		.def_readwrite("elemsize", &Mat::elemsize)
		.def_readwrite("elempack", &Mat::elempack)
		.def_readwrite("allocator", &Mat::allocator)
		.def_readwrite("dims", &Mat::dims)
		.def_readwrite("w", &Mat::w)
		.def_readwrite("h", &Mat::h)
		.def_readwrite("c", &Mat::c)
		.def_readwrite("cstep", &Mat::cstep);

	py::class_<Extractor>(m, "Extractor")
		.def("set_light_mode", &Extractor::set_light_mode)
		.def("set_num_threads", &Extractor::set_num_threads)
		.def("set_blob_allocator", &Extractor::set_blob_allocator)
		.def("set_blob_allocator", &Extractor::set_workspace_allocator)
#if NCNN_STRING
		.def("input", (int(Extractor::*)(const char*, const Mat&))&Extractor::input)
		.def("extract", (int(Extractor::*)(const char*, Mat&))&Extractor::extract)
#endif
		.def("input", (int(Extractor::*)(int, const Mat&))&Extractor::input)
		.def("extract", (int(Extractor::*)(int, Mat&))&Extractor::extract);

	py::class_<Net>(m, "Net")
		.def(py::init<>())
		.def_readwrite("opt", &Net::opt)
		//tode register_custom_layer
#if NCNN_STRING
		.def("load_param", (int(Net::*)(const DataReader&))&Net::load_param)
#endif // NCNN_STRING
		.def("load_param_bin", (int(Net::*)(const DataReader&))&Net::load_param_bin)
		.def("load_model", (int(Net::*)(const DataReader&))&Net::load_model)

#if NCNN_STDIO
#if NCNN_STRING
		.def("load_param", (int(Net::*)(const char*))&Net::load_param)
		.def("load_param_mem", (int(Net::*)(const char*))&Net::load_param_mem)
#endif // NCNN_STRING
		.def("load_param_bin", (int(Net::*)(const char*))&Net::load_param_bin)
		.def("load_model", (int(Net::*)(const char*))&Net::load_model)
#endif // NCNN_STDIO

		//todo load from memory
		//.def("load_param", (int (Net::*)(const unsigned char*))(&Net::load_param))
		//.def("load_model", (int (Net::*)(const unsigned char*))(&Net::load_model))

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