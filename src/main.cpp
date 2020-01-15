#include <pybind11/pybind11.h>
#include <net.h>
#include <datareader.h>
using namespace ncnn;

namespace py = pybind11;

class DataReaderFromEmpty : public ncnn::DataReader {
public:
	virtual int scan(const char* format, void* p) const {
		return 0;
	}
	virtual size_t read(void* buf, size_t size) const {
		memset(buf, 0, size); return size;
	}
};

class PyDataReader : public DataReader {
public:
	using DataReader::DataReader;
#if NCNN_STRING
	int scan(const char* format, void* p) const override {
		PYBIND11_OVERLOAD(int, DataReader, scan, format, p);
	}
#endif
	size_t read(void* buf, size_t size) const override {
		PYBIND11_OVERLOAD(size_t, DataReader, read, buf, size);
	}
};

class PyDataReaderFromMemory : public DataReaderFromMemory {
public:
	using DataReaderFromMemory::DataReaderFromMemory;
#if NCNN_STRING
	int scan(const char* format, void* p) const override {
		PYBIND11_OVERLOAD(int, DataReaderFromMemory, scan, format, p);
	}
#endif
	size_t read(void* buf, size_t size) const override {
		PYBIND11_OVERLOAD(size_t, DataReaderFromMemory, read, buf, size);
	}
};

class PyDataReaderFromEmpty : public DataReaderFromEmpty {
public:
	using DataReaderFromEmpty::DataReaderFromEmpty;
#if NCNN_STRING
	int scan(const char* format, void* p) const override {
		PYBIND11_OVERLOAD(int, DataReaderFromEmpty, scan, format, p);
	}
#endif
	size_t read(void* buf, size_t size) const override {
		PYBIND11_OVERLOAD(size_t, DataReaderFromEmpty, read, buf, size);
	}
};

PYBIND11_MODULE(pyncnn, m) {
	py::class_< DataReader, PyDataReader>(m, "DataReader")
		.def(py::init<>())
		.def("scan", &DataReader::scan)
		.def("read", &DataReader::read)
		;
	//py::class_<DataReaderFromMemory, PyDataReaderFromMemory>(m, "DataReaderFromMemory")
	//	.def(py::init<const unsigned char*>())
	//	.def("scan", &DataReaderFromMemory::scan)
	//	.def("read", &DataReaderFromMemory::read)
	//	;
	py::class_<DataReaderFromEmpty, PyDataReaderFromEmpty>(m, "DataReaderFromEmpty")
		.def(py::init<>())
		.def("scan", &DataReaderFromEmpty::scan)
		.def("read", &DataReaderFromEmpty::read)
		;

	py::class_<Mat>(m, "Mat")
		.def(py::init<>())
		.def(py::init<int, size_t>())
		.def(py::init<int, int, size_t>())
		.def(py::init<int, int, int, size_t>())
		.def(py::init<int, size_t, int>())
		.def(py::init<int, int, size_t, int>())
		//.def("fill", py::overload_cast<float>(&Mat::fill))
		//.def("fill", py::overload_cast<int>(&Mat::fill))
		;

	py::class_<ncnn::Extractor>(m, "Extractor")
		.def("set_light_mode", &Extractor::set_light_mode)
		.def("set_num_threads", &Extractor::set_num_threads)
#if NCNN_STRING
		.def("input", py::overload_cast<const char*, const Mat&>(&Extractor::input))
		.def("extract", py::overload_cast<const char*, Mat&>(&Extractor::extract))
#endif
		.def("input", py::overload_cast<int, const Mat&>(&Extractor::input))
		.def("extract", py::overload_cast<int, Mat&>(&Extractor::extract))
		;

	py::class_<Net>(m, "Net")
		.def(py::init<>())
		.def("load_param", py::overload_cast<const DataReader&>(&Net::load_param))
		.def("load_model", py::overload_cast<const DataReader&>(&Net::load_model))
		//.def("load_model", py::overload_cast<const DataReaderFromEmpty&>(&Net::load_model))
#if NCNN_STRING
		.def("load_param", py::overload_cast<const char*>(&Net::load_param))
		.def("load_model", py::overload_cast<const char*>(&Net::load_model))
#endif
		.def("load_param", py::overload_cast<const unsigned char*>(&Net::load_param))
		.def("load_model", py::overload_cast<const unsigned char*>(&Net::load_model))
		.def("create_extractor", &Net::create_extractor)
		;

	m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

	/*
	m.def("add", &add, R"pbdoc(
	Add two numbers
	Some other explanation about the add function.
	)pbdoc");

	m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
	Subtract two numbers
	Some other explanation about the subtract function.
	)pbdoc");
	*/
#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}