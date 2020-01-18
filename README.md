# pyncnn
python wrapper of [ncnn](https://github.com/Tencent/ncnn) with [pybind11](https://github.com/pybind/pybind11)

## Prerequisites

**On Unix (Linux, OS X)**

* A compiler with C++11 support
* CMake >= 2.8.12

**On Windows**

* Visual Studio 2015 (required for all Python versions, see notes below)
* CMake >= 3.1

## Build
```bash
cd /path/to/pyncnn
mkdir build
cd build
cmake ..
make
cd pyncnn
pip install .
```

## Tests
**test**
```bash
cd /path/to/pyncnn/tests
python3 test.py
```

**benchmark**
```bash
cd /path/to/pyncnn/tests
python3 benchmark.py
```