# pyncnn
python wrapper of [ncnn](https://github.com/Tencent/ncnn) with [pybind11](https://github.com/pybind/pybind11), only support python3.x now.

## Prerequisites

**On Unix (Linux, OS X)**

* A compiler with C++11 support
* CMake >= 2.8.12

**On Windows**

* Visual Studio 2015
* CMake >= 3.1

## Build
1. clone [ncnn](https://github.com/Tencent/ncnn) and [pybind11](https://github.com/pybind/pybind11), build and install.
2. change /path/to to your path and running the fllowing cmd
```bash
cd /path/to/pyncnn
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/ncnn/build/install/lib/cmake/ncnn/ ..
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