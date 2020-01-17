#ifndef PYBIND11_NCNN_ALLOCATOR_H
#define PYBIND11_NCNN_ALLOCATOR_H

#include <allocator.h>

template <class Base = ncnn::Allocator>
class PyAllocator : public Base {
public:
	using Base::Base; // Inherit constructors
	void* fastMalloc(size_t size) override {
		PYBIND11_OVERLOAD_PURE(void*, Base, fastMalloc, size);
	}
	void fastFree(void* ptr) override {
		PYBIND11_OVERLOAD_PURE(void, Base, fastFree, ptr);
	}
};

template <class Other>
class PyAllocatorOther : public PyAllocator<Other> {
public:
	using PyAllocator<Other>::PyAllocator;
	void* fastMalloc(size_t size) override {
		PYBIND11_OVERLOAD(void*, Other, fastMalloc, size);
	}
	void fastFree(void* ptr) override {
		PYBIND11_OVERLOAD(void, Other, fastFree, ptr);
	}
};

#endif