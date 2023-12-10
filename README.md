# WinogradConvKernel

## Introduction
This is a test code for fast Winograd convolution which accelerates the convolution inference by reducing the number of multiplications.
This repo implementes two widely used output tile sizes, i.e., 2 and 4.

There are two versions of Winograd kernel, including a) original Winograd with element-wise matrix multiplication (EWMM) and without parallelization for transformations and; b) parallized Winograd with general matrix multiplication (GEMM) and parallelization for transformations.

Version a) can significantly accelerate the computation process compared with version b), especially when the dimension is large.

## Command for Testing
```bash
python conv_test.py
```
The code will run with regular convolution, Winograd w/o parallelization, and Winograd w/ parallelization.
We can set `m = 2` or `m=4` to change the output tile size.

## Extension
Based on the Winograd kernel, we can easliy apply it to neural networks, e.g., ResNet by replacing the class `nn.Conv2d` with `Winograd()`.
It is worth noting that for Winograd training, remember to modify the test code to make the weight be trainable using `nn.Parameter()` in `__init__`.

