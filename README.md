# cuda_sgemm

通过不同的blocking模型和参数计算4096*4096的sgemm，学习CUDA的优化技术。

1. 通过cublas计算sgemm。作为一个CUDA的使用者，编写一个可以对比官方库的程序是很难得，但是依然可以把cublas作为参考结果。另外，目前也有一些工作通过汇编技术达到超越cublas的性能，这方面暂时是不考虑的。

2. V10。这里主要是用了shared memory的blocking算法，主要代码来源于官方教程，唯一的不同是这里的程序使用列主元（column major）的存储方式，这和cublas是一致的。

3. V20：使用shared memory和rigester的double blocking算法。目前仅仅完成了基本代码，速度比V10还要慢，参数还需要调整。

