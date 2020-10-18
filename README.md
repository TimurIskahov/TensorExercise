# TensorExercise

This repo demonstrates how can be implemented a small-sized multidimensional array. The idea is to keep a multidimensional array as a one-dimensional sequential array and operate with dimensions.
 
A multidimensional array can be initialized as
```c++
Tensor<float> t1({ 10, 50, 100});
```
It is mean that `t1` has three dimensions with 10x50x100.

There are two operations:
1) reshaping

```c++
Tensor<float> t2 = t1.reshape({ 500 , 100 });
```
2) extracting subarray
```c++
Tensor<float> t3 = t1(3, 4);
```
The Tensor contains a shared pointer inside the class. If a subarray modifies, an original array will be modified too.
