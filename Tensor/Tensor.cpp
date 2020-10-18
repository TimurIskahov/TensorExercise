#include <iostream>
#include "Tensor.h"


void printShape(const std::vector<int> data)
{
    std::cout << "shape ";
    for (const auto& v : data)
        std::cout << v << " ";
    std::cout << std::endl;
}

int main()
{

    /*1*/
    Tensor<float> t1({ 2, 2 });
    t1[{0, 0}] = 1;
    t1[{1, 1}] = 2;
    Tensor<float> t4 = t1;
    Tensor<float> t2 = t1.reshape({ 1, 4 });
    t2[{0, 2}] = 3;
    Tensor<float> t3 = t1.reshape({ 4 , 1 });
    t3[{1, 0}] = 4;
    t3 = t4;
    t3[{0, 0}] = 9;
    Tensor<float> t7 = std::move(t3);

    /*2*/
    Tensor<float> t8({ 5, 5, 5 });
    Tensor<float> t9 = t8(3, 5);
    printShape(t9.dim());
    Tensor<float> t10 = t9(1, 2);
    printShape(t10.dim());
    Tensor<float> t11 = t10(0);
    printShape(t11.dim());
    Tensor<float> t12 = t11(1);
    printShape(t12.dim());
    Tensor<float> t13 = t12(1, 3);
    printShape(t13.dim());


    /*3*/
    Tensor<float> m1 = Tensor<float>({ 5, 5, 5 });
    Tensor<float> m2 = m1(3, 4);
    printShape(m2.dim());
    Tensor<float> m3 = m2;
    Tensor<float> m4;
    m4 = m1(3);
    printShape(m4.dim());
    Tensor<float> m5 = m1(3)(4)(2, 5);
    m5[{1}] = 100.0;
    std::cout << m5[{1}] << " " << m1[{3, 4, 3}] << " " << m2[{0, 4, 3}] << " " << m4[{4, 3}] << " but " << m3[{0, 4, 3}] << std::endl;
    m1[{3, 4, 4}] = -100.0;
    std::cout << m1[{3, 4, 4}] << " " << m2[{0, 4, 4}] << " " << m4[{4, 4}] << " " << m5[{2}] << " but " << m3[{0, 4, 4}] << std::endl;

}