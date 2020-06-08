# CMAES.NET

![CMAES.NET](https://buildstats.info/nuget/CMAES.NET)

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) [1] implementation on .NET

This software is a C# implementation of [CyberAgent's CMAES library](https://github.com/CyberAgent/cmaes).

## Usage

```C#  
using System;
using System.Collections.Generic;

using CMAESnet;

namespace CMAESnetExample
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] initial = new double[] { 0, 0 };
            CMAESOptimizer cmaoptimizer = new CMAESOptimizer(TestFunctions, initial, 1.5);

            cmaoptimizer.Optimize();

            Console.WriteLine(cmaoptimizer.ResultVector);
            Console.WriteLine(cmaoptimizer.ResultValue);
        }

        private static double TestFunctions(IList<double> x)
        {
            return Math.Pow(x[0] - 3, 2) + Math.Pow(10 * (x[1] + 2), 2);
        }
    }
}
```

## Link

### Other libraries

I respect all libraries involved in CMA-ES.

* [pycma](https://github.com/CMA-ES/pycma): Most famous CMA-ES implementation by Nikolaus Hansen.
* [libcmaes](https://github.com/beniz/libcmaes): Multithreaded C++11 library with Python bindings.
* [cma-es](https://github.com/srom/cma-es): A Tensorflow v2 implementation
* [CMA-ES](https://github.com/CyberAgent/cmaes): Lightweight Covariance Matrix Adaptation Evolution Strategy (CMA-ES) implementation.

### References

* [1] [N. Hansen, The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772, 2016.](https://arxiv.org/abs/1604.00772)