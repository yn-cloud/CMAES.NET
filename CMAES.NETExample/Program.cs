using System;
using System.Collections.Generic;

using CMAESnet;

namespace CMAESnetExample
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] initial = new double[] { 2, 3 };
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
