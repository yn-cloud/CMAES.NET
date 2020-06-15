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

            double[] optimizedArray = cmaoptimizer.ResultVector;

            Console.WriteLine("x1={0}, x2={1}", optimizedArray[0], optimizedArray[1]);
        }

        private static double TestFunctions(IList<double> x)
        {
            return Math.Pow(x[0] - 3, 2) + Math.Pow(10 * (x[1] + 2), 2);
        }
    }
}
