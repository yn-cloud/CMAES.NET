using Microsoft.VisualStudio.TestTools.UnitTesting;
using CMAESnet;
using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace CMAESnet.Tests
{
    [TestClass()]
    public class CMATests
    {
        [TestMethod()]
        public void CMATest()
        {
            CMA cma = new CMA(Vector<double>.Build.Dense(2), 1.3);

            Console.WriteLine("dd");
        }

    }
}