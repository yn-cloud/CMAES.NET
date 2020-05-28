using Microsoft.VisualStudio.TestTools.UnitTesting;
using CMAES.NET;
using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace CMAES.NET.Tests
{
    [TestClass()]
    public class CMATests
    {
        [TestMethod()]
        public void CMATest()
        {
            CMA cma = new CMA(Vector<double>.Build.Dense(3), 1e-3);
        }

    }
}