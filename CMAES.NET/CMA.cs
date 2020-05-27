using MathNet.Numerics.LinearAlgebra;
using System;

namespace CMAES.NET
{
    public class CMA
    {
        public int Dim { get; private set; }
        public int PopulationSize { get; private set; }
        public int Generation { get; set; }

        public CMA(Vector<double> mean, double sigma, Vector<double> bounds = null, int n_max_resampling = 100, int seed = 0)
        {
            if (!(sigma > 0))
                throw new ArgumentOutOfRangeException("sigma must be non-zero positive value");
        }
    }
}
