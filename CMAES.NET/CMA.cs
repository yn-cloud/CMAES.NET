using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;

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

            int dim = mean.Count;
            if (!(dim > 1))
                throw new ArgumentOutOfRangeException("The dimension of mean must be larger than 1");

            int populationSize = 4 + (int)Math.Floor(3 * Math.Log(dim));  // # (eq. 48)

            int mu = populationSize;

            Vector<double> weightsPrime = Vector<double>.Build.Dense(populationSize);
            for (int i = 0; i < populationSize; i++)
                weightsPrime[i] = Math.Log((populationSize + 1) / 2) - Math.Log(i + 1);

            Vector<double> weightsPrimeMuEff = Vector<double>.Build.Dense(weightsPrime.Take(mu).ToArray());
            double muEff = Math.Pow(weightsPrimeMuEff.Sum(), 2) / Math.Pow(weightsPrimeMuEff.L2Norm(), 2);
            Vector<double> weightsPrimeMuEffMinus = Vector<double>.Build.Dense(weightsPrime.Skip(mu).ToArray());
            double muEffMinus = Math.Pow(weightsPrimeMuEffMinus.Sum(), 2) / Math.Pow(weightsPrimeMuEffMinus.L2Norm(), 2);

            int alphacCov = 2;
            double c1 = alphacCov / (Math.Pow(dim + 1.3, 2) + muEff);
            double cmu = Math.Min(1 - c1, alphacCov * (muEff - 2 + 1 / muEff) / (Math.Pow(dim + 2, 2) + alphacCov * muEff / 2));
            if (!(c1 <= 1 - cmu))
                throw new ArgumentException("invalid learning rate for the rank-one update");
            if (!(cmu <= 1 - c1))
                throw new ArgumentException("invalid learning rate for the rank-μ update");

            double minAlpha = Math.Min(1 + c1 / cmu, Math.Min(1 + (2 * muEffMinus) / (muEff + 2), (1 - c1 - cmu) / (dim * cmu)));


        }
    }
}
