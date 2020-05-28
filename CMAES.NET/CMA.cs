using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using System;
using System.Linq;
using System.Security.Cryptography.X509Certificates;

namespace CMAES.NET
{
    public class CMA
    {
        private int mu;
        private double muEff;
        private double cc;
        private double c1;
        private double cmu;
        private double cSigma;
        private double dSigma;
        private int cm;
        private double chiN;
        private Vector<double> weights;
        private Vector<double> pSigma;
        private Vector<double> pc;
        private Vector<double> mean;
        private Matrix<double> C;
        private double sigma;
        private Matrix<double> D;
        private Matrix<double> B;
        private Matrix<double> bounds;
        private int nMaxResampling;
        private int g;
        private Xorshift rng;
        private double epsilon;

        public int Dim { get; private set; }
        public int PopulationSize { get; private set; }
        public int Generation { get; set; }

        public CMA(Vector<double> mean, double sigma, Matrix<double> bounds = null, int nMaxResampling = 100, int seed = 0)
        {
            if (!(sigma > 0))
            {
                throw new ArgumentOutOfRangeException("sigma must be non-zero positive value");
            }

            int nDim = mean.Count;
            if (!(nDim > 1))
            {
                throw new ArgumentOutOfRangeException("The dimension of mean must be larger than 1");
            }

            int populationSize = 4 + (int)Math.Floor(3 * Math.Log(nDim));  // # (eq. 48)

            int mu = populationSize;

            Vector<double> weightsPrime = Vector<double>.Build.Dense(populationSize);
            for (int i = 0; i < populationSize; i++)
            {
                weightsPrime[i] = Math.Log((populationSize + 1) / 2) - Math.Log(i + 1);
            }

            Vector<double> weightsPrimeMuEff = Vector<double>.Build.Dense(weightsPrime.Take(mu).ToArray());
            double muEff = Math.Pow(weightsPrimeMuEff.Sum(), 2) / Math.Pow(weightsPrimeMuEff.L2Norm(), 2);
            Vector<double> weightsPrimeMuEffMinus = Vector<double>.Build.Dense(weightsPrime.Skip(mu).ToArray());
            double muEffMinus = Math.Pow(weightsPrimeMuEffMinus.Sum(), 2) / Math.Pow(weightsPrimeMuEffMinus.L2Norm(), 2);

            int alphacCov = 2;
            double c1 = alphacCov / (Math.Pow(nDim + 1.3, 2) + muEff);
            double cmu = Math.Min(1 - c1, alphacCov * (muEff - 2 + (1 / muEff)) / (Math.Pow(nDim + 2, 2) + (alphacCov * muEff / 2)));
            if (!(c1 <= 1 - cmu))
            {
                throw new Exception("invalid learning rate for the rank-one update");
            }
            if (!(cmu <= 1 - c1))
            {
                throw new Exception("invalid learning rate for the rank-μ update");
            }

            double minAlpha = Math.Min(1 + (c1 / cmu), Math.Min(1 + (2 * muEffMinus / (muEff + 2)), (1 - c1 - cmu) / (nDim * cmu)));

            double positiveSum = weightsPrime.Where(x => x > 0).Sum();
            double negativeSum = Math.Abs(weightsPrime.Where(x => x < 0).Sum());

            Vector<double> weights = Vector<double>.Build.Dense(weightsPrime.Count);
            weightsPrime.CopyTo(weights);
            bool[] weightsIsNotNegative = weightsPrime.Select(x => x >= 0).ToArray();
            for (int i = 0; i < weights.Count; i++)
            {
                weights[i] = weightsIsNotNegative[i] ? 1 / positiveSum * weightsPrime[i] : minAlpha / negativeSum * weightsPrime[i];
            }
            int cm = 1;

            double cSigma = (muEff + 2) / (nDim + muEff + 5);
            double dSigma = 1 + (2 * Math.Max(0, Math.Sqrt((muEff - 1) / (nDim + 1)) - 1)) + cSigma;
            if (!(cSigma < 1))
            {
                throw new Exception("invalid learning rate for cumulation for the step-size control");
            }

            double cc = (4 + (muEff / nDim)) / (nDim + 4 + (2 * muEff / nDim));
            if (!(cc <= 1))
            {
                throw new Exception("invalid learning rate for cumulation for the rank-one update");
            }

            Dim = nDim;
            PopulationSize = populationSize;
            this.mu = mu;
            this.muEff = muEff;

            this.cc = cc;
            this.c1 = c1;
            this.cmu = cmu;
            this.cSigma = cSigma;
            this.dSigma = dSigma;
            this.cm = cm;

            this.chiN = Math.Sqrt(Dim) * (1.0 - (1.0 / (4.0 * Dim)) + 1.0 / (21.0 * (Math.Pow(Dim, 2))));

            this.weights = weights;

            this.pSigma = Vector<double>.Build.Dense(Dim, 0);
            this.pc = Vector<double>.Build.Dense(Dim, 0);

            this.mean = mean;
            this.C = Matrix<double>.Build.DenseIdentity(Dim, Dim);
            this.sigma = sigma;

            if (!(bounds == null || (bounds.RowCount == Dim && bounds.ColumnCount == 2)))
            {
                throw new Exception("bounds should be (n_dim, 2)-dim matrix");
            }
            this.bounds = bounds;
            this.nMaxResampling = nMaxResampling;

            this.g = 0;
            this.rng = new MathNet.Numerics.Random.Xorshift(seed);

            this.epsilon = 1e-8;
        }

        public void SetBounds(Matrix<double> bounds = null)
        {
            if (!(bounds == null || (bounds.RowCount == Dim && bounds.ColumnCount == 2)))
            {
                throw new Exception("bounds should be (n_dim, 2)-dim matrix");
            }
            this.bounds = bounds;
        }

        private Vector<double> SampleSolution()
        {
            if (B == null || D == null)
            {
                C = (C + C.Transpose()) / 2;
                MathNet.Numerics.LinearAlgebra.Factorization.Evd<double> evdC = C.Evd();
                Matrix<double> tmpD = evdC.D.PointwiseSqrt();
                tmpD += epsilon;
                this.D = tmpD;
                this.B = evdC.EigenVectors;
                Vector<double> BD2 = B * D.PointwiseSqrt().Diagonal();
                C = BD2 * B.Transpose();
            }

            Vector<double> z = Vector<double>.Build.Dense(Dim, Normal.Sample(rng, 0.0, 1.0));
            Matrix<double> y = B * D.Diagonal().ToColumnMatrix() * z.ToColumnMatrix();
            var x = mean + sigma * y;
            return x;
        }
    }
}
