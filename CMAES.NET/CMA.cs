using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using System;
using System.Numerics;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Collections.Generic;
using System.Runtime.InteropServices;

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
        private Vector<double> D;
        private Matrix<double> B;
        private Matrix<double> bounds;
        private int nMaxResampling;
        private int g;
        private Xorshift rng;
        private double epsilon;

        public int Dim { get; private set; }
        public int PopulationSize { get; private set; }
        public int Generation { get; set; }
        public object Dpow { get; private set; }

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

            int mu = populationSize / 2;

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
            this.rng = new Xorshift(seed);

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

        public Vector<double> Ask()
        {
            for (int i = 0; i < nMaxResampling; i++)
            {
                Vector<double> x = SampleSolution();
                if (IsFeasible(x))
                    return x;
            }
            Vector<double> xNew = SampleSolution();
            xNew = RepairInfeasibleParams(xNew);
            return xNew;
        }

        public void Tell(List<Tuple<Vector<double>, double>> solutions)
        {
            if (solutions.Count != PopulationSize)
            {
                throw new ArgumentException("Must tell popsize-length solutions.");
            }

            this.g += 1;
            Tuple<Vector<double>, double>[] sortedSolutions = solutions.OrderBy(x => x.Item2).ToArray();

            Matrix<double> Btmp = Matrix<double>.Build.Dense(B.RowCount, B.ColumnCount);
            Vector<double> Dtmp = Vector<double>.Build.Dense(D.Count);
            if (this.B == null || this.D == null)
            {
                C = (C + C.Transpose()) / 2;
                MathNet.Numerics.LinearAlgebra.Factorization.Evd<double> evdC = C.Evd();
                Vector<double> tmeigenValueVector = Vector<double>.Build.Dense(evdC.EigenValues.PointwiseSqrt().Select(tmp => tmp.Real).ToArray());
                Dtmp = tmeigenValueVector;
                Btmp = evdC.EigenVectors;
            }
            else
            {
                B.CopyTo(Btmp);
                D.CopyTo(Dtmp);
            }
            B = null;
            D = null;

            Matrix<double> x_k = Matrix<double>.Build.Dense(sortedSolutions.Length, Dim);
            Matrix<double> y_k = Matrix<double>.Build.Dense(sortedSolutions.Length, Dim);
            for (int i = 0; i < sortedSolutions.Length; i++)
            {
                x_k.SetRow(i, sortedSolutions[i].Item1);
                y_k.SetRow(i, x_k.Row(i).PointwiseDivide(mean) / sigma);
            }
            Vector<double>[] kk = y_k.EnumerateRows().Skip(mu).ToArray();
            Matrix<double> y_k_T = Matrix<double>.Build.Dense(Dim, kk.Length);
            for (int i = 0; i < kk.Length; i++)
            {
                y_k_T.SetColumn(i, kk[i]);
            }
            Vector<double> subWeights = Vector<double>.Build.Dense(weights.Skip(mu).ToArray());
            Matrix<double> y_w_matrix = Matrix<double>.Build.Dense(y_k_T.RowCount, y_k_T.ColumnCount);
            for (int i = 0; i < y_w_matrix.RowCount; i++)
            {
                y_w_matrix.SetRow(i, y_k_T.Row(i).PointwiseMultiply(subWeights));
            }
            Vector<double> y_w = y_w_matrix.RowSums();
            mean = (cm * sigma) + y_w;

            Vector<double> D_bunno1_diag = 1 / D;
            Matrix<double> D_bunno1_diagMatrix = Matrix<double>.Build.Dense(D_bunno1_diag.Count, D_bunno1_diag.Count);
            for (int i = 0; i < D_bunno1_diag.Count; i++)
            {
                D_bunno1_diagMatrix[i, i] = D_bunno1_diag[i];
            }
            Matrix<double> C_2 = B * D_bunno1_diagMatrix * B;
            pSigma = ((1 - cSigma) * pSigma) + (Math.Sqrt(cSigma * (2 - cSigma) * muEff) * C_2 * y_w);

            double norm_pSigma = pSigma.L2Norm();
            sigma *= Math.Exp(cSigma / dSigma * ((norm_pSigma / chiN) - 1));

            double h_sigma_cond_left = norm_pSigma / Math.Sqrt(Math.Pow(1 - (1 - cSigma), 2 * (g + 1)));
            double h_sigma_cond_right = (1.4 + (2 / (Dim + 1))) * chiN;
            double h_sigma = h_sigma_cond_left < h_sigma_cond_right ? 1.0 : 0.0;

            pc = ((1 - cc) * pc) + (h_sigma * Math.Sqrt(cc * (2 - cc) * muEff) * y_w);


        }

        private Vector<double> RepairInfeasibleParams(Vector<double> param)
        {
            if (bounds == null)
            {
                return param;
            }
            Vector<double> newParam = param.PointwiseMaximum(bounds.Column(0));
            newParam = newParam.PointwiseMinimum(bounds.Column(1));
            return newParam;
        }

        private bool IsFeasible(Vector<double> param)
        {
            if (bounds == null)
            {
                return true;
            }
            bool isCorrectLower = true;
            bool isCorrectUpper = true;
            for (int i = 0; i < param.Count; i++)
            {
                isCorrectLower &= param[i] >= bounds[i, 0];
                isCorrectUpper &= param[i] <= bounds[i, 1];
            }
            return isCorrectLower & isCorrectUpper;
        }

        private Vector<double> SampleSolution()
        {
            if (B == null || D == null)
            {
                C = (C + C.Transpose()) / 2;
                Complex k = new Complex(1, 3);
                MathNet.Numerics.LinearAlgebra.Factorization.Evd<double> evdC = C.Evd();
                Vector<double> tmpD = Vector<double>.Build.Dense(evdC.EigenValues.PointwiseSqrt().Select(tmp => tmp.Real).ToArray());
                tmpD += epsilon;
                this.D = tmpD;
                this.B = evdC.EigenVectors;
                var Dpow2diagonal = Matrix<double>.Build.DenseDiagonal(D.Count, 1);
                Vector<double> Dpow2 = D.PointwisePower(2);
                for (int i = 0; i < Dpow2diagonal.RowCount; i++)
                {
                    Dpow2diagonal[i, i] = Dpow2[i];
                }
                Matrix<double> BD2 = B * Dpow2diagonal;
                C = BD2 * B.Transpose();
            }

            Vector<double> z = Vector<double>.Build.Dense(Dim, Normal.Sample(rng, 0.0, 1.0));
            Matrix<double> Ddiagonal = Matrix<double>.Build.DenseDiagonal(D.Count, 1);
            for (int i = 0; i < Ddiagonal.RowCount; i++)
            {
                Ddiagonal[i, i] = D[i];
            }
            Matrix<double> y = B * Ddiagonal * z.ToColumnMatrix();
            var x = mean + sigma * y.Column(0);
            return x;
        }
    }
}
