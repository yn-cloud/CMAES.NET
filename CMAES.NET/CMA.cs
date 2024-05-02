using System;
using System.Linq;
using System.Collections.Generic;

using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;

namespace CMAESnet
{
    public class CMA
    {
        private readonly int _mu;
        private readonly double _mu_eff;
        private readonly double _cc;
        private readonly double _c1;
        private readonly double _cmu;
        private readonly double _c_sigma;
        private readonly double _d_sigma;
        private readonly int _cm;
        private readonly double _chi_n;
        private readonly Vector<double> _weights;
        private Vector<double> _p_sigma;
        private Vector<double> _pc;
        private Vector<double> _mean;
        private Matrix<double> _C;
        private double _sigma;
        private Vector<double> _D;
        private Matrix<double> _B;
        private Matrix<double> _bounds;
        private readonly int _n_max_resampling;
        private readonly Xorshift _rng;
        private readonly double _epsilon;
        private readonly double _tol_sigma;
        private readonly double _tol_C;

        /// <summary>
        /// A number of dimensions
        /// </summary>
        public int Dim { get; }
        /// <summary>
        /// A population size
        /// </summary>
        public int PopulationSize { get; private set; }
        /// <summary>
        /// Generation number which is monotonically incremented when multi-variate gaussian distribution is updated.
        /// </summary>
        public int Generation { get; private set; }

        /// <summary>
        /// CMA-ES stochastic optimizer class with ask-and-tell interface.
        /// </summary>
        /// <param name="mean">Initial mean vector of multi-variate gaussian distributions.</param>
        /// <param name="sigma">Initial standard deviation of covariance matrix.</param>
        /// <param name="bounds">(Optional) Lower and upper domain boundaries for each parameter, (n_dim, 2)-dim matrix.</param>
        /// <param name="nMaxResampling">(Optional) A maximum number of resampling parameters (default: 100).
        /// If all sampled parameters are infeasible, the last sampled one  will be clipped with lower and upper bounds.</param>
        /// <param name="seed">(Optional) A seed number.</param>
        /// <param name="tol_sigma">(Optional) Threshold for determining the convergence of sigma.</param>
        /// <param name="tol_C">(Optional) Threshold for determining the convergence of Covariance matrix.</param>
        public CMA(IList<double> mean, double sigma, Matrix<double> bounds = null, int nMaxResampling = 100, int seed = 0, double tol_sigma = 1e-4, double tol_C = 1e-4)
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
                weightsPrime[i] = Math.Log((populationSize + 1) / (double)2) - Math.Log(i + 1);
            }

            Vector<double> weightsPrimeMuEff = Vector<double>.Build.Dense(weightsPrime.Take(mu).ToArray());
            double mu_eff = Math.Pow(weightsPrimeMuEff.Sum(), 2) / Math.Pow(weightsPrimeMuEff.L2Norm(), 2);
            Vector<double> weightsPrimeMuEffMinus = Vector<double>.Build.Dense(weightsPrime.Skip(mu).ToArray());
            double muEffMinus = Math.Pow(weightsPrimeMuEffMinus.Sum(), 2) / Math.Pow(weightsPrimeMuEffMinus.L2Norm(), 2);

            int alphacCov = 2;
            double c1 = alphacCov / (Math.Pow(nDim + 1.3, 2) + mu_eff);
            double cmu = Math.Min(1 - c1, alphacCov * (mu_eff - 2 + (1 / mu_eff)) / (Math.Pow(nDim + 2, 2) + (alphacCov * mu_eff / 2)));
            if (!(c1 <= 1 - cmu))
            {
                throw new Exception("invalid learning rate for the rank-one update");
            }
            if (!(cmu <= 1 - c1))
            {
                throw new Exception("invalid learning rate for the rank-μ update");
            }

            double minAlpha = Math.Min(1 + (c1 / cmu), Math.Min(1 + (2 * muEffMinus / (mu_eff + 2)), (1 - c1 - cmu) / (nDim * cmu)));

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

            double c_sigma = (mu_eff + 2) / (nDim + mu_eff + 5);
            double d_sigma = 1 + (2 * Math.Max(0, Math.Sqrt((mu_eff - 1) / (nDim + 1)) - 1)) + c_sigma;
            if (!(c_sigma < 1))
            {
                throw new Exception("invalid learning rate for cumulation for the step-size control");
            }

            double cc = (4 + (mu_eff / nDim)) / (nDim + 4 + (2 * mu_eff / nDim));
            if (!(cc <= 1))
            {
                throw new Exception("invalid learning rate for cumulation for the rank-one update");
            }

            Dim = nDim;
            PopulationSize = populationSize;
            _mu = mu;
            _mu_eff = mu_eff;

            _cc = cc;
            _c1 = c1;
            _cmu = cmu;
            _c_sigma = c_sigma;
            _d_sigma = d_sigma;
            _cm = cm;

            _chi_n = Math.Sqrt(Dim) * (1.0 - (1.0 / (4.0 * Dim)) + 1.0 / (21.0 * (Math.Pow(Dim, 2))));

            _weights = weights;

            _p_sigma = Vector<double>.Build.Dense(Dim, 0);
            _pc = Vector<double>.Build.Dense(Dim, 0);

            _mean = Vector<double>.Build.DenseOfArray(mean.ToArray());
            _C = Matrix<double>.Build.DenseIdentity(Dim, Dim);
            _sigma = sigma;

            if (!(bounds == null || (bounds.RowCount == Dim && bounds.ColumnCount == 2)))
            {
                throw new Exception("bounds should be (n_dim, 2)-dim matrix");
            }
            _bounds = bounds;
            _n_max_resampling = nMaxResampling;

            Generation = 0;
            _rng = new Xorshift(seed);

            _epsilon = 1e-8;

            _tol_sigma = tol_sigma;
            _tol_C = tol_C;
        }

        /// <summary>
        /// Check if the covariance matrix and step size are below the threshold.
        /// </summary>
        /// <returns>Whether the step size of the covariance matrix is converged or not.</returns>
        public bool IsConverged()
        {
            return _sigma < _tol_sigma && _C.L2Norm() < _tol_C;
        }

        public void SetBounds(Matrix<double> bounds = null)
        {
            if (!(bounds == null || (bounds.RowCount == Dim && bounds.ColumnCount == 2)))
            {
                throw new Exception("bounds should be (n_dim, 2)-dim matrix");
            }
            _bounds = bounds;
        }

        /// <summary>
        /// Returns the next search vector based on the current covariance matrix.
        /// </summary>
        /// <returns>The next search vector.</returns>
        public Vector<double> Ask()
        {
            for (int i = 0; i < _n_max_resampling; i++)
            {
                Vector<double> x = SampleSolution();
                if (IsFeasible(x))
                    return x;
            }
            Vector<double> xNew = SampleSolution();
            xNew = RepairInfeasibleParams(xNew);
            return xNew;
        }

        /// <summary>
        /// The covariance matrix and step size are recalculated based on the search vectors and their results.
        /// </summary>
        /// <param name="solutions">Tuple's list of search vectors and result values.</param>
        public void Tell(List<Tuple<Vector<double>, double>> solutions)
        {
            if (solutions.Count != PopulationSize)
            {
                throw new ArgumentException("Must tell popsize-length solutions.");
            }

            Generation += 1;
            Tuple<Vector<double>, double>[] sortedSolutions = solutions.OrderBy(x => x.Item2).ToArray();

            // Sample new population of search_points, for k=1, ..., popsize
            Matrix<double> B = Matrix<double>.Build.Dense(_B.RowCount, _B.ColumnCount);
            Vector<double> D = Vector<double>.Build.Dense(_D.Count);
            if (_B == null || _D == null)
            {
                _C = (_C + _C.Transpose()) / 2;
                MathNet.Numerics.LinearAlgebra.Factorization.Evd<double> evd_C = _C.Evd();
                B = evd_C.EigenVectors;
                D = Vector<double>.Build.Dense(evd_C.EigenValues.PointwiseSqrt().Select(tmp => tmp.Real <= 0 ? _epsilon : tmp.Real).ToArray());
            }
            else
            {
                _B.CopyTo(B);
                _D.CopyTo(D);
            }
            _B = null;
            _D = null;

            Matrix<double> x_k = Matrix<double>.Build.DenseOfRowVectors(sortedSolutions.Select(x => x.Item1));
            Matrix<double> y_k = Matrix<double>.Build.Dense(sortedSolutions.Length, Dim);
            for (int i = 0; i < sortedSolutions.Length; i++)
            {
                y_k.SetRow(i, (x_k.Row(i) - _mean) / _sigma);
            }

            // Selection and recombination
            Vector<double>[] kk = y_k.EnumerateRows().Take(_mu).ToArray();
            Matrix<double> y_k_T = Matrix<double>.Build.Dense(Dim, kk.Length);
            for (int i = 0; i < kk.Length; i++)
            {
                y_k_T.SetColumn(i, kk[i]);
            }
            Vector<double> subWeights = Vector<double>.Build.Dense(_weights.Take(_mu).ToArray());
            Matrix<double> y_w_matrix = Matrix<double>.Build.Dense(y_k_T.RowCount, y_k_T.ColumnCount);
            for (int i = 0; i < y_w_matrix.RowCount; i++)
            {
                y_w_matrix.SetRow(i, y_k_T.Row(i).PointwiseMultiply(subWeights));
            }
            Vector<double> y_w = y_w_matrix.RowSums();
            _mean += _cm * _sigma * y_w;

            Vector<double> D_bunno1_diag = 1 / D;
            Matrix<double> D_bunno1_diagMatrix = Matrix<double>.Build.Dense(D_bunno1_diag.Count, D_bunno1_diag.Count);
            for (int i = 0; i < D_bunno1_diag.Count; i++)
            {
                D_bunno1_diagMatrix[i, i] = D_bunno1_diag[i];
            }
            Matrix<double> C_2 = B * D_bunno1_diagMatrix * B.Transpose();
            _p_sigma = ((1 - _c_sigma) * _p_sigma) + (Math.Sqrt(_c_sigma * (2 - _c_sigma) * _mu_eff) * C_2 * y_w);

            double norm_pSigma = _p_sigma.L2Norm();
            _sigma *= Math.Exp(_c_sigma / _d_sigma * ((norm_pSigma / _chi_n) - 1));
            double h_sigma_cond_left = norm_pSigma / Math.Sqrt(1 - Math.Pow(1 - _c_sigma, 2 * (Generation + 1)));
            double h_sigma_cond_right = (1.4 + (2 / (double)(Dim + 1))) * _chi_n;
            double h_sigma = h_sigma_cond_left < h_sigma_cond_right ? 1.0 : 0.0;

            _pc = ((1 - _cc) * _pc) + (h_sigma * Math.Sqrt(_cc * (2 - _cc) * _mu_eff) * y_w);

            Vector<double> w_io = Vector<double>.Build.Dense(_weights.Count, 1);
            Vector<double> w_iee = (C_2 * y_k.Transpose()).ColumnNorms(2).PointwisePower(2);
            for (int i = 0; i < _weights.Count; i++)
            {
                if (_weights[i] >= 0)
                {
                    w_io[i] = _weights[i] * 1;
                }
                else
                {
                    w_io[i] = _weights[i] * Dim / (w_iee[i] + _epsilon);
                }
            }

            double delta_h_sigma = (1 - h_sigma) * _cc * (2 - _cc);
            if (!(delta_h_sigma <= 1))
            {
                throw new Exception("invalid value of delta_h_sigma");
            }

            Matrix<double> rank_one = _pc.OuterProduct(_pc);
            Matrix<double> rank_mu = Matrix<double>.Build.Dense(y_k.ColumnCount, y_k.ColumnCount, 0);
            for (int i = 0; i < w_io.Count; i++)
            {
                rank_mu += w_io[i] * y_k.Row(i).OuterProduct(y_k.Row(i));
            }
            _C = ((1 + (_c1 * delta_h_sigma) - _c1 - (_cmu * _weights.Sum())) * _C) + (_c1 * rank_one) + (_cmu * rank_mu);
        }

        private Vector<double> RepairInfeasibleParams(Vector<double> param)
        {
            if (_bounds == null)
            {
                return param;
            }
            Vector<double> newParam = param.PointwiseMaximum(_bounds.Column(0));
            newParam = newParam.PointwiseMinimum(_bounds.Column(1));
            return newParam;
        }

        private bool IsFeasible(Vector<double> param)
        {
            if (_bounds == null)
            {
                return true;
            }
            bool isCorrectLower = true;
            bool isCorrectUpper = true;
            for (int i = 0; i < param.Count; i++)
            {
                isCorrectLower &= param[i] >= _bounds[i, 0];
                isCorrectUpper &= param[i] <= _bounds[i, 1];
            }
            return isCorrectLower & isCorrectUpper;
        }

        private Vector<double> SampleSolution()
        {
            if (_B == null || _D == null)
            {
                _C = (_C + _C.Transpose()) / 2;
                MathNet.Numerics.LinearAlgebra.Factorization.Evd<double> evd_C = _C.Evd();
                Matrix<double> B = evd_C.EigenVectors;
                Vector<double> D = Vector<double>.Build.Dense(evd_C.EigenValues.PointwiseSqrt().Select(tmp => tmp.Real <= 0 ? _epsilon : tmp.Real).ToArray());
                _B = B;
                _D = D;
                Matrix<double> D2diagonal = Matrix<double>.Build.DenseDiagonal(D.Count, 1);
                Vector<double> Dpow2 = D.PointwisePower(2);
                for (int i = 0; i < D2diagonal.RowCount; i++)
                {
                    D2diagonal[i, i] = Dpow2[i];
                }
                Matrix<double> BD2 = B * D2diagonal;
                _C = BD2 * B.Transpose();
            }

            Vector<double> z = Vector<double>.Build.Dense(Dim);
            for (int i = 0; i < z.Count; i++)
            {
                z[i] = Normal.Sample(_rng, 0, 1);
            }
            Matrix<double> Ddiagonal = Matrix<double>.Build.DenseDiagonal(_D.Count, 1);
            for (int i = 0; i < Ddiagonal.RowCount; i++)
            {
                Ddiagonal[i, i] = _D[i];
            }
            Matrix<double> y = _B * Ddiagonal * z.ToColumnMatrix();
            Vector<double> x = _mean + (_sigma * y.Column(0));
            return x;
        }
    }
}
