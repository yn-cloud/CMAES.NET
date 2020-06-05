using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CMAESnet
{
    public class CMAESOptimizer
    {
        private readonly CMA cma;
        private readonly Func<IList<double>, double> function;
        private readonly int maxIteration;
        private readonly double tolX;
        private readonly double tolFun;

        public Vector<double> ResultVector { get; private set; }
        public double ResultValue { get; private set; }

        public CMAESOptimizer(Func<IList<double>, double> function, IList<double> initnial, double sigma)
        {
            this.function = function;
            maxIteration = initnial.Count * 200;
            tolX = 1e-4;
            tolFun = 1e-4;

            cma = new CMA(initnial, sigma);
        }

        public CMAESOptimizer(Func<IList<double>, double> function, IList<double> initial, double sigma, IList<double> lowerBounds, IList<double> upperBounds)
        {
            if (initial.Count != lowerBounds.Count)
            {
                throw new ArgumentException("length ouf lowerBounds must be equal to that of initial.");
            }
            if (initial.Count != upperBounds.Count)
            {
                throw new ArgumentException("length ouf upperBounds must be equal to that of initial");
            }

            this.function = function;
            maxIteration = initial.Count;

            Matrix<double> bounds = Matrix<double>.Build.Dense(initial.Count, 2);
            bounds.SetColumn(0, lowerBounds.ToArray());
            bounds.SetColumn(1, upperBounds.ToArray());

            tolX = 1e-4;
            tolFun = 1e-4;

            cma = new CMA(initial, sigma, bounds);
        }

        public void Optimize()
        {
            Vector<double> xPrevious = null;
            double yPrevious = double.MaxValue;
            bool isConverged = false;

            for (int generation = 0; generation < maxIteration; generation++)
            {
                List<Tuple<Vector<double>, double>> solutions = new List<Tuple<Vector<double>, double>>();
                for (int i = 0; i < cma.PopulationSize; i++)
                {
                    Vector<double> x = cma.Ask();
                    double value = function(x);
                    solutions.Add(new Tuple<Vector<double>, double>(x, value));
                }
                cma.Tell(solutions);
                double yBest = solutions.Min(x => x.Item2);
                Vector<double> xBest = solutions.Where(x => x.Item2 == yBest).FirstOrDefault().Item1;

                if (xPrevious == null || yPrevious == double.MaxValue)
                {
                    isConverged = false;
                }
                else
                {
                    double xDiff = (xPrevious - xBest).L2Norm();
                    double yDiff = Math.Abs(yPrevious - yBest);

                    isConverged = xDiff < tolX && yDiff < tolFun;
                }

                xPrevious = yBest < yPrevious ? xBest : xPrevious;
                yPrevious = yBest < yPrevious ? yBest : yPrevious;

                if (isConverged) { break; }
            }

            if (!isConverged)
            {
                Console.WriteLine("Reached max iteration.");
            }

            ResultVector = xPrevious;
            ResultValue = yPrevious;
        }
    }
}
