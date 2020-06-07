﻿using System;
using System.Collections.Generic;
using System.Linq;

using MathNet.Numerics.LinearAlgebra;

namespace CMAESnet
{
    public class CMAESOptimizer
    {
        private readonly CMA cma;
        private readonly Func<IList<double>, double> function;
        private readonly int maxIteration;

        public Vector<double> ResultVector { get; private set; }
        public double ResultValue { get; private set; }

        public CMAESOptimizer(Func<IList<double>, double> function, IList<double> initnial, double sigma, int randSeed = 0)
        {
            this.function = function;
            maxIteration = initnial.Count * 200;

            cma = new CMA(initnial, sigma, seed: randSeed);

            ResultValue = double.MaxValue;
        }

        public CMAESOptimizer(Func<IList<double>, double> function, IList<double> initial, double sigma, IList<double> lowerBounds, IList<double> upperBounds)
        {
            if (initial.Count != lowerBounds.Count)
            {
                throw new ArgumentException("length of lowerBounds must be equal to that of initial.");
            }
            if (initial.Count != upperBounds.Count)
            {
                throw new ArgumentException("length of upperBounds must be equal to that of initial");
            }

            this.function = function;
            maxIteration = initial.Count;

            Matrix<double> bounds = Matrix<double>.Build.Dense(initial.Count, 2);
            bounds.SetColumn(0, lowerBounds.ToArray());
            bounds.SetColumn(1, upperBounds.ToArray());

            cma = new CMA(initial, sigma, bounds);

            ResultValue = double.MaxValue;
        }

        public void Optimize()
        {
            Vector<double> xBest = null;
            double yBest = double.MaxValue;
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
                double yCurrentBest = solutions.Min(x => x.Item2);
                Vector<double> xCurrentBest = solutions.Where(x => x.Item2 == yCurrentBest).FirstOrDefault().Item1;

                if (xBest == null || yBest == double.MaxValue)
                {
                    isConverged = false;
                }
                else
                {
                    double xDiff = (xBest - xCurrentBest).L2Norm();
                    double yDiff = Math.Abs(yBest - yCurrentBest);

                    isConverged = cma.IsConverged();
                }

                xBest = yCurrentBest < yBest ? xCurrentBest : xBest;
                yBest = yCurrentBest < yBest ? yCurrentBest : yBest;

                if (isConverged)
                {
                    break;
                }
            }

            if (!isConverged)
            {
                Console.WriteLine("Reached max iteration.");
            }

            ResultVector = xBest;
            ResultValue = yBest;
        }
    }
}
