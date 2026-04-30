/**
 * Statistical calculation utility functions for benchmark data
 */

import { ViolinDataPoint } from '@/types/benchmark.types';

/**
 * Calculate quartiles and outliers for a dataset
 * @param values - Array of numerical values
 * @returns Quartile statistics including min, q1, median, q3, max, and outliers
 */
export function calculateQuartiles(
  values: number[]
): ViolinDataPoint["quartiles"] {
  if (values.length === 0) {
    return { min: 0, q1: 0, median: 0, q3: 0, max: 0, outliers: [] };
  }

  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;

  const q1Index = Math.floor(n * 0.25);
  const medianIndex = Math.floor(n * 0.5);
  const q3Index = Math.floor(n * 0.75);

  const q1 = sorted[q1Index] ?? 0;
  const median =
    n % 2 === 0
      ? ((sorted[medianIndex - 1] ?? 0) + (sorted[medianIndex] ?? 0)) / 2
      : (sorted[medianIndex] ?? 0);
  const q3 = sorted[q3Index] ?? 0;

  const iqr = q3 - q1;
  const lowerFence = q1 - 1.5 * iqr;
  const upperFence = q3 + 1.5 * iqr;

  const outliers = sorted.filter(
    (value) => value < lowerFence || value > upperFence
  );
  const filteredValues = sorted.filter(
    (value) => value >= lowerFence && value <= upperFence
  );

  const min =
    filteredValues.length > 0 ? Math.min(...filteredValues) : (sorted[0] ?? 0);
  const max =
    filteredValues.length > 0
      ? Math.max(...filteredValues)
      : (sorted[sorted.length - 1] ?? 0);

  return { min, q1, median, q3, max, outliers };
}

/**
 * Calculate basic statistics for a dataset
 * @param values - Array of numerical values
 * @returns Statistics including mean, standard deviation, and count
 */
export function calculateStats(values: number[]): ViolinDataPoint["stats"] {
  if (values.length === 0) {
    return { mean: 0, std: 0, count: 0 };
  }

  const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
  const variance =
    values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
    values.length;
  const std = Math.sqrt(variance);

  return { mean, std, count: values.length };
}

/**
 * Gaussian kernel function for kernel density estimation
 * @param x - Input value
 * @returns Gaussian kernel value
 */
export function gaussianKernel(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/**
 * Calculate kernel density estimation for a dataset
 * @param values - Array of numerical values
 * @param bandwidth - Optional bandwidth parameter (auto-calculated if not provided)
 * @returns Array of density points with value and density
 */
export function calculateKernelDensity(
  values: number[],
  bandwidth?: number
): { value: number; density: number }[] {
  if (values.length === 0) return [];

  const sortedValues = [...values].sort((a, b) => a - b);
  const min = sortedValues[0] ?? 0;
  const max = sortedValues[sortedValues.length - 1] ?? 0;
  const range = max - min;

  // Auto-calculate bandwidth using Silverman's rule of thumb
  const n = values.length;
  const std = calculateStats(values).std;
  const h = bandwidth ?? 1.06 * std * Math.pow(n, -1 / 5);

  // Create evaluation points
  const numPoints = 100;
  const step = range / (numPoints - 1);
  const densityPoints: { value: number; density: number }[] = [];

  for (let i = 0; i < numPoints; i++) {
    const x = min + i * step;
    let density = 0;

    // Calculate kernel density at point x
    for (const value of values) {
      density += gaussianKernel((x - value) / h);
    }

    density = density / (n * h);
    densityPoints.push({ value: x, density });
  }

  return densityPoints;
}
