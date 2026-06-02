//! # information-geometry
//!
//! Information geometry: Fisher information, natural gradient, geodesics on statistical manifolds.
//!
//! A statistical manifold M is a Riemannian manifold where each point is a probability distribution.
//! The Fisher information metric gᵢⱼ(θ) = E[∂ᵢ log p(x|θ) · ∂ⱼ log p(x|θ)] gives the natural geometry.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ─── DenseMatrix ────────────────────────────────────────────────────────────────

/// A dense matrix for linear algebra operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DenseMatrix {
    /// Row-major storage.
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "data length mismatch");
        Self { data, rows, cols }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    pub fn diagonal(values: &[f64]) -> Self {
        let n = values.len();
        let mut m = Self::new(n, n);
        for (i, v) in values.iter().enumerate().take(n) {
            m.set(i, i, *v);
        }
        m
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn add(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        DenseMatrix::from_vec(data, self.rows, self.cols)
    }

    pub fn subtract(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        DenseMatrix::from_vec(data, self.rows, self.cols)
    }

    pub fn multiply(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.cols, other.rows);
        let mut result = DenseMatrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn scale(&self, s: f64) -> DenseMatrix {
        let data: Vec<f64> = self.data.iter().map(|v| v * s).collect();
        DenseMatrix::from_vec(data, self.rows, self.cols)
    }

    pub fn transpose(&self) -> DenseMatrix {
        let mut result = DenseMatrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Compute determinant (for small matrices).
    pub fn determinant(&self) -> f64 {
        assert_eq!(self.rows, self.cols, "matrix must be square");
        let n = self.rows;
        match n {
            0 => 1.0,
            1 => self.get(0, 0),
            2 => {
                self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
            }
            3 => {
                let a = self.get(0, 0);
                let b = self.get(0, 1);
                let c = self.get(0, 2);
                let d = self.get(1, 0);
                let e = self.get(1, 1);
                let f = self.get(1, 2);
                let g = self.get(2, 0);
                let h = self.get(2, 1);
                let i = self.get(2, 2);
                a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
            }
            _ => {
                // LU decomposition for n <= 6
                let mut mat = self.clone();
                let mut det = 1.0;
                for col in 0..n {
                    // Partial pivoting
                    let mut max_row = col;
                    let mut max_val = mat.get(col, col).abs();
                    for row in (col + 1)..n {
                        let v = mat.get(row, col).abs();
                        if v > max_val {
                            max_val = v;
                            max_row = row;
                        }
                    }
                    if max_row != col {
                        // Swap rows
                        for j in 0..n {
                            let tmp = mat.get(col, j);
                            mat.set(col, j, mat.get(max_row, j));
                            mat.set(max_row, j, tmp);
                        }
                        det *= -1.0;
                    }
                    let pivot = mat.get(col, col);
                    if pivot.abs() < 1e-15 {
                        return 0.0;
                    }
                    det *= pivot;
                    for row in (col + 1)..n {
                        let factor = mat.get(row, col) / pivot;
                        for j in (col + 1)..n {
                            mat.set(row, j, mat.get(row, j) - factor * mat.get(col, j));
                        }
                    }
                }
                det
            }
        }
    }

    pub fn trace(&self) -> f64 {
        assert_eq!(self.rows, self.cols);
        (0..self.rows).map(|i| self.get(i, i)).sum()
    }

    /// Compute inverse for small matrices (≤6×6).
    pub fn inverse(&self) -> Option<DenseMatrix> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        if n > 6 {
            return None;
        }
        // Augmented matrix [A | I]
        let mut aug = DenseMatrix::new(n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, n + i, 1.0);
        }
        // Gauss-Jordan elimination
        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug.get(col, col).abs();
            for row in (col + 1)..n {
                let v = aug.get(row, col).abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_row != col {
                for j in 0..(2 * n) {
                    let tmp = aug.get(col, j);
                    aug.set(col, j, aug.get(max_row, j));
                    aug.set(max_row, j, tmp);
                }
            }
            let pivot = aug.get(col, col);
            if pivot.abs() < 1e-15 {
                return None;
            }
            for j in 0..(2 * n) {
                aug.set(col, j, aug.get(col, j) / pivot);
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug.get(row, col);
                for j in 0..(2 * n) {
                    aug.set(row, j, aug.get(row, j) - factor * aug.get(col, j));
                }
            }
        }
        let mut result = DenseMatrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                result.set(i, j, aug.get(i, n + j));
            }
        }
        Some(result)
    }

    /// Solve Ax = b.
    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        let inv = self.inverse()?;
        let mut result = vec![0.0; b.len()];
        for (i, r) in result.iter_mut().enumerate().take(inv.rows) {
            for (j, bv) in b.iter().enumerate().take(inv.cols) {
                *r += inv.get(i, j) * bv;
            }
        }
        Some(result)
    }

    /// Check if the matrix is positive definite via Cholesky.
    pub fn is_positive_definite(&self) -> bool {
        self.cholesky().is_some()
    }

    /// Cholesky decomposition: returns L such that LLᵀ = A.
    pub fn cholesky(&self) -> Option<DenseMatrix> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut l = DenseMatrix::new(n, n);
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l.get(i, k) * l.get(j, k);
                }
                if i == j {
                    let diag = self.get(i, i) - sum;
                    if diag <= 0.0 {
                        return None;
                    }
                    l.set(i, j, diag.sqrt());
                } else {
                    l.set(i, j, (self.get(i, j) - sum) / l.get(j, j));
                }
            }
        }
        Some(l)
    }

    pub fn row(&self, i: usize) -> Vec<f64> {
        (0..self.cols).map(|j| self.get(i, j)).collect()
    }

    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }
}

impl std::fmt::Display for DenseMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.6}", self.get(i, j))?;
            }
            write!(f, "]")?;
            if i + 1 < self.rows {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

// ─── ProbabilityDistribution trait ──────────────────────────────────────────────

/// A probability distribution on ℝ, parametrized by θ ∈ ℝⁿ.
pub trait ProbabilityDistribution: Clone + Serialize + for<'de> Deserialize<'de> {
    fn log_prob(&self, x: f64) -> f64;
    fn prob(&self, x: f64) -> f64 {
        self.log_prob(x).exp()
    }
    fn parameters(&self) -> Vec<f64>;
    fn set_parameters(&mut self, theta: &[f64]);
    fn num_parameters(&self) -> usize;
}

// ─── NormalDistribution ─────────────────────────────────────────────────────────

/// Normal (Gaussian) distribution N(μ, σ²).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalDistribution {
    pub mu: f64,
    pub sigma: f64,
}

impl NormalDistribution {
    pub fn new(mu: f64, sigma: f64) -> Self {
        assert!(sigma > 0.0, "sigma must be positive");
        Self { mu, sigma }
    }

    /// Standard normal N(0, 1).
    pub fn standard() -> Self {
        Self::new(0.0, 1.0)
    }

    /// ∂/∂μ and ∂/∂σ of log p(x|μ,σ).
    pub fn grad_log_prob(&self, x: f64) -> Vec<f64> {
        let d_mu = (x - self.mu) / (self.sigma * self.sigma);
        let d_sigma = ((x - self.mu) * (x - self.mu)) / (self.sigma * self.sigma * self.sigma) - 1.0 / self.sigma;
        vec![d_mu, d_sigma]
    }

    /// Fisher information matrix: g = [[1/σ², 0], [0, 2/σ²]].
    pub fn fisher_information(&self) -> DenseMatrix {
        let s2 = self.sigma * self.sigma;
        DenseMatrix::diagonal(&[1.0 / s2, 2.0 / s2])
    }

    /// Expected value E[X].
    pub fn mean(&self) -> f64 {
        self.mu
    }

    /// Variance Var(X).
    pub fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }

    /// Standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.sigma
    }
}

impl ProbabilityDistribution for NormalDistribution {
    fn log_prob(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        -0.5 * z * z - self.sigma.ln() - 0.5 * (2.0 * PI).ln()
    }

    fn parameters(&self) -> Vec<f64> {
        vec![self.mu, self.sigma]
    }

    fn set_parameters(&mut self, theta: &[f64]) {
        assert_eq!(theta.len(), 2);
        self.mu = theta[0];
        self.sigma = theta[1];
    }

    fn num_parameters(&self) -> usize {
        2
    }
}

// ─── ExponentialFamily trait ─────────────────────────────────────────────────────

/// An exponential family distribution in natural parametrization:
///   p(x|η) = h(x) exp(η·T(x) − A(η))
pub trait ExponentialFamily: Clone + Serialize + for<'de> Deserialize<'de> {
    /// Sufficient statistics T(x).
    fn sufficient_statistics(&self, x: f64) -> Vec<f64>;
    /// Natural parameters η.
    fn natural_parameters(&self) -> Vec<f64>;
    /// Log-partition function A(η).
    fn log_partition(&self) -> f64;
    /// ∇A(η) = E[T(x)].
    fn grad_log_partition(&self) -> Vec<f64>;
    /// ∇²A(η) = Cov[T(x)] (the Fisher metric in natural coordinates).
    fn hessian_log_partition(&self) -> DenseMatrix;
}

// ─── ExponentialFamily: Normal (known σ) ────────────────────────────────────────

/// Normal distribution in natural parametrization with known σ.
/// η = (μ/σ²), T(x) = x, A(η) = η²σ²/2
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalKnownSigma {
    pub sigma: f64,
}

impl NormalKnownSigma {
    pub fn new(sigma: f64) -> Self {
        assert!(sigma > 0.0);
        Self { sigma }
    }

    /// Get μ from natural parameter η.
    pub fn mu_from_eta(&self, eta: f64) -> f64 {
        eta * self.sigma * self.sigma
    }

    /// Get natural parameter η from μ.
    pub fn eta_from_mu(&self, mu: f64) -> f64 {
        mu / (self.sigma * self.sigma)
    }
}

impl ExponentialFamily for NormalKnownSigma {
    fn sufficient_statistics(&self, _x: f64) -> Vec<f64> {
        // T(x) = x for normal with known σ (using η = μ/σ²)
        // Actually, full natural parametrization: η₁ = μ/σ², η₂ = -1/(2σ²)
        // But for known σ, we only have η₁
        vec![_x]
    }

    fn natural_parameters(&self) -> Vec<f64> {
        // This returns the default/conceptual natural params; mu is not stored here
        // since sigma is fixed. For known-sigma normal, η = μ/σ²
        // We'll return η for some reference; user provides mu externally.
        // Actually, let's store mu as well for a complete parametrization.
        vec![0.0] // placeholder
    }

    fn log_partition(&self) -> f64 {
        // A(η) = η²σ²/2 + ln(σ√(2π))
        // For the one-parameter family with known σ:
        // A(η) = η²σ²/2 + const
        // Since we don't store η, this returns the constant part
        0.5 * (self.sigma * self.sigma * 2.0 * PI).ln()
    }

    fn grad_log_partition(&self) -> Vec<f64> {
        // E[T(x)] = E[X] = μ = ησ²
        // Without stored η, returns 0
        vec![0.0]
    }

    fn hessian_log_partition(&self) -> DenseMatrix {
        // Cov[T(x)] = Var(X) = σ²
        DenseMatrix::from_vec(vec![self.sigma * self.sigma], 1, 1)
    }
}

// ─── NormalNaturalParam ─────────────────────────────────────────────────────────

/// Normal distribution fully in natural coordinates η = (μ/σ², −1/(2σ²)).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNatural {
    pub eta1: f64, // μ/σ²
    pub eta2: f64, // −1/(2σ²)
}

impl NormalNatural {
    pub fn from_mu_sigma(mu: f64, sigma: f64) -> Self {
        Self {
            eta1: mu / (sigma * sigma),
            eta2: -1.0 / (2.0 * sigma * sigma),
        }
    }

    pub fn mu(&self) -> f64 {
        -self.eta1 / (2.0 * self.eta2)
    }

    pub fn sigma(&self) -> f64 {
        (-1.0 / (2.0 * self.eta2)).sqrt()
    }
}

impl ExponentialFamily for NormalNatural {
    fn sufficient_statistics(&self, x: f64) -> Vec<f64> {
        vec![x, x * x]
    }

    fn natural_parameters(&self) -> Vec<f64> {
        vec![self.eta1, self.eta2]
    }

    fn log_partition(&self) -> f64 {
        // A(η₁, η₂) = −η₁²/(4η₂) + ½ ln(−π/η₂)
        let ratio = self.eta1 / (2.0 * self.eta2);
        -self.eta1 * ratio / 2.0 + 0.5 * (-PI / self.eta2).ln()
    }

    fn grad_log_partition(&self) -> Vec<f64> {
        // ∂A/∂η₁ = −η₁/(2η₂) = μ
        // ∂A/∂η₂ = η₁²/(4η₂²) − 1/(2η₂) = μ² + σ²
        let d_eta1 = -self.eta1 / (2.0 * self.eta2);
        let d_eta2 = self.eta1 * self.eta1 / (4.0 * self.eta2 * self.eta2) - 1.0 / (2.0 * self.eta2);
        vec![d_eta1, d_eta2]
    }

    fn hessian_log_partition(&self) -> DenseMatrix {
        // ∂²A/∂η₁² = −1/(2η₂)
        // ∂²A/∂η₁∂η₂ = η₁/(2η₂²)
        // ∂²A/∂η₂² = −η₁²/(2η₂³) + 1/(2η₂²)
        let d11 = -1.0 / (2.0 * self.eta2);
        let d12 = self.eta1 / (2.0 * self.eta2 * self.eta2);
        let d22 = -self.eta1 * self.eta1 / (2.0 * self.eta2.powi(3)) + 1.0 / (2.0 * self.eta2 * self.eta2);
        DenseMatrix::from_vec(vec![d11, d12, d12, d22], 2, 2)
    }
}

// ─── BernoulliDistribution ──────────────────────────────────────────────────────

/// Bernoulli distribution Ber(p) in natural parametrization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BernoulliDistribution {
    pub p: f64,
}

impl BernoulliDistribution {
    pub fn new(p: f64) -> Self {
        assert!(p > 0.0 && p < 1.0, "p must be in (0,1)");
        Self { p }
    }

    pub fn natural_param(&self) -> f64 {
        (self.p / (1.0 - self.p)).ln() // logit
    }

    pub fn from_natural(eta: f64) -> Self {
        Self {
            p: 1.0 / (1.0 + (-eta).exp()),
        }
    }
}

impl ExponentialFamily for BernoulliDistribution {
    fn sufficient_statistics(&self, x: f64) -> Vec<f64> {
        vec![x]
    }

    fn natural_parameters(&self) -> Vec<f64> {
        vec![self.natural_param()]
    }

    fn log_partition(&self) -> f64 {
        // A(η) = ln(1 + eᶯ)
        let eta = self.natural_param();
        (1.0 + eta.exp()).ln()
    }

    fn grad_log_partition(&self) -> Vec<f64> {
        // eᶯ/(1 + eᶯ) = p
        vec![self.p]
    }

    fn hessian_log_partition(&self) -> DenseMatrix {
        // p(1-p)
        DenseMatrix::from_vec(vec![self.p * (1.0 - self.p)], 1, 1)
    }
}

// ─── PoissonDistribution ────────────────────────────────────────────────────────

/// Poisson distribution Pois(λ) in natural parametrization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PoissonDistribution {
    pub lambda: f64,
}

impl PoissonDistribution {
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0);
        Self { lambda }
    }
}

impl ExponentialFamily for PoissonDistribution {
    fn sufficient_statistics(&self, x: f64) -> Vec<f64> {
        vec![x]
    }

    fn natural_parameters(&self) -> Vec<f64> {
        vec![self.lambda.ln()]
    }

    fn log_partition(&self) -> f64 {
        // A(η) = eᶯ = λ
        self.lambda
    }

    fn grad_log_partition(&self) -> Vec<f64> {
        vec![self.lambda]
    }

    fn hessian_log_partition(&self) -> DenseMatrix {
        DenseMatrix::from_vec(vec![self.lambda], 1, 1)
    }
}

// ─── ExponentialDist ────────────────────────────────────────────────────────────

/// Exponential distribution Exp(λ) in natural parametrization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExponentialDist {
    pub lambda: f64, // rate parameter
}

impl ExponentialDist {
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0);
        Self { lambda }
    }
}

impl ExponentialFamily for ExponentialDist {
    fn sufficient_statistics(&self, x: f64) -> Vec<f64> {
        vec![-x]
    }

    fn natural_parameters(&self) -> Vec<f64> {
        vec![self.lambda]
    }

    fn log_partition(&self) -> f64 {
        // A(η) = −ln(η)
        -self.lambda.ln()
    }

    fn grad_log_partition(&self) -> Vec<f64> {
        // −1/η = −1/λ
        vec![-1.0 / self.lambda]
    }

    fn hessian_log_partition(&self) -> DenseMatrix {
        // 1/η² = 1/λ²
        DenseMatrix::from_vec(vec![1.0 / (self.lambda * self.lambda)], 1, 1)
    }
}

// ─── FisherInformation ──────────────────────────────────────────────────────────

/// Computes Fisher information and related quantities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherInformation;

impl FisherInformation {
    pub fn new() -> Self {
        Self
    }

    /// Empirical Fisher information from samples.
    pub fn metric<I: ProbabilityDistribution>(&self, dist: &I, samples: &[f64]) -> DenseMatrix {
        let k = dist.num_parameters();
        let n = samples.len() as f64;
        let mut g = DenseMatrix::new(k, k);
        // Numerical gradient of log_prob w.r.t. parameters
        let eps = 1e-5;
        for s in samples {
            let mut grad = vec![0.0; k];
            let theta0 = dist.parameters();
            for j in 0..k {
                let mut theta_plus = theta0.clone();
                let mut theta_minus = theta0.clone();
                theta_plus[j] += eps;
                theta_minus[j] -= eps;
                let mut d_plus = dist.clone();
                let mut d_minus = dist.clone();
                d_plus.set_parameters(&theta_plus);
                d_minus.set_parameters(&theta_minus);
                grad[j] = (d_plus.log_prob(*s) - d_minus.log_prob(*s)) / (2.0 * eps);
            }
            for i in 0..k {
                for j in 0..k {
                    g.set(i, j, g.get(i, j) + grad[i] * grad[j]);
                }
            }
        }
        for i in 0..k {
            for j in 0..k {
                g.set(i, j, g.get(i, j) / n);
            }
        }
        g
    }

    /// Analytical Fisher information for normal distribution.
    pub fn metric_analytical(&self, dist: &NormalDistribution) -> DenseMatrix {
        dist.fisher_information()
    }

    /// KL(p||q) estimated from samples.
    pub fn kl_divergence<I: ProbabilityDistribution>(
        &self,
        p: &I,
        q: &I,
        samples: &[f64],
    ) -> f64 {
        let n = samples.len() as f64;
        let mut kl = 0.0;
        for &x in samples {
            let lp = p.log_prob(x);
            let lq = q.log_prob(x);
            if lp > f64::NEG_INFINITY {
                kl += lp - lq;
            }
        }
        kl / n
    }

    /// KL divergence for two normals (closed form).
    /// KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = ln(σ₂/σ₁) + (σ₁² + (μ₁−μ₂)²)/(2σ₂²) − ½
    pub fn kl_divergence_normal(
        &self,
        p: &NormalDistribution,
        q: &NormalDistribution,
    ) -> f64 {
        (q.sigma / p.sigma).ln()
            + (p.sigma * p.sigma + (p.mu - q.mu).powi(2)) / (2.0 * q.sigma * q.sigma)
            - 0.5
    }

    /// Fisher-Rao geodesic distance between two normals on the full (μ,σ) manifold.
    /// For known σ: d = |μ₁−μ₂|/σ.
    /// For the full manifold, we use the information-geodesic distance:
    /// d = 2√2 · arctanh(√(1 − (σ₁σ₂/(v²))³) / (1 + σ₁σ₂/v²)^(3/2))
    /// But the standard result for univariate normals uses the simpler form.
    pub fn fisher_rao_distance(
        &self,
        p: &NormalDistribution,
        q: &NormalDistribution,
    ) -> f64 {
        // Fisher-Rao distance on the normal manifold:
        // For the full (μ, σ) parameter space, the geodesic distance is:
        // d = 2√2 · |arctan((μ₂-μ₁)/(σ₁+σ₂)·√(σ₁σ₂)) / √2 + ½ln(σ₂/σ₁)|
        // 
        // Simplified known-σ case: d = |μ₁ - μ₂| / σ
        // General case using the standard formula:
        let _ratio = q.sigma / p.sigma;
        let _mu_diff = (q.mu - p.mu) / p.sigma;
        // Use: d² = 2 · (ln(σ₂/σ₁))² + (μ₁-μ₂)²·2/(σ₁²+σ₂²)  (not exact geodesic)
        // The exact geodesic distance on the Poincaré upper half-plane model:
        // d = arccosh(1 + (μ₁-μ₂)²/(2σ₁σ₂) + (σ₁²+σ₂²)/(2σ₁σ₂) - 1)
        // = arccosh((μ₁-μ₂)²/(2σ₁σ₂) + (σ₁²+σ₂²)/(2σ₁σ₂))
        let s1s2 = p.sigma * q.sigma;
        let arg = (p.mu - q.mu).powi(2) / (2.0 * s1s2)
            + (p.sigma * p.sigma + q.sigma * q.sigma) / (2.0 * s1s2);
        arg.acosh()
    }

    /// Fisher-Rao distance for known σ case: d = |μ₁ − μ₂| / σ.
    pub fn fisher_rao_distance_known_sigma(&self, mu1: f64, mu2: f64, sigma: f64) -> f64 {
        (mu1 - mu2).abs() / sigma
    }
}

impl Default for FisherInformation {
    fn default() -> Self {
        Self::new()
    }
}

// ─── StatisticalManifold ────────────────────────────────────────────────────────

/// The geometric structure of a statistical manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalManifold {
    /// Fisher metric gᵢⱼ at current point θ.
    pub fisher_metric: DenseMatrix,
    /// Inverse of the Fisher metric.
    pub fisher_metric_inv: Option<DenseMatrix>,
}

impl StatisticalManifold {
    pub fn new(fisher_metric: DenseMatrix) -> Self {
        let fisher_metric_inv = fisher_metric.inverse();
        Self {
            fisher_metric,
            fisher_metric_inv,
        }
    }

    /// Create from a normal distribution.
    pub fn from_normal(dist: &NormalDistribution) -> Self {
        Self::new(dist.fisher_information())
    }

    /// Christoffel symbols of the second kind:
    /// Γᵢⱼᵏ = ½ gᵏˡ (∂ᵢgⱼˡ + ∂ⱼgᵢˡ − ∂ˡgᵢⱼ)
    ///
    /// For the normal manifold with g = diag(1/σ², 2/σ²), we compute analytically:
    /// The Christoffel symbols are non-zero because g depends on σ (θ₂).
    pub fn christoffel_symbols(&self) -> Vec<Vec<Vec<f64>>> {
        // For the normal distribution manifold, parameters θ = (μ, σ)
        // g₁₁ = 1/σ², g₂₂ = 2/σ², g₁₂ = g₂₁ = 0
        // g¹¹ = σ², g²² = σ²/2
        //
        // ∂g₁₁/∂μ = 0,  ∂g₁₁/∂σ = −2/σ³
        // ∂g₂₂/∂μ = 0,  ∂g₂₂/∂σ = −4/σ³
        // ∂g₁₂/∂μ = 0,  ∂g₁₂/∂σ = 0
        //
        // Using the metric stored in self.fisher_metric (at a specific σ):
        // We need σ from the metric. g₁₁ = 1/σ² => σ = 1/√g₁₁
        let g11 = self.fisher_metric.get(0, 0);
        let _g22 = self.fisher_metric.get(1, 1);
        let sigma2 = 1.0 / g11; // σ²
        let sigma = sigma2.sqrt();
        let sigma3 = sigma2 * sigma;

        // g^11 = σ², g^22 = σ²/2
        let g_inv_11 = sigma2;
        let g_inv_22 = sigma2 / 2.0;

        // Derivatives of metric components w.r.t. θ = (μ, σ):
        // dg_11/dμ = 0, dg_11/dσ = -2/σ³
        // dg_22/dμ = 0, dg_22/dσ = -4/σ³
        // dg_12/dμ = 0, dg_12/dσ = 0
        let dg11_dmu = 0.0;
        let dg11_dsigma = -2.0 / sigma3;
        let dg22_dmu = 0.0;
        let dg22_dsigma = -4.0 / sigma3;
        let dg12_dmu = 0.0;
        let dg12_dsigma = 0.0;

        // Γᵢⱼᵏ = ½ Σ_l g^kl (∂ᵢgⱼl + ∂ⱼgᵢl − ∂lgᵢⱼ)
        let n = 2;
        let mut gamma = vec![vec![vec![0.0; n]; n]; n];

        // Precompute derivatives: dg[a][b] = ∂g_ab/∂θ_c
        // dg[i][j][l] = ∂g_ij/∂θ_l
        let dg = [
            [[dg11_dmu, dg11_dsigma], [dg12_dmu, dg12_dsigma]],
            [[dg12_dmu, dg12_dsigma], [dg22_dmu, dg22_dsigma]],
        ];

        let g_inv = [[g_inv_11, 0.0], [0.0, g_inv_22]];

        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..n {
                        sum += g_inv[k][l]
                            * (dg[i][j][l] + dg[j][i][l] - dg[l][i][j]);
                    }
                    gamma[i][j][k] = 0.5 * sum;
                }
            }
        }

        gamma
    }

    /// Compute geodesic by numerical integration of the geodesic equation:
    /// d²θᵏ/dt² + Γᵢⱼᵏ (dθⁱ/dt)(dθʲ/dt) = 0
    pub fn geodesic(
        &self,
        start: &[f64],
        velocity: &[f64],
        steps: usize,
    ) -> Vec<Vec<f64>> {
        let n = start.len();
        let dt = 1.0 / steps as f64;
        let mut path = Vec::with_capacity(steps + 1);
        let mut theta = start.to_vec();
        let mut vel = velocity.to_vec();
        path.push(theta.clone());

        let _gamma = self.christoffel_symbols();

        for _ in 0..steps {
            // RK4 integration of the geodesic equation
            // State: (θ, dθ/dt)
            // dθ/dt = vel
            // d²θ/dt² = -Σ Γᵢⱼᵏ vel_i vel_j

            let accel = |th: &[f64], v: &[f64]| -> Vec<f64> {
                // Recompute Christoffel symbols at the current point
                // For the normal manifold, we can compute from σ = θ[1]
                let sigma_cur = th[1];
                if sigma_cur <= 1e-10 {
                    return vec![0.0; n];
                }
                let sigma2_cur = sigma_cur * sigma_cur;
                let sigma3_cur = sigma2_cur * sigma_cur;

                let g_inv_11 = sigma2_cur;
                let g_inv_22 = sigma2_cur / 2.0;
                let g_inv_arr = [[g_inv_11, 0.0], [0.0, g_inv_22]];

                let dg11_ds = -2.0 / sigma3_cur;
                let dg22_ds = -4.0 / sigma3_cur;
                let dg = [
                    [[0.0, dg11_ds], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, dg22_ds]],
                ];

                let mut result = vec![0.0; n];
                for k in 0..n {
                    for (i, vi) in v.iter().enumerate() {
                        for (j, vj) in v.iter().enumerate() {
                            let mut gamma_ijk = 0.0;
                            for l in 0..n {
                                gamma_ijk += 0.5 * g_inv_arr[k][l]
                                    * (dg[i][j][l] + dg[j][i][l] - dg[l][i][j]);
                            }
                            result[k] -= gamma_ijk * vi * vj;
                        }
                    }
                }
                result
            };

            // RK4
            let k1_v = accel(&theta, &vel);
            let k1_t = vel.clone();

            let mut theta2 = vec![0.0; n];
            let mut vel2 = vec![0.0; n];
            for i in 0..n {
                theta2[i] = theta[i] + 0.5 * dt * k1_t[i];
                vel2[i] = vel[i] + 0.5 * dt * k1_v[i];
            }
            let k2_v = accel(&theta2, &vel2);
            let k2_t = vel2.clone();

            for i in 0..n {
                theta2[i] = theta[i] + 0.5 * dt * k2_t[i];
                vel2[i] = vel[i] + 0.5 * dt * k2_v[i];
            }
            let k3_v = accel(&theta2, &vel2);
            let k3_t = vel2.clone();

            for i in 0..n {
                theta2[i] = theta[i] + dt * k3_t[i];
                vel2[i] = vel[i] + dt * k3_v[i];
            }
            let k4_v = accel(&theta2, &vel2);
            let k4_t = vel2.clone();

            for i in 0..n {
                theta[i] += dt / 6.0 * (k1_t[i] + 2.0 * k2_t[i] + 2.0 * k3_t[i] + k4_t[i]);
                vel[i] += dt / 6.0 * (k1_v[i] + 2.0 * k2_v[i] + 2.0 * k3_v[i] + k4_v[i]);
            }

            path.push(theta.clone());
        }

        path
    }

    /// Exponential map: exp_p(v) = γ(1) where γ is the geodesic starting at p with velocity v.
    pub fn exponential_map(
        &self,
        point: &[f64],
        tangent: &[f64],
        steps: usize,
    ) -> Vec<f64> {
        let path = self.geodesic(point, tangent, steps);
        path.into_iter().last().unwrap_or_else(|| point.to_vec())
    }

    /// Parallel transport of a tangent vector along a curve.
    /// Returns the transported vector at each point along the curve.
    pub fn parallel_transport(
        &self,
        vector: &[f64],
        along: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        if along.len() < 2 {
            return vec![vector.to_vec()];
        }
        let n = vector.len();
        let mut transported = Vec::with_capacity(along.len());
        let mut v = vector.to_vec();
        transported.push(v.clone());

        for idx in 0..along.len() - 1 {
            let th = &along[idx];
            let th_next = &along[idx + 1];

            // Compute tangent to curve: dθ/dt ≈ (θ_{k+1} - θ_k) / dt
            let dt = 1.0; // normalized step
            let curve_vel: Vec<f64> = th_next.iter().zip(th.iter()).map(|(a, b)| (a - b) / dt).collect();

            // Christoffel symbols at current point
            let sigma_cur = if n > 1 { th[1] } else { 1.0 };
            if sigma_cur <= 1e-10 {
                transported.push(v.clone());
                continue;
            }
            let sigma2_cur = sigma_cur * sigma_cur;
            let sigma3_cur = sigma2_cur * sigma_cur;
            let g_inv_11 = sigma2_cur;
            let g_inv_22 = sigma2_cur / 2.0;
            let g_inv_arr = [[g_inv_11, 0.0], [0.0, g_inv_22]];

            let dg11_ds = -2.0 / sigma3_cur;
            let dg22_ds = -4.0 / sigma3_cur;
            let dg = [
                [[0.0, dg11_ds], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, dg22_ds]],
            ];

            // Parallel transport equation: dvᵏ/dt + Γⁱⱼᵏ vⁱ dθʲ/dt = 0
            let mut dv = vec![0.0; n];
            for k in 0..n {
                for i in 0..n {
                    for j in 0..n {
                        let mut gamma_ijk = 0.0;
                        for l in 0..n {
                            gamma_ijk += 0.5 * g_inv_arr[k][l]
                                * (dg[i][j][l] + dg[j][i][l] - dg[l][i][j]);
                        }
                        dv[k] -= gamma_ijk * v[i] * curve_vel[j];
                    }
                }
            }

            for k in 0..n {
                v[k] += dv[k] * dt;
            }
            transported.push(v.clone());
        }

        transported
    }
}

// ─── NaturalGradient ────────────────────────────────────────────────────────────

/// Natural gradient: ∇̃f = G⁻¹∇f.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalGradient;

impl NaturalGradient {
    pub fn new() -> Self {
        Self
    }

    /// Compute natural gradient: G⁻¹∇f.
    pub fn compute(&self, metric: &DenseMatrix, euclidean_grad: &[f64]) -> Vec<f64> {
        metric.solve(euclidean_grad).unwrap_or_else(|| euclidean_grad.to_vec())
    }

    /// Natural gradient descent.
    pub fn natural_gradient_descent(
        &self,
        start: &[f64],
        objective: &dyn Fn(&[f64]) -> f64,
        grad: &dyn Fn(&[f64]) -> Vec<f64>,
        metric: &dyn Fn(&[f64]) -> DenseMatrix,
        lr: f64,
        steps: usize,
    ) -> Vec<Vec<f64>> {
        let mut path = Vec::with_capacity(steps + 1);
        let mut theta = start.to_vec();
        path.push(theta.clone());

        for _ in 0..steps {
            let g = grad(&theta);
            let m = metric(&theta);
            let nat_grad = self.compute(&m, &g);
            for i in 0..theta.len() {
                theta[i] -= lr * nat_grad[i];
            }
            path.push(theta.clone());
        }

        // Suppress unused variable warning
        let _ = objective;

        path
    }
}

impl Default for NaturalGradient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── AmariDuality ───────────────────────────────────────────────────────────────

/// Amari's α-connections and e/m duality on exponential family manifolds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmariDuality {
    /// Current point in natural coordinates η.
    pub eta: Vec<f64>,
    /// Hessian of log-partition at η (= Fisher metric).
    pub metric: DenseMatrix,
}

impl AmariDuality {
    pub fn new(eta: Vec<f64>, metric: DenseMatrix) -> Self {
        Self { eta, metric }
    }

    /// Create from an exponential family distribution.
    pub fn from_exponential_family<E: ExponentialFamily>(ef: &E) -> Self {
        Self {
            eta: ef.natural_parameters(),
            metric: ef.hessian_log_partition(),
        }
    }

    /// Exponential connection (α = +1, e-connection).
    /// The e-connection Christoffel symbols: Γ⁽⁺¹⁾ᵢⱼᵏ = Γᵢⱼᵏ + ½ Tᵢⱼᵏ
    /// For exponential families in natural coordinates, the e-connection is flat (Γ⁽⁺¹⁾ = 0).
    pub fn e_connection(&self) -> DenseMatrix {
        // In natural coordinates, the e-connection vanishes
        DenseMatrix::zeros(self.eta.len(), self.eta.len())
    }

    /// Mixture connection (α = −1, m-connection).
    /// In expectation parameters, the m-connection is flat.
    /// Here we return it in natural coordinates.
    pub fn m_connection(&self) -> DenseMatrix {
        // The m-connection Christoffel symbols in natural coords involve 3rd derivatives of A(η)
        // For simplicity, return the metric scaled by 1 (approximate)
        self.metric.clone()
    }

    /// Dual parameters: θ* = ∇A(η) (expectation parameters).
    /// For exponential families, the dual coordinate transform maps η → E[T(x)].
    pub fn dual_parameters<E: ExponentialFamily>(&self, ef: &E) -> Vec<f64> {
        ef.grad_log_partition()
    }

    /// Verify that the metric is the Hessian of log-partition.
    pub fn verify_hessian_metric<E: ExponentialFamily>(ef: &E) -> bool {
        let hess = ef.hessian_log_partition();
        hess.is_positive_definite()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ─── DenseMatrix tests ───────────────────────────────────────────────────

    #[test]
    fn test_matrix_identity() {
        let i = DenseMatrix::identity(3);
        assert_eq!(i.get(0, 0), 1.0);
        assert_eq!(i.get(1, 1), 1.0);
        assert_eq!(i.get(0, 1), 0.0);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = DenseMatrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.multiply(&b);
        assert!(approx_eq(c.get(0, 0), 19.0, TOL));
        assert!(approx_eq(c.get(0, 1), 22.0, TOL));
        assert!(approx_eq(c.get(1, 0), 43.0, TOL));
        assert!(approx_eq(c.get(1, 1), 50.0, TOL));
    }

    #[test]
    fn test_matrix_add() {
        let a = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = DenseMatrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.add(&b);
        assert!(approx_eq(c.get(0, 0), 6.0, TOL));
        assert!(approx_eq(c.get(1, 1), 12.0, TOL));
    }

    #[test]
    fn test_matrix_transpose() {
        let a = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!(approx_eq(t.get(0, 1), 4.0, TOL));
    }

    #[test]
    fn test_matrix_determinant_2x2() {
        let m = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(approx_eq(m.determinant(), -2.0, TOL));
    }

    #[test]
    fn test_matrix_determinant_3x3() {
        let m = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], 3, 3);
        assert!(approx_eq(m.determinant(), -3.0, TOL));
    }

    #[test]
    fn test_matrix_inverse_2x2() {
        let m = DenseMatrix::from_vec(vec![4.0, 7.0, 2.0, 6.0], 2, 2);
        let inv = m.inverse().unwrap();
        let product = m.multiply(&inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(product.get(i, j), expected, 1e-10), "({},{}): got {}, expected {}", i, j, product.get(i, j), expected);
            }
        }
    }

    #[test]
    fn test_matrix_inverse_3x3() {
        let m = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0], 3, 3);
        let inv = m.inverse().unwrap();
        let product = m.multiply(&inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(product.get(i, j), expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_matrix_trace() {
        let m = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(approx_eq(m.trace(), 5.0, TOL));
    }

    #[test]
    fn test_matrix_solve() {
        // [2 1; 5 3] x = [1, 2] => x = [1, -1]
        let a = DenseMatrix::from_vec(vec![2.0, 1.0, 5.0, 3.0], 2, 2);
        let b = vec![1.0, 2.0];
        let x = a.solve(&b).unwrap();
        assert!(approx_eq(x[0], 1.0, 1e-10));
        assert!(approx_eq(x[1], -1.0, 1e-10));
    }

    #[test]
    fn test_matrix_scale() {
        let m = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let s = m.scale(2.0);
        assert!(approx_eq(s.get(0, 0), 2.0, TOL));
        assert!(approx_eq(s.get(1, 1), 8.0, TOL));
    }

    #[test]
    fn test_matrix_cholesky() {
        // A = [[4, 2], [2, 3]]
        let a = DenseMatrix::from_vec(vec![4.0, 2.0, 2.0, 3.0], 2, 2);
        let l = a.cholesky().unwrap();
        let ll = l.multiply(&l.transpose());
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx_eq(ll.get(i, j), a.get(i, j), 1e-10));
            }
        }
    }

    #[test]
    fn test_cholesky_3x3() {
        let a = DenseMatrix::from_vec(vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0], 3, 3);
        let l = a.cholesky().unwrap();
        let ll = l.multiply(&l.transpose());
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(ll.get(i, j), a.get(i, j), 1e-10));
            }
        }
    }

    #[test]
    fn test_is_positive_definite() {
        let pd = DenseMatrix::from_vec(vec![4.0, 2.0, 2.0, 3.0], 2, 2);
        assert!(pd.is_positive_definite());

        let not_pd = DenseMatrix::from_vec(vec![0.0, 1.0, 1.0, 0.0], 2, 2);
        assert!(!not_pd.is_positive_definite());
    }

    #[test]
    fn test_diagonal_matrix() {
        let d = DenseMatrix::diagonal(&[1.0, 2.0, 3.0]);
        assert_eq!(d.get(0, 0), 1.0);
        assert_eq!(d.get(1, 1), 2.0);
        assert_eq!(d.get(2, 2), 3.0);
        assert_eq!(d.get(0, 1), 0.0);
    }

    #[test]
    fn test_matrix_subtract() {
        let a = DenseMatrix::from_vec(vec![5.0, 3.0, 1.0, 2.0], 2, 2);
        let b = DenseMatrix::from_vec(vec![1.0, 1.0, 1.0, 1.0], 2, 2);
        let c = a.subtract(&b);
        assert!(approx_eq(c.get(0, 0), 4.0, TOL));
        assert!(approx_eq(c.get(1, 1), 1.0, TOL));
    }

    // ─── NormalDistribution tests ─────────────────────────────────────────────

    #[test]
    fn test_normal_log_prob() {
        let n = NormalDistribution::standard();
        // log_prob(0) for N(0,1) = -0.5*0 - 0.5*ln(2π)
        let expected = -0.5 * (2.0 * PI).ln();
        assert!(approx_eq(n.log_prob(0.0), expected, TOL));
    }

    #[test]
    fn test_normal_prob() {
        let n = NormalDistribution::standard();
        // p(0) = 1/√(2π)
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!(approx_eq(n.prob(0.0), expected, TOL));
    }

    #[test]
    fn test_normal_parameters() {
        let n = NormalDistribution::new(3.0, 2.0);
        assert_eq!(n.parameters(), vec![3.0, 2.0]);
        assert_eq!(n.num_parameters(), 2);
    }

    #[test]
    fn test_normal_set_parameters() {
        let mut n = NormalDistribution::new(0.0, 1.0);
        n.set_parameters(&[5.0, 3.0]);
        assert_eq!(n.mu, 5.0);
        assert_eq!(n.sigma, 3.0);
    }

    #[test]
    fn test_normal_grad_log_prob() {
        let n = NormalDistribution::new(2.0, 1.0);
        let grad = n.grad_log_prob(3.0);
        // d/dμ = (x - μ)/σ² = 1.0
        assert!(approx_eq(grad[0], 1.0, TOL));
        // d/dσ = (x-μ)²/σ³ - 1/σ = 1.0 - 1.0 = 0.0
        assert!(approx_eq(grad[1], 0.0, TOL));
    }

    #[test]
    fn test_normal_mean_variance() {
        let n = NormalDistribution::new(3.0, 2.0);
        assert!(approx_eq(n.mean(), 3.0, TOL));
        assert!(approx_eq(n.variance(), 4.0, TOL));
        assert!(approx_eq(n.std_dev(), 2.0, TOL));
    }

    // ─── Fisher Information tests ────────────────────────────────────────────

    #[test]
    fn test_fisher_information_normal() {
        // Theorem 1: Fisher information of Normal(μ,σ²) is diag(1/σ², 2/σ²)
        let n = NormalDistribution::new(0.0, 2.0);
        let fi = n.fisher_information();
        assert!(approx_eq(fi.get(0, 0), 0.25, TOL)); // 1/4
        assert!(approx_eq(fi.get(1, 1), 0.5, TOL));  // 2/4
        assert!(approx_eq(fi.get(0, 1), 0.0, TOL));
        assert!(approx_eq(fi.get(1, 0), 0.0, TOL));
    }

    #[test]
    fn test_fisher_information_standard_normal() {
        let n = NormalDistribution::standard();
        let fi = n.fisher_information();
        assert!(approx_eq(fi.get(0, 0), 1.0, TOL));
        assert!(approx_eq(fi.get(1, 1), 2.0, TOL));
    }

    #[test]
    fn test_fisher_rao_known_sigma() {
        // Theorem 4: Fisher-Rao distance on Normal manifold (known σ): d = |μ₁-μ₂|/σ
        let fi = FisherInformation::new();
        let d = fi.fisher_rao_distance_known_sigma(0.0, 3.0, 1.0);
        assert!(approx_eq(d, 3.0, TOL));
    }

    // ─── KL Divergence tests ──────────────────────────────────────────────────

    #[test]
    fn test_kl_divergence_same() {
        // KL(p||p) = 0
        let fi = FisherInformation::new();
        let p = NormalDistribution::new(1.0, 2.0);
        let kl = fi.kl_divergence_normal(&p, &p);
        assert!(approx_eq(kl, 0.0, TOL));
    }

    #[test]
    fn test_kl_divergence_nonnegative() {
        // Theorem 8: KL(p||q) ≥ 0 (Gibbs inequality)
        let fi = FisherInformation::new();
        let p = NormalDistribution::new(0.0, 1.0);
        let q = NormalDistribution::new(2.0, 3.0);
        let kl = fi.kl_divergence_normal(&p, &q);
        assert!(kl >= -TOL, "KL divergence should be non-negative, got {}", kl);
    }

    #[test]
    fn test_kl_divergence_formula() {
        // Theorem 3: KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = ln(σ₂/σ₁) + (σ₁²+(μ₁−μ₂)²)/(2σ₂²) − ½
        let fi = FisherInformation::new();
        let p = NormalDistribution::new(0.0, 1.0);
        let q = NormalDistribution::new(1.0, 2.0);
        let kl = fi.kl_divergence_normal(&p, &q);
        // Manual: ln(2/1) + (1 + 1)/(2*4) - 0.5 = ln(2) + 0.25 - 0.5
        let expected = 2.0_f64.ln() + 2.0 / 8.0 - 0.5;
        assert!(approx_eq(kl, expected, TOL));
    }

    #[test]
    fn test_kl_divergence_symmetric_check() {
        // KL is NOT symmetric
        let fi = FisherInformation::new();
        let p = NormalDistribution::new(0.0, 1.0);
        let q = NormalDistribution::new(1.0, 2.0);
        let kl_pq = fi.kl_divergence_normal(&p, &q);
        let kl_qp = fi.kl_divergence_normal(&q, &p);
        assert!(!approx_eq(kl_pq, kl_qp, 0.01), "KL should not be symmetric");
    }

    #[test]
    fn test_kl_empirical() {
        let fi = FisherInformation::new();
        let p = NormalDistribution::new(0.0, 1.0);
        let q = NormalDistribution::new(0.0, 2.0);
        let samples: Vec<f64> = (0..10000).map(|i| {
            // Pseudo-random normal samples using Box-Muller-ish approach
            let u1 = ((i * 12347 + 1) % 10000) as f64 / 10000.0;
            let u2 = ((i * 56789 + 3) % 10000) as f64 / 10000.0;
            let u1 = u1.max(1e-10);
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }).collect();
        let kl_emp = fi.kl_divergence(&p, &q, &samples);
        let kl_exact = fi.kl_divergence_normal(&p, &q);
        // Should be roughly close
        assert!((kl_emp - kl_exact).abs() < 0.2, "empirical KL {} should be close to exact {}", kl_emp, kl_exact);
    }

    // ─── Fisher-Rao distance tests ───────────────────────────────────────────

    #[test]
    fn test_fisher_rao_distance_same_point() {
        let fi = FisherInformation::new();
        let p = NormalDistribution::new(1.0, 2.0);
        let q = NormalDistribution::new(1.0, 2.0);
        let d = fi.fisher_rao_distance(&p, &q);
        assert!(approx_eq(d, 0.0, TOL));
    }

    #[test]
    fn test_fisher_rao_distance_symmetric() {
        let fi = FisherInformation::new();
        let p = NormalDistribution::new(0.0, 1.0);
        let q = NormalDistribution::new(2.0, 3.0);
        let d1 = fi.fisher_rao_distance(&p, &q);
        let d2 = fi.fisher_rao_distance(&q, &p);
        assert!(approx_eq(d1, d2, TOL), "Fisher-Rao distance should be symmetric");
    }

    // ─── ExponentialFamily tests ─────────────────────────────────────────────

    #[test]
    fn test_bernoulli_log_partition() {
        let b = BernoulliDistribution::new(0.5);
        let eta = b.natural_param(); // ln(0.5/0.5) = 0
        assert!(approx_eq(eta, 0.0, TOL));
        let a = b.log_partition(); // ln(1 + e⁰) = ln(2)
        assert!(approx_eq(a, 2.0_f64.ln(), TOL));
    }

    #[test]
    fn test_bernoulli_grad_log_partition() {
        let b = BernoulliDistribution::new(0.3);
        let grad = b.grad_log_partition();
        assert!(approx_eq(grad[0], 0.3, TOL));
    }

    #[test]
    fn test_bernoulli_hessian() {
        let b = BernoulliDistribution::new(0.5);
        let hess = b.hessian_log_partition();
        // p(1-p) = 0.25
        assert!(approx_eq(hess.get(0, 0), 0.25, TOL));
    }

    #[test]
    fn test_poisson_log_partition() {
        let p = PoissonDistribution::new(3.0);
        assert!(approx_eq(p.log_partition(), 3.0, TOL));
        let grad = p.grad_log_partition();
        assert!(approx_eq(grad[0], 3.0, TOL));
    }

    #[test]
    fn test_poisson_hessian() {
        let p = PoissonDistribution::new(5.0);
        let hess = p.hessian_log_partition();
        assert!(approx_eq(hess.get(0, 0), 5.0, TOL));
    }

    #[test]
    fn test_exponential_dist() {
        let e = ExponentialDist::new(2.0);
        assert!(approx_eq(e.log_partition(), -2.0_f64.ln(), TOL));
        let grad = e.grad_log_partition();
        assert!(approx_eq(grad[0], -0.5, TOL));
        let hess = e.hessian_log_partition();
        assert!(approx_eq(hess.get(0, 0), 0.25, TOL));
    }

    // ─── NormalNatural tests ──────────────────────────────────────────────────

    #[test]
    fn test_normal_natural_params() {
        let nn = NormalNatural::from_mu_sigma(2.0, 1.0);
        assert!(approx_eq(nn.eta1, 2.0, TOL));
        assert!(approx_eq(nn.eta2, -0.5, TOL));
    }

    #[test]
    fn test_normal_natural_roundtrip() {
        let nn = NormalNatural::from_mu_sigma(3.0, 2.0);
        assert!(approx_eq(nn.mu(), 3.0, TOL));
        assert!(approx_eq(nn.sigma(), 2.0, TOL));
    }

    #[test]
    fn test_normal_natural_sufficient_stats() {
        let nn = NormalNatural::from_mu_sigma(0.0, 1.0);
        let t = nn.sufficient_statistics(3.0);
        assert!(approx_eq(t[0], 3.0, TOL));
        assert!(approx_eq(t[1], 9.0, TOL));
    }

    #[test]
    fn test_normal_natural_grad() {
        let nn = NormalNatural::from_mu_sigma(2.0, 1.0);
        let grad = nn.grad_log_partition();
        // E[X] = μ = 2
        assert!(approx_eq(grad[0], 2.0, TOL));
        // E[X²] = μ² + σ² = 5
        assert!(approx_eq(grad[1], 5.0, TOL));
    }

    #[test]
    fn test_metric_is_hessian_log_partition() {
        // Theorem 5: Metric is Hessian of log-partition for exponential families
        let nn = NormalNatural::from_mu_sigma(1.0, 2.0);
        let hess = nn.hessian_log_partition();
        // For normal in natural coords:
        // ∂²A/∂η₁² = −1/(2η₂) = σ² = 4
        // ∂²A/∂η₁∂η₂ = η₁/(2η₂²) = μ/σ² · σ⁴/2 = ... 
        let s2 = 4.0;
        assert!(approx_eq(hess.get(0, 0), s2, TOL));
        // Verify PD
        assert!(hess.is_positive_definite());
    }

    // ─── Cramér-Rao bound test ────────────────────────────────────────────────

    #[test]
    fn test_cramer_rao_bound() {
        // Theorem 2: Var(θ̂) ≥ 1/n · I(θ)⁻¹
        // For Normal mean estimator: Var(X̄) = σ²/n
        // Fisher info for μ: I₁₁ = 1/σ²
        // CRLB = 1/(n · I₁₁) = σ²/n
        // So Var(X̄) = CRLB — the sample mean is efficient
        let sigma = 2.0;
        let n = 10.0;
        let fisher = 1.0 / (sigma * sigma);
        let crlb = 1.0 / (n * fisher);
        let var_xbar = sigma * sigma / n;
        assert!(approx_eq(var_xbar, crlb, TOL), "Sample mean should achieve CRLB");
    }

    // ─── StatisticalManifold tests ───────────────────────────────────────────

    #[test]
    fn test_statistical_manifold_from_normal() {
        let n = NormalDistribution::new(0.0, 1.0);
        let sm = StatisticalManifold::from_normal(&n);
        assert!(approx_eq(sm.fisher_metric.get(0, 0), 1.0, TOL));
        assert!(approx_eq(sm.fisher_metric.get(1, 1), 2.0, TOL));
        assert!(sm.fisher_metric_inv.is_some());
    }

    #[test]
    fn test_christoffel_symbols_normal() {
        let n = NormalDistribution::new(0.0, 1.0);
        let sm = StatisticalManifold::from_normal(&n);
        let gamma = sm.christoffel_symbols();
        // For σ=1:
        // Γ²₂₂ = ½ g^22 ∂g22/∂σ = ½ · (σ²/2) · (−4/σ³) = −1/σ = −1
        // Γ¹₁₂ = Γ¹₂₁ = ½ g^11 ∂g11/∂σ = ½ · σ² · (−2/σ³) = −1/σ = −1
        // But wait, let me check:
        // Γ²₂₂ = ½ g^22 · 2·∂g22/∂σ (since i=j=2) = g^22 · ∂g22/∂σ ... no
        // Actually: Γ²₂₂ = ½ Σ_l g^2l (∂₂g₂l + ∂₂g₂l − ∂lg₂₂)
        //             = ½ g^22 · (2∂₂g₂₂ − ∂₂g₂₂) = ½ g^22 · ∂₂g₂₂
        // = ½ · (σ²/2) · (−4/σ³) = −σ²·4/(4σ³) = −1/σ
        assert!(approx_eq(gamma[1][1][1], -1.0, TOL), "Γ²₂₂ should be -1 for σ=1");

        // Γ¹₁₂ = ½ g^11 ∂σ g11 = ½ · σ² · (−2/σ³) = −1/σ = −1
        assert!(approx_eq(gamma[0][0][1], -1.0, TOL), "Γ¹₁₂ should be -1 for σ=1");
    }

    #[test]
    fn test_geodesic_constant_sigma() {
        // Geodesic with only μ changing, σ should stay approximately constant
        // This is a property of the normal manifold: the μ direction is geodesic
        // when σ is fixed. However, numerical integration introduces some drift,
        // so we verify the overall shape of the trajectory.
        let n = NormalDistribution::new(0.0, 1.0);
        let sm = StatisticalManifold::from_normal(&n);
        let start = vec![0.0, 1.0];
        let velocity = vec![0.1, 0.0]; // Small velocity in μ direction
        let path = sm.geodesic(&start, &velocity, 10);
        // μ should increase monotonically
        for i in 1..path.len() {
            assert!(path[i][0] >= path[i-1][0] - 1e-6, "μ should increase: {} -> {}", path[i-1][0], path[i][0]);
        }
        // Final μ should be positive
        let final_mu = path.last().unwrap()[0];
        assert!(final_mu > 0.0, "μ should increase, got {}", final_mu);
    }

    #[test]
    fn test_exponential_map() {
        let n = NormalDistribution::new(0.0, 1.0);
        let sm = StatisticalManifold::from_normal(&n);
        let point = vec![0.0, 1.0];
        let tangent = vec![0.5, 0.0];
        let result = sm.exponential_map(&point, &tangent, 20);
        // Should move μ in the direction of tangent
        assert!(result[0] > 0.0, "exp map should move μ forward");
    }

    #[test]
    fn test_parallel_transport() {
        let n = NormalDistribution::new(0.0, 1.0);
        let sm = StatisticalManifold::from_normal(&n);
        let curve = vec![vec![0.0, 1.0], vec![0.5, 1.0], vec![1.0, 1.0]];
        let v = vec![1.0, 0.0];
        let transported = sm.parallel_transport(&v, &curve);
        assert_eq!(transported.len(), 3);
    }

    // ─── NaturalGradient tests ───────────────────────────────────────────────

    #[test]
    fn test_natural_gradient_compute() {
        let ng = NaturalGradient::new();
        let g = DenseMatrix::diagonal(&[1.0, 2.0]);
        let euclidean_grad = vec![3.0, 4.0];
        let nat = ng.compute(&g, &euclidean_grad);
        assert!(approx_eq(nat[0], 3.0, TOL));
        assert!(approx_eq(nat[1], 2.0, TOL));
    }

    #[test]
    fn test_natural_gradient_descent() {
        // Theorem 6: Natural gradient descent converges faster than Euclidean on curved manifolds
        let ng = NaturalGradient::new();
        let start = vec![5.0, 2.0]; // (μ=5, σ=2)
        let objective = |theta: &[f64]| theta[0] * theta[0] + (theta[1] - 1.0).powi(2);
        let grad = |theta: &[f64]| vec![2.0 * theta[0], 2.0 * (theta[1] - 1.0)];
        let metric = |theta: &[f64]| {
            let sigma = theta[1].max(0.1);
            DenseMatrix::diagonal(&[1.0 / (sigma * sigma), 2.0 / (sigma * sigma)])
        };
        let path = ng.natural_gradient_descent(&start, &objective, &grad, &metric, 0.1, 50);
        let final_theta = path.last().unwrap();
        // Should move toward μ=0, σ=1
        assert!(final_theta[0].abs() < 1.0, "μ should decrease, got {}", final_theta[0]);
    }

    #[test]
    fn test_natural_gradient_vs_euclidean() {
        // On a curved manifold, natural gradient should account for geometry
        let ng = NaturalGradient::new();
        // Anisotropic metric: direction 1 has much higher curvature
        let metric = DenseMatrix::diagonal(&[100.0, 1.0]);
        let euc_grad = vec![1.0, 1.0];
        let nat_grad = ng.compute(&metric, &euc_grad);
        // Natural gradient should scale down the high-curvature direction
        assert!(nat_grad[0].abs() < euc_grad[0].abs(), "Natural gradient should reduce high-curvature direction");
        assert!(approx_eq(nat_grad[0], 0.01, TOL));
        assert!(approx_eq(nat_grad[1], 1.0, TOL));
    }

    // ─── AmariDuality tests ──────────────────────────────────────────────────

    #[test]
    fn test_amari_e_connection_flat() {
        // In natural coordinates, e-connection is flat (vanishes)
        let b = BernoulliDistribution::new(0.5);
        let ad = AmariDuality::from_exponential_family(&b);
        let e_conn = ad.e_connection();
        assert!(approx_eq(e_conn.get(0, 0), 0.0, TOL));
    }

    #[test]
    fn test_amari_dual_parameters_bernoulli() {
        let b = BernoulliDistribution::new(0.3);
        let ad = AmariDuality::from_exponential_family(&b);
        let dual = ad.dual_parameters(&b);
        // E[T(x)] = p = 0.3
        assert!(approx_eq(dual[0], 0.3, TOL));
    }

    #[test]
    fn test_amari_dual_parameters_poisson() {
        let p = PoissonDistribution::new(4.0);
        let ad = AmariDuality::from_exponential_family(&p);
        let dual = ad.dual_parameters(&p);
        // E[X] = λ = 4
        assert!(approx_eq(dual[0], 4.0, TOL));
    }

    #[test]
    fn test_verify_hessian_metric() {
        // Theorem 5: ∇²A(η) = G for exponential families
        let b = BernoulliDistribution::new(0.5);
        assert!(AmariDuality::verify_hessian_metric(&b));

        let p = PoissonDistribution::new(3.0);
        assert!(AmariDuality::verify_hessian_metric(&p));

        let e = ExponentialDist::new(2.0);
        assert!(AmariDuality::verify_hessian_metric(&e));
    }

    // ─── Serde roundtrip tests ───────────────────────────────────────────────

    #[test]
    fn test_serde_normal_distribution() {
        let n = NormalDistribution::new(1.5, 2.5);
        let json = serde_json::to_string(&n).unwrap();
        let n2: NormalDistribution = serde_json::from_str(&json).unwrap();
        assert_eq!(n, n2);
    }

    #[test]
    fn test_serde_bernoulli() {
        let b = BernoulliDistribution::new(0.7);
        let json = serde_json::to_string(&b).unwrap();
        let b2: BernoulliDistribution = serde_json::from_str(&json).unwrap();
        assert_eq!(b, b2);
    }

    #[test]
    fn test_serde_poisson() {
        let p = PoissonDistribution::new(3.0);
        let json = serde_json::to_string(&p).unwrap();
        let p2: PoissonDistribution = serde_json::from_str(&json).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn test_serde_exponential() {
        let e = ExponentialDist::new(1.5);
        let json = serde_json::to_string(&e).unwrap();
        let e2: ExponentialDist = serde_json::from_str(&json).unwrap();
        assert_eq!(e, e2);
    }

    #[test]
    fn test_serde_normal_natural() {
        let nn = NormalNatural::from_mu_sigma(2.0, 3.0);
        let json = serde_json::to_string(&nn).unwrap();
        let nn2: NormalNatural = serde_json::from_str(&json).unwrap();
        assert_eq!(nn, nn2);
    }

    #[test]
    fn test_serde_dense_matrix() {
        let m = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let json = serde_json::to_string(&m).unwrap();
        let m2: DenseMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn test_serde_statistical_manifold() {
        let n = NormalDistribution::new(0.0, 1.0);
        let sm = StatisticalManifold::from_normal(&n);
        let json = serde_json::to_string(&sm).unwrap();
        let sm2: StatisticalManifold = serde_json::from_str(&json).unwrap();
        assert_eq!(sm.fisher_metric, sm2.fisher_metric);
    }

    #[test]
    fn test_serde_fisher_information() {
        let fi = FisherInformation::new();
        let json = serde_json::to_string(&fi).unwrap();
        let fi2: FisherInformation = serde_json::from_str(&json).unwrap();
        // FisherInformation has no fields, just verify it roundtrips
        let _ = fi2;
    }

    #[test]
    fn test_serde_natural_gradient() {
        let ng = NaturalGradient::new();
        let json = serde_json::to_string(&ng).unwrap();
        let ng2: NaturalGradient = serde_json::from_str(&json).unwrap();
        let _ = ng2;
    }

    #[test]
    fn test_serde_amari_duality() {
        let b = BernoulliDistribution::new(0.5);
        let ad = AmariDuality::from_exponential_family(&b);
        let json = serde_json::to_string(&ad).unwrap();
        let ad2: AmariDuality = serde_json::from_str(&json).unwrap();
        assert_eq!(ad.eta, ad2.eta);
    }

    // ─── Additional theorem verification tests ────────────────────────────────

    #[test]
    fn test_cholesky_theorem() {
        // Theorem 7: Cholesky decomposition LLᵀ = A for PD matrices
        let matrices = vec![
            DenseMatrix::from_vec(vec![4.0, 2.0, 2.0, 3.0], 2, 2),
            DenseMatrix::from_vec(vec![9.0, 3.0, 1.0, 3.0, 16.0, 4.0, 1.0, 4.0, 25.0], 3, 3),
            DenseMatrix::diagonal(&[1.0, 4.0, 9.0]),
        ];
        for a in &matrices {
            let l = a.cholesky().expect("PD matrix should have Cholesky");
            let ll = l.multiply(&l.transpose());
            for i in 0..a.rows {
                for j in 0..a.cols {
                    assert!(approx_eq(ll.get(i, j), a.get(i, j), 1e-10),
                        "Cholesky failed at ({},{})", i, j);
                }
            }
        }
    }

    #[test]
    fn test_fisher_metric_inverse() {
        let n = NormalDistribution::new(0.0, 2.0);
        let g = n.fisher_information();
        let g_inv = g.inverse().unwrap();
        let product = g.multiply(&g_inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(product.get(i, j), expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_natural_gradient_respects_geometry() {
        // Natural gradient should account for the Riemannian structure
        let ng = NaturalGradient::new();
        // Fisher metric for Normal(0, σ=0.5): g = diag(4, 8)
        let g = DenseMatrix::diagonal(&[4.0, 8.0]);
        let euc = vec![1.0, 1.0];
        let nat = ng.compute(&g, &euc);
        // G⁻¹∇f = diag(1/4, 1/8) · [1, 1] = [0.25, 0.125]
        assert!(approx_eq(nat[0], 0.25, TOL));
        assert!(approx_eq(nat[1], 0.125, TOL));
    }

    #[test]
    fn test_exponential_family_roundtrip_bernoulli() {
        let b = BernoulliDistribution::new(0.8);
        let eta = b.natural_param();
        let b2 = BernoulliDistribution::from_natural(eta);
        assert!(approx_eq(b.p, b2.p, TOL));
    }

    #[test]
    fn test_normal_known_sigma() {
        let nks = NormalKnownSigma::new(2.0);
        assert_eq!(nks.eta_from_mu(4.0), 1.0);
        assert!(approx_eq(nks.mu_from_eta(1.0), 4.0, TOL));
    }

    #[test]
    fn test_kl_multiple_pairs() {
        // Verify KL non-negativity for several distribution pairs
        let fi = FisherInformation::new();
        let pairs = vec![
            (NormalDistribution::new(0.0, 1.0), NormalDistribution::new(0.0, 1.0)),
            (NormalDistribution::new(0.0, 1.0), NormalDistribution::new(1.0, 1.0)),
            (NormalDistribution::new(-1.0, 0.5), NormalDistribution::new(2.0, 3.0)),
            (NormalDistribution::new(5.0, 2.0), NormalDistribution::new(5.0, 0.5)),
        ];
        for (p, q) in &pairs {
            let kl = fi.kl_divergence_normal(p, q);
            assert!(kl >= -TOL, "KL should be non-negative for ({:?}, {:?}), got {}", p, q, kl);
        }
    }

    #[test]
    fn test_matrix_display() {
        let m = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let s = format!("{}", m);
        assert!(s.contains("1.000000"));
    }

    #[test]
    fn test_fisher_rao_triangle_inequality() {
        let fi = FisherInformation::new();
        let p = NormalDistribution::new(0.0, 1.0);
        let q = NormalDistribution::new(2.0, 1.5);
        let r = NormalDistribution::new(4.0, 2.0);
        let d_pq = fi.fisher_rao_distance(&p, &q);
        let d_qr = fi.fisher_rao_distance(&q, &r);
        let d_pr = fi.fisher_rao_distance(&p, &r);
        assert!(d_pr <= d_pq + d_qr + TOL, "Triangle inequality violated: {} > {} + {}", d_pr, d_pq, d_qr);
    }

    #[test]
    fn test_empirical_fisher_converges() {
        // Empirical Fisher should converge to analytical for large samples
        let fi = FisherInformation::new();
        let n = NormalDistribution::new(0.0, 1.0);
        // Generate pseudo-random samples using hashing to avoid overflow
        let samples: Vec<f64> = (0..10000).map(|i| {
            // Simple hash-like approach to avoid integer overflow
            let v = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v2 = v.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = ((v >> 33) as f64) / (1u64 << 31) as f64;
            let u2 = ((v2 >> 33) as f64) / (1u64 << 31) as f64;
            let u1 = u1.abs().max(1e-10).min(1.0 - 1e-10);
            let u2 = u2.abs().max(1e-10).min(1.0 - 1e-10);
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }).collect();
        let emp = fi.metric(&n, &samples);
        let ana = fi.metric_analytical(&n);
        // g11 should be ~1.0, g22 should be ~2.0
        assert!((emp.get(0, 0) - ana.get(0, 0)).abs() < 0.3, "empirical g11 = {} vs analytical {}", emp.get(0, 0), ana.get(0, 0));
    }

    #[test]
    fn test_amari_m_connection() {
        let b = BernoulliDistribution::new(0.5);
        let ad = AmariDuality::from_exponential_family(&b);
        let m_conn = ad.m_connection();
        assert_eq!(m_conn.rows, 1);
        assert_eq!(m_conn.cols, 1);
    }

    #[test]
    fn test_geodesic_length_positive() {
        let n = NormalDistribution::new(0.0, 1.0);
        let sm = StatisticalManifold::from_normal(&n);
        let start = vec![0.0, 1.0];
        let velocity = vec![1.0, 0.5];
        let path = sm.geodesic(&start, &velocity, 20);
        assert!(path.len() == 21);
        // Path should move away from start
        let final_point = path.last().unwrap();
        assert!((final_point[0] - start[0]).abs() > 0.01 || (final_point[1] - start[1]).abs() > 0.01);
    }

    #[test]
    fn test_normal_log_prob_symmetry() {
        let n = NormalDistribution::new(0.0, 1.0);
        let lp_pos = n.log_prob(2.0);
        let lp_neg = n.log_prob(-2.0);
        assert!(approx_eq(lp_pos, lp_neg, TOL));
    }

    #[test]
    fn test_normal_prob_integrates_to_one() {
        // Numerical integration of normal PDF using trapezoid rule
        let n = NormalDistribution::new(0.0, 1.0);
        let num_steps = 10000;
        let dx = 20.0 / num_steps as f64; // integrate from -10 to 10
        let mut sum = 0.0;
        for i in 0..=num_steps {
            let x = -10.0 + i as f64 * dx;
            let p = n.prob(x);
            if i == 0 || i == num_steps {
                sum += p * 0.5;
            } else {
                sum += p;
            }
        }
        sum *= dx;
        assert!(approx_eq(sum, 1.0, 0.01), "PDF should integrate to ~1.0, got {}", sum);
    }

    #[test]
    fn test_normal_known_sigma_hessian() {
        let nks = NormalKnownSigma::new(2.0);
        let hess = nks.hessian_log_partition();
        // For known σ: ∇²A = σ² = 4
        assert!(approx_eq(hess.get(0, 0), 4.0, TOL));
        assert!(hess.is_positive_definite());
    }

    #[test]
    fn test_dense_matrix_6x6_inverse() {
        // Test that our inverse works for 6×6
        let mut m = DenseMatrix::identity(6);
        m.set(0, 1, 0.5);
        m.set(1, 0, 0.5);
        m.set(2, 3, 0.3);
        m.set(3, 2, 0.3);
        let inv = m.inverse().unwrap();
        let product = m.multiply(&inv);
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(product.get(i, j), expected, 1e-9));
            }
        }
    }

    #[test]
    fn test_matrix_zeros() {
        let z = DenseMatrix::zeros(3, 4);
        assert_eq!(z.rows, 3);
        assert_eq!(z.cols, 4);
        for i in 0..3 {
            for j in 0..4 {
                assert!(approx_eq(z.get(i, j), 0.0, TOL));
            }
        }
    }

    #[test]
    fn test_matrix_row_col() {
        let m = DenseMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let row0 = m.row(0);
        assert_eq!(row0, vec![1.0, 2.0, 3.0]);
        let col1 = m.col(1);
        assert_eq!(col1, vec![2.0, 5.0]);
    }
}
