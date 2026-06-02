# information-geometry

> Information geometry in Rust. Where statistics meets differential geometry.

[![tests](https://img.shields.io/badge/tests-83-green)]()
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)]()

## What This Does

This crate implements **information geometry** — the study of statistical manifolds equipped with the Fisher information metric — with support for geodesics, natural gradient descent, exponential family duality, and optimal transport-inspired divergences.

It provides:

- **Fisher information metric** — both analytical (for known families) and empirical (from samples via numerical gradients)
- **5 exponential family distributions** — Normal, Normal (known σ), Normal (natural params), Bernoulli, Poisson, Exponential — each with sufficient statistics, log-partition, and Hessian
- **Statistical manifold geometry** — Christoffel symbols (analytically computed for the Normal manifold), geodesics via RK4 integration, exponential map, and parallel transport
- **Natural gradient descent** — G⁻¹∇f optimization that respects Riemannian curvature
- **KL divergence** — closed-form for normals, empirical from samples
- **Fisher-Rao distance** — geodesic distance between distributions (acosh formula for normals)
- **Amari's α-connections** — e-connection and m-connection duality for exponential families

Every type is `serde`-serializable. Pure Rust, no external solver dependencies.

## Install

```bash
cargo add information-geometry
```

## Quick Start

```rust
use information_geometry::*;

// Normal distribution and Fisher information
let normal = NormalDistribution::new(0.0, 1.0);
let fisher = normal.fisher_information();
// g = [[1, 0], [0, 2]] for N(0,1)

// Fisher-Rao geodesic distance between two normals
let fi = FisherInformation::new();
let p = NormalDistribution::new(0.0, 1.0);
let q = NormalDistribution::new(2.0, 3.0);
let dist = fi.fisher_rao_distance(&p, &q);

// KL divergence (closed form for normals)
let kl = fi.kl_divergence_normal(&p, &q);
assert!(kl >= 0.0); // Gibbs inequality

// Natural gradient descent
let ng = NaturalGradient::new();
let path = ng.natural_gradient_descent(
    &[5.0, 2.0],
    &|theta| theta[0] * theta[0] + (theta[1] - 1.0).powi(2),
    &|theta| vec![2.0 * theta[0], 2.0 * (theta[1] - 1.0)],
    &|theta| {
        let s = theta[1].max(0.1);
        DenseMatrix::diagonal(&[1.0 / (s * s), 2.0 / (s * s)])
    },
    0.1, 50,
);

// Geodesic on the Normal manifold
let sm = StatisticalManifold::from_normal(&NormalDistribution::new(0.0, 1.0));
let geodesic = sm.geodesic(&[0.0, 1.0], &[0.5, 0.0], 20);
```

## Testing

**83 tests** covering all functionality. Run with:

```bash
cargo test
```

## License

MIT OR Apache-2.0
