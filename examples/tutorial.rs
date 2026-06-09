//! # Information Geometry Tutorial
//!
//! A progressive, hands-on tour of information geometry in Rust.
//! Each lesson teaches ONE concept with clear println output.
//!
//! Run with: cargo run --example tutorial

use information_geometry::*;

fn separator(title: &str) {
    println!("\n{}", "═".repeat(60));
    println!("  Lesson: {}", title);
    println!("{}\n", "═".repeat(60));
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     Information Geometry — Progressive Tutorial         ║");
    println!("║     Statistical manifolds, Fisher info, geodesics       ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    lesson_01_distributions_and_log_prob();
    lesson_02_fisher_information_matrix();
    lesson_03_kl_divergence();
    lesson_04_fisher_rao_distance();
    lesson_05_exponential_families();
    lesson_06_natural_gradient();
    lesson_07_geodesics_and_parallel_transport();
    lesson_08_amari_duality();
}

// ─── Lesson 1: Probability Distributions ─────────────────────

fn lesson_01_distributions_and_log_prob() {
    separator("1 — Normal Distribution & Log-Probability");

    let standard = NormalDistribution::standard();
    let custom = NormalDistribution::new(5.0, 2.0);

    println!("A Normal distribution N(μ, σ²) is a point on a statistical manifold.");
    println!();

    println!("Standard normal N(0, 1):");
    println!("  log p(0.0)  = {:.6}", standard.log_prob(0.0));
    println!("  log p(1.0)  = {:.6}", standard.log_prob(1.0));
    println!("  log p(-1.0) = {:.6}", standard.log_prob(-1.0));
    println!();

    println!("Custom normal N(5, 4):");
    println!("  μ = {:.1}, σ = {:.1}, σ² = {:.1}", custom.mu, custom.sigma, custom.variance());
    println!("  Parameters (μ, σ) = {:?}", custom.parameters());
    println!();

    println!("Log-probability is symmetric around the mean:");
    println!("  log p(μ + δ) = log p(μ - δ)");
    println!("  log p(4.0) = {:.6}", custom.log_prob(4.0));
    println!("  log p(6.0) = {:.6}", custom.log_prob(6.0));
    println!();

    // Gradient of log-probability: the score function
    println!("Score function ∇ log p(x|θ):");
    let grad = custom.grad_log_prob(7.0);
    println!("  At x=7.0: ∂/∂μ = {:.4}, ∂/∂σ = {:.4}", grad[0], grad[1]);
    println!("  ∂/∂μ = (x - μ)/σ²  tells us how to shift the mean");
    println!("  ∂/∂σ = (x-μ)²/σ³ - 1/σ  tells us how to adjust spread");
}

// ─── Lesson 2: Fisher Information Matrix ─────────────────────

fn lesson_02_fisher_information_matrix() {
    separator("2 — Fisher Information: The Metric of Statistics");

    println!("The Fisher information matrix G is the Riemannian metric on");
    println!("the manifold of probability distributions.");
    println!();

    // Analytical Fisher information for Normal distribution
    let normal = NormalDistribution::new(0.0, 1.0);
    let fisher = normal.fisher_information();
    println!("Fisher information for N(0, 1):");
    println!("  G = {}", fisher);
    println!();

    println!("Theorem: For N(μ, σ²), the Fisher metric is:");
    println!("  g_μμ = 1/σ² = {:.4}", fisher.get(0, 0));
    println!("  g_σσ = 2/σ² = {:.4}", fisher.get(1, 1));
    println!("  g_μσ = 0     = {:.4}", fisher.get(0, 1));
    println!();

    // Effect of σ on the metric
    println!("How σ changes the geometry:");
    for sigma in [0.5, 1.0, 2.0, 4.0] {
        let n = NormalDistribution::new(0.0, sigma);
        let fi = n.fisher_information();
        println!("  σ = {:.1} → g_μμ = {:.4}, g_σσ = {:.4} (more σ = flatter manifold)",
                 sigma, fi.get(0, 0), fi.get(1, 1));
    }
    println!();

    // Cramér-Rao bound
    println!("Cramér-Rao Bound: Var(θ̂) ≥ I(θ)⁻¹ / n");
    let sigma = 2.0;
    let n_samples = 100.0;
    let fisher_mu = 1.0 / (sigma * sigma);
    let crlb = 1.0 / (n_samples * fisher_mu);
    let var_sample_mean = sigma * sigma / n_samples;
    println!("  For μ estimation: I_μμ = {:.4}, CRLB = {:.6}", fisher_mu, crlb);
    println!("  Var(X̄) = σ²/n = {:.6}", var_sample_mean);
    println!("  → X̄ achieves the Cramér-Rao bound (efficient estimator!) ✓");
}

// ─── Lesson 3: KL Divergence ─────────────────────────────────

fn lesson_03_kl_divergence() {
    separator("3 — KL Divergence: Asymmetric Distributional Distance");

    let fi = FisherInformation::new();
    let p = NormalDistribution::new(0.0, 1.0);
    let q = NormalDistribution::new(2.0, 1.0);
    let r = NormalDistribution::new(0.0, 3.0);

    println!("KL divergence measures how much information is lost when");
    println!("using q to approximate p.");
    println!();

    // Same distribution
    let kl_same = fi.kl_divergence_normal(&p, &p);
    println!("KL(p || p) = {:.6}  (always zero)", kl_same);

    // Shifted mean
    let kl_pq = fi.kl_divergence_normal(&p, &q);
    let kl_qp = fi.kl_divergence_normal(&q, &p);
    println!("KL(N(0,1) || N(2,1)) = {:.6}", kl_pq);
    println!("KL(N(2,1) || N(0,1)) = {:.6}", kl_qp);
    println!("→ KL is NOT symmetric! These differ by factor ≈ 1");
    println!();

    // Different variance
    let kl_pr = fi.kl_divergence_normal(&p, &r);
    println!("KL(N(0,1) || N(0,3)) = {:.6}", kl_pr);
    println!("  = ln(3/1) + 1/(2·9) - 0.5 = {:.6}", (3.0_f64).ln() + 1.0/18.0 - 0.5);
    println!();

    // Formula breakdown
    println!("Formula: KL(p||q) = ln(σ_q/σ_p) + (σ_p² + (μ_p-μ_q)²)/(2σ_q²) - 1/2");
    println!("  Term 1: ln(σ_q/σ_p) = {:.4}  (spread mismatch)", (q.sigma / p.sigma).ln());
    println!("  Term 2: (σ_p² + Δμ²)/(2σ_q²) = {:.4}  (location + spread)", 
             (p.sigma.powi(2) + (p.mu - q.mu).powi(2)) / (2.0 * q.sigma.powi(2)));
    println!("  Term 3: -1/2 = -0.5000");
}

// ─── Lesson 4: Fisher-Rao Distance ───────────────────────────

fn lesson_04_fisher_rao_distance() {
    separator("4 — Fisher-Rao Distance: The Geodesic on Stat Space");

    let fi = FisherInformation::new();

    println!("Fisher-Rao distance is the TRUE geodesic distance on the");
    println!("manifold of probability distributions — unlike KL, it is");
    println!("symmetric and satisfies the triangle inequality.");
    println!();

    // Known-sigma case (simple)
    println!("Known-σ case: d = |μ₁ - μ₂| / σ");
    let d = fi.fisher_rao_distance_known_sigma(0.0, 3.0, 1.0);
    println!("  d(N(0,1), N(3,1)) = |0-3|/1 = {:.4}", d);
    println!();

    // Full manifold
    let p = NormalDistribution::new(0.0, 1.0);
    let q = NormalDistribution::new(2.0, 3.0);
    let dist = fi.fisher_rao_distance(&p, &q);
    println!("Full manifold distance:");
    println!("  d(N(0,1), N(2,3)) = {:.6}", dist);
    println!();

    // Symmetry check
    let dist_rev = fi.fisher_rao_distance(&q, &p);
    println!("Symmetry check: d(q,p) = {:.6} ≈ d(p,q) = {:.6} ✓", dist_rev, dist);
    println!();

    // Distance landscape
    println!("Distance from N(0,1) to various distributions:");
    let base = NormalDistribution::new(0.0, 1.0);
    for (mu, sigma) in [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (0.0, 2.0), (2.0, 2.0)] {
        let target = NormalDistribution::new(mu, sigma);
        let d = fi.fisher_rao_distance(&base, &target);
        println!("  d(N(0,1), N({},{:.0})) = {:.4}", mu, sigma, d);
    }
}

// ─── Lesson 5: Exponential Families ──────────────────────────

fn lesson_05_exponential_families() {
    separator("5 — Exponential Families & Natural Parametrization");

    println!("Exponential families: p(x|η) = h(x) exp(η·T(x) - A(η))");
    println!("where η = natural params, T(x) = sufficient stats, A(η) = log-partition");
    println!();

    // Bernoulli
    println!("── Bernoulli Distribution ──");
    let bern = BernoulliDistribution::new(0.3);
    println!("  p = 0.3");
    println!("  Natural param η = logit(p) = ln(p/(1-p)) = {:.4}", bern.natural_param());
    println!("  Sufficient stat T(x) = x");
    println!("  Log-partition A(η) = ln(1 + eᶯ) = {:.4}", bern.log_partition());
    println!("  E[T(x)] = p = {:.4}", bern.grad_log_partition()[0]);
    println!("  Var[T(x)] = p(1-p) = {:.4}", bern.hessian_log_partition().get(0, 0));
    println!();

    // Poisson
    println!("── Poisson Distribution ──");
    let pois = PoissonDistribution::new(5.0);
    println!("  λ = 5.0");
    println!("  Natural param η = ln(λ) = {:.4}", pois.natural_parameters()[0]);
    println!("  Log-partition A(η) = eᶯ = λ = {:.4}", pois.log_partition());
    println!("  E[X] = λ = {:.4}", pois.grad_log_partition()[0]);
    println!("  Var[X] = λ = {:.4}", pois.hessian_log_partition().get(0, 0));
    println!();

    // Normal in natural coordinates
    println!("── Normal Distribution (full natural coords) ──");
    let nn = NormalNatural::from_mu_sigma(2.0, 1.5);
    println!("  μ = 2.0, σ = 1.5");
    println!("  η₁ = μ/σ² = {:.4}", nn.eta1);
    println!("  η₂ = -1/(2σ²) = {:.4}", nn.eta2);
    println!("  Roundtrip: μ = {:.4}, σ = {:.4}", nn.mu(), nn.sigma());
    println!("  Sufficient stats T(x=3) = {:?}", nn.sufficient_statistics(3.0));
    println!("  E[T₁] = E[X] = {:.4}", nn.grad_log_partition()[0]);
    println!("  E[T₂] = E[X²] = μ²+σ² = {:.4}", nn.grad_log_partition()[1]);
    println!();

    // Key theorem
    println!("KEY THEOREM: The Fisher metric G = ∇²A(η) (Hessian of log-partition)");
    let hess = nn.hessian_log_partition();
    println!("  ∇²A = {}", hess);
    println!("  Positive definite? {} ✓", hess.is_positive_definite());
}

// ─── Lesson 6: Natural Gradient ──────────────────────────────

fn lesson_06_natural_gradient() {
    separator("6 — Natural Gradient: Geometry-Aware Optimization");

    let ng = NaturalGradient::new();

    println!("Standard gradient ∇f treats all directions equally.");
    println!("Natural gradient G⁻¹∇f accounts for manifold curvature.");
    println!();

    // Simple example: anisotropic metric
    println!("── Anisotropic Metric Example ──");
    let metric = DenseMatrix::diagonal(&[100.0, 1.0]);
    let euc_grad = vec![1.0, 1.0];
    let nat_grad = ng.compute(&metric, &euc_grad);
    println!("  Euclidean gradient: {:?}", euc_grad);
    println!("  Fisher metric G = diag(100, 1)");
    println!("  Natural gradient G⁻¹∇f = [{:.4}, {:.4}]", nat_grad[0], nat_grad[1]);
    println!("  → Direction 1 has 100× less natural gradient despite equal");
    println!("    Euclidean gradient — the manifold is very stiff in that direction.");
    println!();

    // Natural gradient descent
    println!("── Natural Gradient Descent on Normal Manifold ──");
    let start = vec![5.0, 2.0];
    println!("  Start: μ = {:.1}, σ = {:.1}", start[0], start[1]);
    println!("  Objective: minimize μ² + (σ - 1)²");

    let path = ng.natural_gradient_descent(
        &start,
        &|theta| theta[0] * theta[0] + (theta[1] - 1.0).powi(2),
        &|theta| vec![2.0 * theta[0], 2.0 * (theta[1] - 1.0)],
        &|theta| {
            let sigma = theta[1].max(0.1);
            DenseMatrix::diagonal(&[1.0 / (sigma * sigma), 2.0 / (sigma * sigma)])
        },
        0.1,
        50,
    );

    println!("  Trajectory (every 10 steps):");
    for (i, theta) in path.iter().enumerate().step_by(10) {
        println!("    step {:3}: μ = {:+.4}, σ = {:.4}", i, theta[0], theta[1]);
    }
    let final_t = path.last().unwrap();
    println!("  Final: μ = {:+.6}, σ = {:.6}", final_t[0], final_t[1]);
}

// ─── Lesson 7: Geodesics & Parallel Transport ────────────────

fn lesson_07_geodesics_and_parallel_transport() {
    separator("7 — Geodesics & Parallel Transport on the Normal Manifold");

    let normal = NormalDistribution::new(0.0, 1.0);
    let manifold = StatisticalManifold::from_normal(&normal);

    // Christoffel symbols
    println!("── Christoffel Symbols (connection coefficients) ──");
    let gamma = manifold.christoffel_symbols();
    println!("  For N(0,1), parameters (μ, σ):");
    println!("  Γ¹₁₂ = Γ¹₂₁ = {:.4}  (μ-σ coupling)", gamma[0][0][1]);
    println!("  Γ²₁₁ = {:.4}          (μ-μ → σ acceleration)", gamma[0][0][1]); // same by symmetry
    println!("  Γ²₂₂ = {:.4}          (σ-σ → σ acceleration)", gamma[1][1][1]);
    println!();

    // Geodesic
    println!("── Geodesic from (μ=0, σ=1) with velocity (0.5, 0.1) ──");
    let start = vec![0.0, 1.0];
    let velocity = vec![0.5, 0.1];
    let path = manifold.geodesic(&start, &velocity, 20);

    println!("  Step |   μ      |   σ");
    println!("  -----+----------+----------");
    for (i, point) in path.iter().enumerate().step_by(5) {
        println!("  {:4} | {:+8.4} | {:8.4}", i, point[0], point[1]);
    }
    println!();

    // Exponential map
    println!("── Exponential Map ──");
    let point = vec![0.0, 1.0];
    let tangent = vec![1.0, 0.0];
    let result = manifold.exponential_map(&point, &tangent, 50);
    println!("  exp_(μ=0,σ=1)(v = [1.0, 0.0]) = (μ={:.4}, σ={:.4})", result[0], result[1]);
    println!();

    // Parallel transport
    println!("── Parallel Transport ──");
    let curve = vec![
        vec![0.0, 1.0],
        vec![0.5, 1.0],
        vec![1.0, 1.0],
        vec![1.5, 1.0],
        vec![2.0, 1.0],
    ];
    let vector = vec![1.0, 0.0];
    let transported = manifold.parallel_transport(&vector, &curve);
    println!("  Transporting v = [1.0, 0.0] along μ-axis (σ=1):");
    for (i, v) in transported.iter().enumerate() {
        println!("    At μ = {:.1}: v = [{:.4}, {:.4}]", curve[i][0], v[0], v[1]);
    }
    println!("  → On a flat manifold, parallel transport doesn't change the vector.");
}

// ─── Lesson 8: Amari Duality ─────────────────────────────────

fn lesson_08_amari_duality() {
    separator("8 — Amari α-Connections & e/m Duality");

    println!("Amari's framework gives two flat connections on exponential families:");
    println!("  e-connection (α=+1): flat in natural coordinates η");
    println!("  m-connection (α=-1): flat in expectation coordinates μ = ∇A(η)");
    println!();

    // Bernoulli duality
    println!("── Bernoulli Duality ──");
    let bern = BernoulliDistribution::new(0.3);
    let amari = AmariDuality::from_exponential_family(&bern);

    println!("  p = 0.3");
    println!("  Natural param η = logit(0.3) = {:.4}", bern.natural_param());
    println!("  Expectation param μ = E[T(x)] = p = {:.4}", amari.dual_parameters(&bern)[0]);
    println!();

    let e_conn = amari.e_connection();
    println!("  e-connection in natural coords: Γ⁽⁺¹⁾ = {}", e_conn.get(0, 0));
    println!("  → The e-connection VANISHES in natural coordinates (flat!) ✓");
    println!();

    // Poisson duality
    println!("── Poisson Duality ──");
    let pois = PoissonDistribution::new(4.0);
    let amari_pois = AmariDuality::from_exponential_family(&pois);
    println!("  λ = 4.0");
    println!("  η = ln(4) = {:.4}", pois.natural_parameters()[0]);
    println!("  E[X] = {:.4}", amari_pois.dual_parameters(&pois)[0]);
    println!("  Log-partition = {:.4}", pois.log_partition());
    println!();

    // Verification theorem
    println!("── Verification: Metric = Hessian of Log-Partition ──");
    let distributions = [
        ("Bernoulli(0.5)", AmariDuality::verify_hessian_metric(&BernoulliDistribution::new(0.5)) as bool),
        ("Poisson(3)", AmariDuality::verify_hessian_metric(&PoissonDistribution::new(3.0)) as bool),
        ("Exponential(2)", AmariDuality::verify_hessian_metric(&ExponentialDist::new(2.0)) as bool),
    ];
    for (name, is_pd) in &distributions {
        println!("  {} → Hessian positive definite? {} ✓", name, is_pd);
    }
    println!();

    println!("── The Big Picture ──");
    println!("  1. Every exponential family has TWO natural coordinate systems:");
    println!("     η (natural) ↔ μ (expectation), connected by ∇A(η)");
    println!("  2. The Fisher metric G = ∇²A(η) bridges these coordinate systems");
    println!("  3. Natural gradient = G⁻¹∇f uses this geometry for optimization");
    println!("  4. Geodesics give shortest paths between distributions");
    println!("  5. Parallel transport moves vectors along curves on the manifold");
}
