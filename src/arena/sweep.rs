//! Steepness sweep: robustness analysis for tournament decomposition.
//!
//! Runs the same tournament at multiple sigmoid steepness values and
//! compares the decomposition results. If cycle strength and rankings
//! are stable across steepness values, the results are robust. If they
//! vary wildly, the intransitivity is an artifact of the outcome model.
//!
//! This is exactly the kind of sensitivity analysis reviewers will ask for.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::criteria::CriterionId;
use super::decomposition::DecompositionResult;
use super::ratings::EloConfig;
use super::tournament::{Tournament, TournamentConfig, TournamentResults};

/// Result of a single steepness probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteepnessProbe {
    /// The steepness value used.
    pub steepness: f64,
    /// Cycle strength from the decomposition.
    pub cycle_strength: f64,
    /// Top-ranked creature ID.
    pub top_creature: u64,
    /// Top Elo rating.
    pub top_rating: f64,
    /// Number of detected 3-cycles.
    pub cycle_count: usize,
    /// Rank correlation (Spearman) with the baseline steepness.
    pub rank_correlation: Option<f64>,
    /// Full tournament results (optional, for deeper analysis).
    #[serde(skip)]
    pub full_results: Option<TournamentResults>,
}

/// Complete steepness sweep results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteepnessSweep {
    /// The steepness values tested.
    pub steepness_values: Vec<f64>,
    /// Results for each steepness value.
    pub probes: Vec<SteepnessProbe>,
    /// Baseline steepness (used for rank correlation).
    pub baseline_steepness: f64,
    /// Is the decomposition robust? (cycle strength variation < threshold)
    pub is_robust: bool,
    /// Coefficient of variation of cycle strength across steepness values.
    pub cycle_strength_cv: f64,
    /// Mean rank correlation with baseline.
    pub mean_rank_correlation: f64,
}

/// Run a steepness sweep on pre-computed scores.
///
/// Tests the tournament at each steepness value in `steepness_values`,
/// comparing cycle strength and rankings against the baseline.
pub fn run_sweep(
    scores: &HashMap<u64, f32>,
    steepness_values: &[f64],
    baseline_steepness: f64,
    rounds_per_pair: usize,
    criterion: CriterionId,
) -> SteepnessSweep {
    // Run baseline tournament first
    let baseline_config = TournamentConfig {
        elo_config: EloConfig::default(),
        rounds_per_pair,
        criterion: criterion.clone(),
        sigmoid_steepness: baseline_steepness,
    };
    let baseline_results = Tournament::run_from_scores(baseline_config, scores);
    let baseline_ranking: Vec<u64> = baseline_results
        .leaderboard()
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    let mut probes = Vec::new();

    for &steepness in steepness_values {
        let config = TournamentConfig {
            elo_config: EloConfig::default(),
            rounds_per_pair,
            criterion: criterion.clone(),
            sigmoid_steepness: steepness,
        };
        let results = Tournament::run_from_scores(config, scores);

        let cycle_strength = results.cycle_strength().unwrap_or(0.0);
        let leaderboard = results.leaderboard();
        let top_creature = leaderboard.first().map(|(id, _)| *id).unwrap_or(0);
        let top_rating = leaderboard.first().map(|(_, r)| *r).unwrap_or(0.0);
        let cycle_count = results
            .decomposition
            .as_ref()
            .map(|d| d.detected_cycles.len())
            .unwrap_or(0);

        // Compute rank correlation with baseline
        let current_ranking: Vec<u64> = leaderboard.into_iter().map(|(id, _)| id).collect();
        let rank_correlation = spearman_rank_correlation(&baseline_ranking, &current_ranking);

        probes.push(SteepnessProbe {
            steepness,
            cycle_strength,
            top_creature,
            top_rating,
            cycle_count,
            rank_correlation: Some(rank_correlation),
            full_results: Some(results),
        });
    }

    // Compute summary statistics
    let cycle_strengths: Vec<f64> = probes.iter().map(|p| p.cycle_strength).collect();
    let mean_cs: f64 = cycle_strengths.iter().sum::<f64>() / cycle_strengths.len() as f64;
    let std_cs: f64 = if cycle_strengths.len() > 1 {
        let variance = cycle_strengths
            .iter()
            .map(|cs| (cs - mean_cs).powi(2))
            .sum::<f64>()
            / (cycle_strengths.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };
    let cycle_strength_cv = if mean_cs.abs() > 1e-10 {
        std_cs / mean_cs
    } else {
        0.0
    };

    let rank_correlations: Vec<f64> = probes
        .iter()
        .filter_map(|p| p.rank_correlation)
        .collect();
    let mean_rank_correlation = if !rank_correlations.is_empty() {
        rank_correlations.iter().sum::<f64>() / rank_correlations.len() as f64
    } else {
        0.0
    };

    // Robust if CV < 0.3 and mean rank correlation > 0.8
    let is_robust = cycle_strength_cv < 0.3 && mean_rank_correlation > 0.8;

    SteepnessSweep {
        steepness_values: steepness_values.to_vec(),
        probes,
        baseline_steepness,
        is_robust,
        cycle_strength_cv,
        mean_rank_correlation,
    }
}

/// Default steepness values for a standard sweep.
pub fn default_steepness_values() -> Vec<f64> {
    vec![1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
}

/// Compute Spearman rank correlation between two rankings.
///
/// Both inputs are ordered lists of creature IDs (best first).
/// Returns a value in [-1, 1] where 1 = identical rankings.
fn spearman_rank_correlation(ranking_a: &[u64], ranking_b: &[u64]) -> f64 {
    if ranking_a.is_empty() {
        return 0.0;
    }

    let n = ranking_a.len();

    // Build rank maps (creature_id -> rank)
    let rank_a: HashMap<u64, usize> = ranking_a
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let rank_b: HashMap<u64, usize> = ranking_b
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Compute sum of squared rank differences
    let mut d_sq_sum: f64 = 0.0;
    for &id in ranking_a {
        let ra = rank_a.get(&id).copied().unwrap_or(n) as f64;
        let rb = rank_b.get(&id).copied().unwrap_or(n) as f64;
        d_sq_sum += (ra - rb).powi(2);
    }

    let n_f = n as f64;
    1.0 - (6.0 * d_sq_sum) / (n_f * (n_f * n_f - 1.0))
}

/// Pretty-print a steepness sweep result.
pub fn format_sweep(sweep: &SteepnessSweep) -> String {
    let mut out = String::new();

    out.push_str("=== Steepness Sweep (Sensitivity Analysis) ===\n\n");
    out.push_str(&format!(
        "Baseline steepness: {:.1}\n",
        sweep.baseline_steepness
    ));
    out.push_str(&format!(
        "Robustness: {} (CV={:.3}, mean rank corr={:.3})\n\n",
        if sweep.is_robust { "ROBUST" } else { "FRAGILE" },
        sweep.cycle_strength_cv,
        sweep.mean_rank_correlation,
    ));

    out.push_str("  Steepness | Cycle Str | Cycles | Top Creature | Rank Corr\n");
    out.push_str("  ----------+-----------+--------+--------------+----------\n");

    for probe in &sweep.probes {
        out.push_str(&format!(
            "  {:>9.1} | {:>9.3} | {:>6} | {:>12} | {:>9.3}\n",
            probe.steepness,
            probe.cycle_strength,
            probe.cycle_count,
            probe.top_creature,
            probe.rank_correlation.unwrap_or(0.0),
        ));
    }

    out.push('\n');
    if sweep.is_robust {
        out.push_str("  Decomposition results are stable across steepness values.\n");
        out.push_str("  The intransitive dynamics are a real property of the population,\n");
        out.push_str("  not an artifact of the score-to-outcome conversion.\n");
    } else {
        out.push_str("  WARNING: Decomposition results vary with steepness.\n");
        out.push_str("  The intransitive dynamics may be sensitive to the outcome model.\n");
        out.push_str("  Consider reporting results at multiple steepness values.\n");
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spearman_identical() {
        let ranking = vec![1, 2, 3, 4, 5];
        let corr = spearman_rank_correlation(&ranking, &ranking);
        assert!((corr - 1.0).abs() < 0.001, "Identical rankings should have correlation 1.0: {}", corr);
    }

    #[test]
    fn test_spearman_reversed() {
        let ranking_a = vec![1, 2, 3, 4, 5];
        let ranking_b = vec![5, 4, 3, 2, 1];
        let corr = spearman_rank_correlation(&ranking_a, &ranking_b);
        assert!((corr - (-1.0)).abs() < 0.001, "Reversed rankings should have correlation -1.0: {}", corr);
    }

    #[test]
    fn test_sweep_transitive_is_robust() {
        // Clear transitive ordering should be robust across steepness values
        let mut scores = HashMap::new();
        scores.insert(1, 10.0);
        scores.insert(2, 5.0);
        scores.insert(3, 2.0);
        scores.insert(4, 0.5);

        let steepness_values = vec![1.0, 3.0, 5.0, 10.0, 20.0];
        let sweep = run_sweep(&scores, &steepness_values, 5.0, 3, CriterionId::LocomotionDistance);

        // Rankings should be very stable for clearly separated scores
        assert!(
            sweep.mean_rank_correlation > 0.8,
            "Transitive ordering should be stable: rank_corr={}",
            sweep.mean_rank_correlation
        );
    }

    #[test]
    fn test_sweep_has_all_probes() {
        let mut scores = HashMap::new();
        scores.insert(1, 10.0);
        scores.insert(2, 5.0);
        scores.insert(3, 1.0);

        let steepness_values = vec![1.0, 5.0, 10.0];
        let sweep = run_sweep(&scores, &steepness_values, 5.0, 3, CriterionId::LocomotionDistance);

        assert_eq!(sweep.probes.len(), 3);
        assert_eq!(sweep.steepness_values.len(), 3);
    }

    #[test]
    fn test_format_sweep() {
        let mut scores = HashMap::new();
        scores.insert(1, 10.0);
        scores.insert(2, 5.0);
        scores.insert(3, 1.0);

        let steepness_values = vec![1.0, 5.0, 10.0];
        let sweep = run_sweep(&scores, &steepness_values, 5.0, 3, CriterionId::LocomotionDistance);

        let formatted = format_sweep(&sweep);
        assert!(formatted.contains("Steepness Sweep"));
        assert!(formatted.contains("Cycle Str"));
    }
}
