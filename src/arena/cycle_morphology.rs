//! Mapping intransitive cycles onto morphological space.
//!
//! Given detected 3-cycles from the Balduzzi decomposition and morphological
//! descriptors for each creature, this module answers: *what body plan features
//! create the rock-paper-scissors dynamic?*
//!
//! Key insight (from Claudius): if creatures A, B, C form a cycle where A→B→C→A,
//! we can examine their morphological descriptors and identify which structural
//! features give each creature its edge over the next. This turns the cyclic
//! residual from a number into a narrative — exactly what makes it a paper figure.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::genotype::analysis::MorphologyDescriptor;

use super::decomposition::DecompositionResult;

/// A named morphological feature with its value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDifference {
    /// Human-readable feature name
    pub feature: String,
    /// Winner's value for this feature
    pub winner_value: f64,
    /// Loser's value for this feature
    pub loser_value: f64,
    /// Signed difference (positive = winner has more)
    pub difference: f64,
    /// Normalized difference: how many population SDs apart
    pub normalized_diff: f64,
}

/// Analysis of one edge in a cycle: why does the winner beat the loser?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeAnalysis {
    /// Winner creature ID
    pub winner_id: u64,
    /// Loser creature ID
    pub loser_id: u64,
    /// Cyclic matrix score for this edge (strength of the dominance)
    pub cyclic_score: f64,
    /// Features where the winner significantly exceeds the loser
    pub winner_advantages: Vec<FeatureDifference>,
    /// Features where the loser significantly exceeds the winner
    /// (potential vulnerabilities that the third creature exploits)
    pub loser_advantages: Vec<FeatureDifference>,
}

/// Complete morphological analysis of a single intransitive cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleMorphologyAnalysis {
    /// Creature IDs in cycle order: A→B→C→A (A beats B, B beats C, C beats A)
    pub creature_ids: [u64; 3],
    /// Edge analyses for each dominance relationship
    pub edges: [EdgeAnalysis; 3],
    /// Product of cyclic edge weights (overall cycle strength)
    pub cycle_strength: f64,
}

/// Full morphological mapping for all detected cycles in a tournament.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleMorphologyReport {
    /// Number of creatures in the tournament
    pub population_size: usize,
    /// Number of cycles detected
    pub cycle_count: usize,
    /// Per-cycle analyses, sorted by cycle strength (strongest first)
    pub cycles: Vec<CycleMorphologyAnalysis>,
    /// Population-wide feature statistics (for normalization context)
    pub feature_stats: Vec<FeatureStats>,
}

/// Population-level statistics for a single morphological feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub feature: String,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

/// Extract the scalar features from a MorphologyDescriptor as a named vector.
fn extract_features(desc: &MorphologyDescriptor) -> Vec<(&'static str, f64)> {
    let mut features = vec![
        ("node_count", desc.node_count as f64),
        ("connection_count", desc.connection_count as f64),
        ("total_volume", desc.total_volume as f64),
        ("mean_volume", desc.mean_volume as f64),
        ("volume_std", desc.volume_std as f64),
        ("max_depth", desc.max_depth as f64),
        ("mean_branching", desc.mean_branching as f64),
        ("leaf_count", desc.leaf_count as f64),
        ("total_dof", desc.total_dof as f64),
        ("total_neurons", desc.total_neurons as f64),
        ("total_sensors", desc.total_sensors as f64),
        ("total_effectors", desc.total_effectors as f64),
        ("symmetry_score", desc.symmetry_score as f64),
    ];

    // Joint type distribution
    let joint_names = [
        "joint_rigid",
        "joint_revolute",
        "joint_twist",
        "joint_universal",
        "joint_bendtwist",
        "joint_twistbend",
        "joint_spherical",
    ];
    for (i, name) in joint_names.iter().enumerate() {
        features.push((name, desc.joint_type_distribution[i] as f64));
    }

    features
}

/// Compute population-level statistics for feature normalization.
fn compute_feature_stats(descriptors: &[&MorphologyDescriptor]) -> Vec<FeatureStats> {
    if descriptors.is_empty() {
        return Vec::new();
    }

    let all_features: Vec<Vec<(&str, f64)>> = descriptors
        .iter()
        .map(|d| extract_features(d))
        .collect();

    let num_features = all_features[0].len();
    let n = descriptors.len() as f64;

    (0..num_features)
        .map(|fi| {
            let name = all_features[0][fi].0;
            let values: Vec<f64> = all_features.iter().map(|f| f[fi].1).collect();

            let mean = values.iter().sum::<f64>() / n;
            let variance = if n > 1.0 {
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
            } else {
                0.0
            };
            let std_dev = variance.sqrt();
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            FeatureStats {
                feature: name.to_string(),
                mean,
                std_dev,
                min,
                max,
            }
        })
        .collect()
}

/// Analyze morphological differences for a single edge (winner beats loser).
fn analyze_edge(
    winner_id: u64,
    loser_id: u64,
    winner_desc: &MorphologyDescriptor,
    loser_desc: &MorphologyDescriptor,
    cyclic_score: f64,
    feature_stats: &[FeatureStats],
    significance_threshold: f64,
) -> EdgeAnalysis {
    let winner_features = extract_features(winner_desc);
    let loser_features = extract_features(loser_desc);

    let mut winner_advantages = Vec::new();
    let mut loser_advantages = Vec::new();

    for (i, ((name, w_val), (_, l_val))) in winner_features
        .iter()
        .zip(loser_features.iter())
        .enumerate()
    {
        let diff = w_val - l_val;
        let std_dev = feature_stats[i].std_dev;
        let normalized = if std_dev > 1e-10 {
            diff / std_dev
        } else {
            0.0
        };

        if normalized.abs() < significance_threshold {
            continue;
        }

        let fd = FeatureDifference {
            feature: name.to_string(),
            winner_value: *w_val,
            loser_value: *l_val,
            difference: diff,
            normalized_diff: normalized,
        };

        if diff > 0.0 {
            winner_advantages.push(fd);
        } else {
            loser_advantages.push(fd);
        }
    }

    // Sort by magnitude of normalized difference (most distinctive first)
    winner_advantages.sort_by(|a, b| {
        b.normalized_diff
            .abs()
            .partial_cmp(&a.normalized_diff.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    loser_advantages.sort_by(|a, b| {
        b.normalized_diff
            .abs()
            .partial_cmp(&a.normalized_diff.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    EdgeAnalysis {
        winner_id,
        loser_id,
        cyclic_score,
        winner_advantages,
        loser_advantages,
    }
}

/// Build a complete morphological report for detected cycles.
///
/// # Arguments
/// * `decomposition` - Balduzzi decomposition result with detected cycles
/// * `creature_ids` - Ordered creature IDs matching the decomposition matrix indices
/// * `descriptors` - Map from creature ID to morphological descriptor
/// * `significance_threshold` - Minimum normalized difference (in SDs) to report
pub fn analyze_cycles(
    decomposition: &DecompositionResult,
    creature_ids: &[u64],
    descriptors: &HashMap<u64, MorphologyDescriptor>,
    significance_threshold: f64,
) -> CycleMorphologyReport {
    // Compute feature stats across all creatures with descriptors
    let all_descs: Vec<&MorphologyDescriptor> = creature_ids
        .iter()
        .filter_map(|id| descriptors.get(id))
        .collect();
    let feature_stats = compute_feature_stats(&all_descs);

    let mut cycles = Vec::new();

    for cycle_indices in &decomposition.detected_cycles {
        if cycle_indices.len() != 3 {
            continue;
        }

        let (i, j, k) = (cycle_indices[0], cycle_indices[1], cycle_indices[2]);

        // Map indices to creature IDs
        if i >= creature_ids.len() || j >= creature_ids.len() || k >= creature_ids.len() {
            continue;
        }
        let ids = [creature_ids[i], creature_ids[j], creature_ids[k]];

        // Get descriptors for all three creatures
        let descs: Vec<&MorphologyDescriptor> =
            match ids.iter().map(|id| descriptors.get(id)).collect::<Option<Vec<_>>>() {
                Some(d) => d,
                None => continue,
            };

        // Cycle order: i→j→k→i (i beats j, j beats k, k beats i)
        let edges = [
            analyze_edge(
                ids[0],
                ids[1],
                descs[0],
                descs[1],
                decomposition.cyclic_matrix[i][j],
                &feature_stats,
                significance_threshold,
            ),
            analyze_edge(
                ids[1],
                ids[2],
                descs[1],
                descs[2],
                decomposition.cyclic_matrix[j][k],
                &feature_stats,
                significance_threshold,
            ),
            analyze_edge(
                ids[2],
                ids[0],
                descs[2],
                descs[0],
                decomposition.cyclic_matrix[k][i],
                &feature_stats,
                significance_threshold,
            ),
        ];

        let cycle_strength = decomposition.cyclic_matrix[i][j]
            * decomposition.cyclic_matrix[j][k]
            * decomposition.cyclic_matrix[k][i];

        cycles.push(CycleMorphologyAnalysis {
            creature_ids: ids,
            edges,
            cycle_strength,
        });
    }

    // Sort by cycle strength (strongest first)
    cycles.sort_by(|a, b| {
        b.cycle_strength
            .abs()
            .partial_cmp(&a.cycle_strength.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    CycleMorphologyReport {
        population_size: creature_ids.len(),
        cycle_count: cycles.len(),
        cycles,
        feature_stats,
    }
}

/// Pretty-print a cycle morphology report (for CLI output).
pub fn format_report(report: &CycleMorphologyReport) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "=== Cycle Morphology Analysis ===\n\
         Population: {} creatures, {} cycles detected\n\n",
        report.population_size, report.cycle_count
    ));

    for (ci, cycle) in report.cycles.iter().enumerate() {
        out.push_str(&format!(
            "--- Cycle {} (strength: {:.4}) ---\n",
            ci + 1,
            cycle.cycle_strength
        ));
        out.push_str(&format!(
            "  {} → {} → {} → {}\n\n",
            cycle.creature_ids[0],
            cycle.creature_ids[1],
            cycle.creature_ids[2],
            cycle.creature_ids[0]
        ));

        for edge in &cycle.edges {
            out.push_str(&format!(
                "  {} beats {} (cyclic score: {:.3})\n",
                edge.winner_id, edge.loser_id, edge.cyclic_score
            ));

            if !edge.winner_advantages.is_empty() {
                out.push_str("    Winner advantages:\n");
                for fd in edge.winner_advantages.iter().take(3) {
                    out.push_str(&format!(
                        "      {:>20}: {:.2} vs {:.2} ({:+.1}σ)\n",
                        fd.feature, fd.winner_value, fd.loser_value, fd.normalized_diff
                    ));
                }
            }

            if !edge.loser_advantages.is_empty() {
                out.push_str("    Loser advantages:\n");
                for fd in edge.loser_advantages.iter().take(3) {
                    out.push_str(&format!(
                        "      {:>20}: {:.2} vs {:.2} ({:+.1}σ)\n",
                        fd.feature, fd.loser_value, fd.winner_value, fd.normalized_diff
                    ));
                }
            }
            out.push('\n');
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal MorphologyDescriptor for testing
    fn make_descriptor(
        node_count: usize,
        total_volume: f32,
        max_depth: usize,
        total_dof: usize,
        symmetry: f32,
    ) -> MorphologyDescriptor {
        MorphologyDescriptor {
            node_count,
            connection_count: node_count.saturating_sub(1),
            total_volume,
            mean_volume: if node_count > 0 {
                total_volume / node_count as f32
            } else {
                0.0
            },
            volume_std: 0.1,
            max_depth,
            mean_branching: if node_count > 1 { 1.0 } else { 0.0 },
            leaf_count: 1,
            total_dof,
            total_neurons: 2,
            total_sensors: 1,
            total_effectors: 1,
            symmetry_score: symmetry,
            joint_type_distribution: [0.0; 7],
        }
    }

    #[test]
    fn test_extract_features() {
        let desc = make_descriptor(5, 2.0, 3, 4, 0.8);
        let features = extract_features(&desc);
        assert!(!features.is_empty());
        assert_eq!(features[0], ("node_count", 5.0));
    }

    #[test]
    fn test_feature_stats() {
        let d1 = make_descriptor(3, 1.0, 2, 3, 0.5);
        let d2 = make_descriptor(7, 3.0, 4, 7, 0.9);
        let d3 = make_descriptor(5, 2.0, 3, 5, 0.7);
        let descs = vec![&d1, &d2, &d3];
        let stats = compute_feature_stats(&descs);

        // node_count: mean=(3+7+5)/3=5.0, std=2.0
        let nc_stat = &stats[0];
        assert_eq!(nc_stat.feature, "node_count");
        assert!((nc_stat.mean - 5.0).abs() < 0.01);
        assert!((nc_stat.std_dev - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_analyze_cycles_with_rps() {
        use crate::arena::decomposition::TransitiveCyclicDecomposition;

        // Rock-paper-scissors payoff
        let payoff = vec![
            vec![0.5, 1.0, 0.0],
            vec![0.0, 0.5, 1.0],
            vec![1.0, 0.0, 0.5],
        ];
        let decomp = TransitiveCyclicDecomposition::decompose(&payoff);

        // Three creatures with different morphologies
        let mut descriptors = HashMap::new();
        // Creature 1: big, deep, high DoF (exploits creature 2's simplicity)
        descriptors.insert(1, make_descriptor(8, 4.0, 5, 10, 0.3));
        // Creature 2: small, flat, symmetric (exploits creature 3's asymmetry)
        descriptors.insert(2, make_descriptor(3, 1.0, 1, 2, 0.9));
        // Creature 3: medium, flexible (exploits creature 1's bulk)
        descriptors.insert(3, make_descriptor(5, 2.5, 3, 6, 0.5));

        let creature_ids = vec![1, 2, 3];
        let report = analyze_cycles(&decomp, &creature_ids, &descriptors, 0.3);

        assert!(report.cycle_count > 0, "Should detect the RPS cycle");

        let cycle = &report.cycles[0];
        // Verify all edges have analyses
        for edge in &cycle.edges {
            assert!(
                !edge.winner_advantages.is_empty() || !edge.loser_advantages.is_empty(),
                "Each edge should have morphological differences: {} vs {}",
                edge.winner_id,
                edge.loser_id
            );
        }
    }

    #[test]
    fn test_format_report_not_empty() {
        use crate::arena::decomposition::TransitiveCyclicDecomposition;

        let payoff = vec![
            vec![0.5, 1.0, 0.0],
            vec![0.0, 0.5, 1.0],
            vec![1.0, 0.0, 0.5],
        ];
        let decomp = TransitiveCyclicDecomposition::decompose(&payoff);

        let mut descriptors = HashMap::new();
        descriptors.insert(1, make_descriptor(8, 4.0, 5, 10, 0.3));
        descriptors.insert(2, make_descriptor(3, 1.0, 1, 2, 0.9));
        descriptors.insert(3, make_descriptor(5, 2.5, 3, 6, 0.5));

        let report = analyze_cycles(&decomp, &[1, 2, 3], &descriptors, 0.3);
        let formatted = format_report(&report);

        assert!(formatted.contains("Cycle Morphology Analysis"));
        assert!(formatted.contains("beats"));
        assert!(formatted.len() > 100, "Report should have substantial content");
    }

    #[test]
    fn test_no_cycles_empty_report() {
        use crate::arena::decomposition::TransitiveCyclicDecomposition;

        // Fully transitive — no cycles
        let payoff = vec![
            vec![0.5, 0.7, 0.9],
            vec![0.3, 0.5, 0.7],
            vec![0.1, 0.3, 0.5],
        ];
        let decomp = TransitiveCyclicDecomposition::decompose(&payoff);

        let mut descriptors = HashMap::new();
        descriptors.insert(1, make_descriptor(5, 2.0, 3, 5, 0.5));
        descriptors.insert(2, make_descriptor(5, 2.0, 3, 5, 0.5));
        descriptors.insert(3, make_descriptor(5, 2.0, 3, 5, 0.5));

        let report = analyze_cycles(&decomp, &[1, 2, 3], &descriptors, 0.3);
        assert_eq!(report.cycle_count, 0, "Transitive ordering should have no cycles");
    }
}
