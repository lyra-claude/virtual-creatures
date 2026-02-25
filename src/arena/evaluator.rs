//! Evaluation bridge: connects the physics simulation to the tournament system.
//!
//! The `Evaluator` trait is the interface that Claudius's neural evaluation
//! pipeline and Nick's Bevy/Rapier simulation implement. The tournament
//! system only sees scores — how those scores are produced is the evaluator's
//! business.
//!
//! Design decision: evaluate all criteria in a single simulation run rather
//! than re-running per criterion. This means one headless spawn → simulate →
//! measure, producing a multi-score result.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::criteria::CriterionId;
use crate::genotype::morphology::CreatureGenotype;

/// Result of evaluating a single creature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Scores per fitness criterion (all measured in a single simulation run).
    pub scores: HashMap<CriterionId, f32>,
    /// Number of simulation timesteps executed.
    pub simulation_steps: usize,
    /// Wall-clock time in milliseconds.
    pub wall_time_ms: u64,
}

impl EvaluationResult {
    /// Get the score for a specific criterion, defaulting to 0.
    pub fn score(&self, criterion: &CriterionId) -> f32 {
        self.scores.get(criterion).copied().unwrap_or(0.0)
    }
}

/// Configuration for the evaluation environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Number of simulation timesteps per evaluation.
    pub timesteps: usize,
    /// Physics timestep in seconds.
    pub dt: f32,
    /// Which criteria to measure.
    pub criteria: Vec<CriterionId>,
    /// Random seed for reproducibility (perturbation tests, etc.)
    pub seed: Option<u64>,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            timesteps: 1000,
            dt: 1.0 / 60.0,
            criteria: vec![CriterionId::LocomotionDistance],
            seed: None,
        }
    }
}

/// The evaluation bridge trait.
///
/// Implementations handle the full pipeline:
/// 1. Decode genotype → phenotype (body plan + neural controller)
/// 2. Spawn creature in a headless physics world
/// 3. Step the simulation with the neural controller running
/// 4. Measure fitness criteria and return scores
///
/// The tournament system calls `evaluate()` for each creature and uses
/// the resulting scores to run pairwise comparisons.
pub trait Evaluator: Send + Sync {
    /// Evaluate a single creature and return its scores.
    fn evaluate(
        &self,
        genotype: &CreatureGenotype,
        config: &EvaluationConfig,
    ) -> EvaluationResult;

    /// Evaluate a batch of creatures (default: sequential).
    ///
    /// Override for parallel evaluation if the implementation supports it.
    fn evaluate_batch(
        &self,
        genotypes: &[(u64, &CreatureGenotype)],
        config: &EvaluationConfig,
    ) -> HashMap<u64, EvaluationResult> {
        genotypes
            .iter()
            .map(|(id, geno)| (*id, self.evaluate(geno, config)))
            .collect()
    }
}

/// A no-op evaluator that returns zeros. Useful for testing tournament
/// logic without a physics simulation.
pub struct NullEvaluator;

impl Evaluator for NullEvaluator {
    fn evaluate(
        &self,
        _genotype: &CreatureGenotype,
        config: &EvaluationConfig,
    ) -> EvaluationResult {
        let scores = config
            .criteria
            .iter()
            .map(|c| (c.clone(), 0.0))
            .collect();
        EvaluationResult {
            scores,
            simulation_steps: config.timesteps,
            wall_time_ms: 0,
        }
    }
}

/// A score-based evaluator that looks up pre-computed fitness values.
///
/// This is what the current tournament uses: creatures already have
/// fitness scores from evolution, so we just look them up rather than
/// re-simulating. Useful as a fallback and for testing.
pub struct PrecomputedEvaluator {
    scores: HashMap<u64, HashMap<CriterionId, f32>>,
}

impl PrecomputedEvaluator {
    /// Create from a simple fitness map (single criterion).
    pub fn from_fitness_map(
        fitness: &HashMap<u64, f32>,
        criterion: CriterionId,
    ) -> Self {
        let scores = fitness
            .iter()
            .map(|(&id, &score)| {
                let mut criteria_scores = HashMap::new();
                criteria_scores.insert(criterion.clone(), score);
                (id, criteria_scores)
            })
            .collect();
        Self { scores }
    }

    /// Get scores for a creature ID.
    pub fn get_scores(&self, id: u64) -> Option<&HashMap<CriterionId, f32>> {
        self.scores.get(&id)
    }
}

impl Evaluator for PrecomputedEvaluator {
    fn evaluate(
        &self,
        _genotype: &CreatureGenotype,
        config: &EvaluationConfig,
    ) -> EvaluationResult {
        // PrecomputedEvaluator doesn't use the genotype — it needs the creature ID.
        // This is a design tension: the trait takes a genotype but this impl
        // needs an ID. For now, return zeros. Use `get_scores()` directly
        // for ID-based lookups.
        let scores = config
            .criteria
            .iter()
            .map(|c| (c.clone(), 0.0))
            .collect();
        EvaluationResult {
            scores,
            simulation_steps: 0,
            wall_time_ms: 0,
        }
    }
}

/// Run a full evaluated tournament: evaluate all creatures, then run the tournament.
///
/// This is the "real" tournament pipeline that replaces `Tournament::run_from_scores`
/// once the evaluation bridge is functional. Each creature is evaluated using the
/// provided evaluator, producing multi-criterion scores, and then a tournament is
/// run for each criterion independently.
pub fn run_evaluated_tournament(
    evaluator: &dyn Evaluator,
    genotypes: &[(u64, &CreatureGenotype)],
    eval_config: &EvaluationConfig,
    tournament_config: &super::tournament::TournamentConfig,
) -> super::tournament::TournamentResults {
    use super::tournament::Tournament;

    // Evaluate all creatures
    let results = evaluator.evaluate_batch(genotypes, eval_config);

    // Extract scores for the tournament criterion
    let criterion = &tournament_config.criterion;
    let scores: HashMap<u64, f32> = results
        .iter()
        .map(|(&id, result)| (id, result.score(criterion)))
        .collect();

    // Run the tournament
    Tournament::run_from_scores(tournament_config.clone(), &scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::genotype::morphology::{MorphologyNode, JointType};
    use bevy::prelude::Vec3;

    fn test_genotype() -> CreatureGenotype {
        CreatureGenotype::new(MorphologyNode::new(
            Vec3::new(1.0, 0.5, 0.5),
            JointType::Rigid,
        ))
    }

    #[test]
    fn test_null_evaluator() {
        let evaluator = NullEvaluator;
        let genotype = test_genotype();
        let config = EvaluationConfig::default();
        let result = evaluator.evaluate(&genotype, &config);
        assert_eq!(result.score(&CriterionId::LocomotionDistance), 0.0);
    }

    #[test]
    fn test_evaluation_result_default_score() {
        let result = EvaluationResult {
            scores: HashMap::new(),
            simulation_steps: 0,
            wall_time_ms: 0,
        };
        assert_eq!(result.score(&CriterionId::EnergyEfficiency), 0.0);
    }

    #[test]
    fn test_precomputed_evaluator() {
        let mut fitness = HashMap::new();
        fitness.insert(1, 10.0);
        fitness.insert(2, 5.0);

        let evaluator =
            PrecomputedEvaluator::from_fitness_map(&fitness, CriterionId::LocomotionDistance);
        let scores = evaluator.get_scores(1).unwrap();
        assert_eq!(
            *scores.get(&CriterionId::LocomotionDistance).unwrap(),
            10.0
        );
    }
}
