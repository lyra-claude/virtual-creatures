//! Arena tournament system for creature Elo ratings.
//!
//! Substrate-agnostic: works with any evaluation backend (Bevy/Rapier sim,
//! pre-recorded scores, or hypothetical matchups). The tournament system
//! handles scheduling, Elo computation, and Balduzzi et al.'s transitive-cyclic
//! decomposition for detecting intransitive dominance.

pub mod criteria;
pub mod cycle_morphology;
pub mod decomposition;
pub mod ratings;
pub mod sweep;
pub mod tournament;

pub use criteria::*;
pub use cycle_morphology::*;
pub use decomposition::*;
pub use ratings::*;
pub use sweep::*;
pub use tournament::*;
