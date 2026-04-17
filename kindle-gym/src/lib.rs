//! Built-in environments for testing and development.
//!
//! Ported from OpenAI Gymnasium where noted.

#![allow(
    clippy::match_like_matches_macro,
    clippy::redundant_pattern_matching,
    clippy::needless_lifetimes,
    clippy::new_without_default,
    clippy::single_match,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::collapsible_match
)]
#![warn(
    trivial_numeric_casts,
    unused_extern_crates,
    clippy::pattern_type_mismatch
)]

pub mod acrobot;
pub mod cart_pole;
pub mod grid_world;
pub mod mountain_car;
pub mod pendulum;
pub mod random_walk;
pub mod taxi;
