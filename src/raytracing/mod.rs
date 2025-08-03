/// Reference CPU implementation for Boxtree-Ray intersetction
pub mod cpu;

/// Internal unittesting for the module
mod tests;

/// Real-time rendering for large voxel models with the help of GPU raytracing
#[cfg(feature = "bevy_wgpu")]
pub mod bevy;

/// Lightray definition with origin and direction
pub use crate::spatial::raytracing::Ray;

#[cfg(feature = "bevy_wgpu")]
pub use bevy::types::{
    BoxTreeGPUHost, BoxTreeGPUView, BoxTreeRenderData, BoxTreeSpyGlass, RenderBevyPlugin,
    VhxViewSet, Viewport,
};
