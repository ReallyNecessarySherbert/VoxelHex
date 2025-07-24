/// As in: Look-up Tables
pub mod lut;

pub mod math;

#[cfg(feature = "raytracing")]
pub mod raytracing;

mod tests;

use crate::{
    boxtree::BOX_NODE_DIMENSION, spatial::lut::SECTANT_OFFSET_LUT, spatial::math::offset_sectant,
    spatial::math::vector::V3c,
};

#[derive(Default, Clone, Copy, Debug)]
pub(crate) struct Cube {
    pub(crate) min_position: V3c<f32>,
    pub(crate) size: f32,
}

impl Cube {
    /// Creates boundaries starting from (0,0,0), with the given size
    pub(crate) const fn root_bounds(size: f32) -> Self {
        Self {
            min_position: V3c::unit(0.),
            size,
        }
    }

    /// True if the given position is within the boundaries of the object
    pub(crate) const fn contains(&self, position: &V3c<f32>) -> bool {
        position.x >= self.min_position.x
            && position.y >= self.min_position.y
            && position.z >= self.min_position.z
            && position.x < (self.min_position.x + self.size)
            && position.y < (self.min_position.y + self.size)
            && position.z < (self.min_position.z + self.size)
    }

    pub(crate) fn sectant_for(&self, position: &V3c<f32>) -> u8 {
        debug_assert!(
            self.contains(position),
            "Position {:?}, out of {:?}",
            position,
            self
        );
        offset_sectant(&(*position - self.min_position), self.size)
    }

    /// Creates a bounding box within an area described by the min_position and size, for the given sectant
    pub(crate) fn child_bounds_for(&self, sectant: u8) -> Cube {
        Cube {
            min_position: (self.min_position + (SECTANT_OFFSET_LUT[sectant as usize] * self.size)),
            size: self.size / BOX_NODE_DIMENSION as f32,
        }
    }
}
