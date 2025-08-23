/// As in: Look-up Tables
pub mod lut;

pub mod math;

#[cfg(feature = "raytracing")]
pub mod raytracing;

mod tests;

use crate::{
    boxtree::BOX_NODE_DIMENSION,
    spatial::{
        lut::{SECTANT_OFFSET_LUT, SECTANT_STEP_RESULT_LUT},
        math::{offset_sectant, vector::V3c},
    },
};

/// Provides the resulting sectant based on the given sectant
/// Returns with a value larger than `BOX_NODE_CHILDREN_COUNT - 1` when out of bounds.
/// Important note: the specs of `signum` behvaes differently for f32 and i32
/// So the conversion to i32 is absolutely required
pub(crate) const fn step_sectant(sectant: u8, step: V3c<f32>) -> u8 {
    SECTANT_STEP_RESULT_LUT[sectant as usize][((step.x as i32).signum() + 1) as usize]
        [((step.y as i32).signum() + 1) as usize][((step.z as i32).signum() + 1) as usize]
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum CubeSides {
    Back = 0,
    Front = 1,
    Top = 2,
    Bottom = 3,
    Left = 4,
    Right = 5,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
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
            "Position {position:?}, out of {self:?}"
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
