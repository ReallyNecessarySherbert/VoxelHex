use crate::{
    boxtree::{
        types::NodeContent, Albedo, BoxTree, MIPResamplingMethods, VoxelData, BOX_NODE_DIMENSION,
    },
    spatial::{math::vector::V3c, Cube},
};
use std::collections::HashMap;

pub(crate) trait MIPResamplingFunction {
    /// Provides a color value from the given range acquired from the sampling function
    /// * `sample_start` - The start position of the range to sample from
    /// * `sample_size` - The size of the range to sample from
    /// * `sample_fn` - The function providing the samples. It will be called on each position given by the range
    fn execute<F: Fn(&V3c<u32>) -> Option<Albedo>>(
        self,
        sample_start: &V3c<u32>,
        sample_size: u32,
        sample_fn: F,
    ) -> Option<Albedo>;
}

/// Calls the given function for every child position inside the given update range
/// The function is called at least once for the relevant sectants
/// Target bounds are calculataed based on node bounds, independently from cell_size
/// * `node_bounds` - The bounds of the updated node
/// * `position` - The position of the intended update
/// * `update_size` - Range of the intended update starting from position
/// * `cell_size` - The size of one child inside the updated node
/// * `fun` - The function to execute: |position_in_target(global), update_size_in_target, target_child_sectant, &target_bounds| { ... }
///
/// returns with update size
pub(crate) fn execute_for_relevant_sectants<F: FnMut(V3c<u32>, V3c<u32>, u8, &Cube)>(
    node_bounds: &Cube,
    position_: &V3c<u32>,
    update_size: u32,
    mut fun: F,
) -> V3c<usize> {
    if (position_.x as f32 > node_bounds.min_position.x + node_bounds.size)
        || (position_.y as f32 > node_bounds.min_position.y + node_bounds.size)
        || (position_.z as f32 > node_bounds.min_position.z + node_bounds.size)
    {
        return V3c::unit(0); // Nothing to do when update will never reach node bounds..
    }

    //In case position is smaller, start from node, but do not overstep because of it!
    // --> trim update_size in case position is smaller
    let position = V3c::new(
        (position_.x as f32).max(node_bounds.min_position.x),
        (position_.y as f32).max(node_bounds.min_position.y),
        (position_.z as f32).max(node_bounds.min_position.z),
    );
    let update_size = V3c::from(*position_) + V3c::unit(update_size as f32) - position;
    let cell_size = node_bounds.size / BOX_NODE_DIMENSION as f32;

    let mut shifted_position = position;
    while shifted_position.x <= (position.x + update_size.x) {
        shifted_position.y = position.y;
        while shifted_position.y <= (position.y + update_size.y) {
            shifted_position.z = position.z;
            while shifted_position.z <= (position.z + update_size.z) {
                if !node_bounds.contains(&shifted_position) {
                    shifted_position.z += cell_size;
                    continue;
                }

                let target_child_sectant = node_bounds.sectant_for(&shifted_position);
                let target_bounds = node_bounds.child_bounds_for(target_child_sectant);

                // In case smaller brick dimensions, it might happen that one update affects multiple sectants
                // e.g. when a uniform leaf has a parted brick of 2x2x2 --> Setting a value in one element
                // affects multiple sectants. In these cases, the target size is 0.5, and positions
                // also move inbetween voxels. Logically this is needed for e.g. setting the correct occupied bits
                // for a given node. The worst case scenario is some cells are given a value multiple times,
                // which is acceptable to reduce complexity
                let target_bounds = Cube {
                    min_position: target_bounds.min_position.floor(),
                    size: target_bounds.size.ceil(),
                };

                let position_in_target = V3c::new(
                    position.x.max(target_bounds.min_position.x),
                    position.y.max(target_bounds.min_position.y),
                    position.z.max(target_bounds.min_position.z),
                );
                let update_size_remains = position + update_size - position_in_target;
                let update_size_in_target = (target_bounds.min_position
                    + V3c::unit(target_bounds.size)
                    - position_in_target)
                    .cut_by(update_size_remains);

                if 0. < update_size_in_target.x
                    && 0. < update_size_in_target.y
                    && 0. < update_size_in_target.z
                {
                    fun(
                        position_in_target.into(),
                        update_size_in_target.into(),
                        target_child_sectant,
                        &target_bounds,
                    );
                }

                shifted_position.z += cell_size;
            }

            shifted_position.y += cell_size;
        }

        shifted_position.x += cell_size;
    }

    V3c::from(update_size)
}

impl<T: VoxelData> BoxTree<T>
{
    pub(crate) fn get_node_internal(
        &self,
        mut current_node_key: usize,
        node_bounds: &mut Cube,
        position: &V3c<f32>,
    ) -> Option<usize> {
        debug_assert!(
            node_bounds.contains(position),
            "Expected position to be inside given bounds"
        );

        loop {
            match self.nodes.get(current_node_key) {
                NodeContent::Nothing | NodeContent::Leaf(_) | NodeContent::UniformLeaf(_) => {
                    return Some(current_node_key);
                }
                NodeContent::Internal(occupied_bits) => {
                    // Hash the position to the target child
                    let child_sectant_at_position = node_bounds.sectant_for(position);
                    let child_at_position =
                        self.node_children[current_node_key].child(child_sectant_at_position);

                    // There is a valid child at the given position inside the node, recurse into it
                    if self.nodes.key_is_valid(child_at_position) {
                        debug_assert_ne!(
                            0,
                            occupied_bits & (0x01 << child_sectant_at_position),
                            "Node[{:?}] under {:?} \n has a child(node[{:?}]) in sectant[{:?}](global position: {:?}), which is incompatible with the occupancy bitmap: {:#10X}; \n child node: {:?}; child node children: {:?};",
                            current_node_key,
                            node_bounds,
                            self.node_children[current_node_key].child(child_sectant_at_position),
                            child_sectant_at_position,
                            position, occupied_bits,
                            self.nodes.get(self.node_children[current_node_key].child(child_sectant_at_position)),
                            self.node_children[self.node_children[current_node_key].child(child_sectant_at_position)]
                        );
                        current_node_key = child_at_position;
                        *node_bounds =
                            Cube::child_bounds_for(node_bounds, child_sectant_at_position);
                    } else {
                        return Some(current_node_key);
                    }
                }
            }
        }
    }
}

/// Container to store intermediate values in a higher capacity type ( u8 overflows a lot )
/// do do do do doo do do do do du doo
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
struct Albedou32 {
    r: u32,
    g: u32,
    b: u32,
    a: u32,
}

impl Albedou32 {
    fn length(&self) -> f32 {
        ((self.r.pow(2) + self.g.pow(2) + self.b.pow(2) + self.a.pow(2)) as f32).sqrt()
    }
    fn sqrt(mut self) -> Self {
        self.r = (self.r as f32).sqrt().round() as u32;
        self.g = (self.g as f32).sqrt().round() as u32;
        self.b = (self.b as f32).sqrt().round() as u32;
        self.a = (self.a as f32).sqrt().round() as u32;
        self
    }
    fn pow2(mut self) -> Self {
        self.r = self.r.pow(2);
        self.g = self.g.pow(2);
        self.b = self.b.pow(2);
        self.a = self.a.pow(2);
        self
    }
}

impl std::ops::Sub for Albedou32 {
    type Output = Albedou32;
    fn sub(self, other: Albedou32) -> Albedou32 {
        Albedou32 {
            r: self.r - other.r,
            g: self.g - other.g,
            b: self.b - other.b,
            a: self.a - other.a,
        }
    }
}

impl std::ops::Add for Albedou32 {
    type Output = Albedou32;
    fn add(self, other: Albedou32) -> Albedou32 {
        Albedou32 {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
            a: self.a + other.a,
        }
    }
}

impl std::ops::Div<u32> for Albedou32 {
    type Output = Albedou32;
    fn div(self, divisor: u32) -> Albedou32 {
        Albedou32 {
            r: (self.r as f32 / divisor as f32).round() as u32,
            g: (self.g as f32 / divisor as f32).round() as u32,
            b: (self.b as f32 / divisor as f32).round() as u32,
            a: (self.a as f32 / divisor as f32).round() as u32,
        }
    }
}

impl From<Albedo> for Albedou32 {
    fn from(other: Albedo) -> Self {
        Albedou32 {
            r: other.r as u32,
            g: other.g as u32,
            b: other.b as u32,
            a: other.a as u32,
        }
    }
}

impl From<Albedou32> for Albedo {
    fn from(other: Albedou32) -> Self {
        Albedo {
            r: (other.r).min(255) as u8,
            g: (other.g).min(255) as u8,
            b: (other.b).min(255) as u8,
            a: (other.a).min(255) as u8,
        }
    }
}

impl MIPResamplingFunction for MIPResamplingMethods {
    fn execute<F: Fn(&V3c<u32>) -> Option<Albedo>>(
        self,
        sample_start: &V3c<u32>,
        sample_size: u32,
        sample_fn: F,
    ) -> Option<Albedo> {
        match self {
            MIPResamplingMethods::BoxFilter => {
                // Calculate gamma corrected average albedo in the sampling range
                let mut avg_albedo = None;
                let mut entry_count = 0;
                for x in sample_start.x..(sample_start.x + sample_size) {
                    for y in sample_start.y..(sample_start.y + sample_size) {
                        for z in sample_start.z..(sample_start.z + sample_size) {
                            match (&mut avg_albedo, sample_fn(&V3c::new(x, y, z))) {
                                (None, Some(new_albedo)) => {
                                    debug_assert_eq!(0, entry_count);
                                    entry_count = 1;
                                    avg_albedo = Some((
                                        (new_albedo.r as f32).powf(2.),
                                        (new_albedo.g as f32).powf(2.),
                                        (new_albedo.b as f32).powf(2.),
                                        (new_albedo.a as f32).powf(2.),
                                    ));
                                }
                                (Some(current_avg_albedo), Some(new_albedo)) => {
                                    entry_count += 1;
                                    current_avg_albedo.0 += (new_albedo.r as f32).powf(2.);
                                    current_avg_albedo.1 += (new_albedo.g as f32).powf(2.);
                                    current_avg_albedo.2 += (new_albedo.b as f32).powf(2.);
                                    current_avg_albedo.3 += (new_albedo.a as f32).powf(2.);
                                }
                                (None, None) | (Some(_), None) => {}
                            }
                        }
                    }
                }

                if let Some(albedo) = avg_albedo {
                    debug_assert_ne!(0, entry_count, "Expected to have non-zero entries in MIP");
                    let r = (albedo.0 / entry_count as f32).sqrt().min(255.) as u8;
                    let g = (albedo.1 / entry_count as f32).sqrt().min(255.) as u8;
                    let b = (albedo.2 / entry_count as f32).sqrt().min(255.) as u8;
                    let a = (albedo.3 / entry_count as f32).sqrt().min(255.) as u8;
                    Some(Albedo { r, g, b, a })
                } else {
                    None
                }
            }
            MIPResamplingMethods::PointFilter | MIPResamplingMethods::PointFilterBD => {
                // Collect Albedo occurences in the sampling range
                let mut albedo_counts = HashMap::new();
                for x in sample_start.x..(sample_start.x + sample_size) {
                    for y in sample_start.y..(sample_start.y + sample_size) {
                        for z in sample_start.z..(sample_start.z + sample_size) {
                            if let Some(color) = sample_fn(&V3c::new(x, y, z)) {
                                albedo_counts
                                    .entry(color)
                                    .and_modify(|e| *e += 1)
                                    .or_insert(1);
                            }
                        }
                    }
                }

                // return with the most frequent albedo
                albedo_counts
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .unzip()
                    .0
            }
            MIPResamplingMethods::Posterize(thr) | MIPResamplingMethods::PosterizeBD(thr) => {
                // Collect Albedo occurences in the sampling range
                // the map collects squared albedo sums, along with occurence counts
                // to build the function: sqrt((x1^2 + .... xn^2)/n)
                let mut albedo_counts = HashMap::<Albedou32, u32>::new();
                for x in sample_start.x..(sample_start.x + sample_size) {
                    for y in sample_start.y..(sample_start.y + sample_size) {
                        for z in sample_start.z..(sample_start.z + sample_size) {
                            if let Some(color) = sample_fn(&V3c::new(x, y, z)) {
                                let mut old_albedo_sum = None;
                                let mut new_albedo_sum = Albedou32 {
                                    r: 0,
                                    g: 0,
                                    b: 0,
                                    a: 0,
                                };

                                for (albedo_sum, albedo_count) in albedo_counts.iter() {
                                    // Convert stored albedo back from gamma space
                                    let poster_color = (albedo_sum.clone() / *albedo_count).sqrt();
                                    if (poster_color - Albedou32::from(color)).length()
                                        < (thr * 255.)
                                    {
                                        old_albedo_sum = Some(albedo_sum.clone());
                                        new_albedo_sum =
                                            albedo_sum.clone() + Albedou32::from(color).pow2();
                                        break;
                                    }
                                }

                                if let Some(old_albedo_sum) = old_albedo_sum {
                                    let new_albedo_count = albedo_counts
                                        .remove(&old_albedo_sum)
                                        .expect(
                                        "Expected albdeo value to be previously present in HashSet",
                                    ) + 1;
                                    albedo_counts.insert(new_albedo_sum, new_albedo_count);
                                } else {
                                    albedo_counts.insert(Albedou32::from(color).pow2(), 1);
                                }
                            }
                        }
                    }
                }

                // return with the most frequent albedo
                albedo_counts
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(powered_color, color_count)| (powered_color / color_count).sqrt().into())
            }
        }
    }
}
