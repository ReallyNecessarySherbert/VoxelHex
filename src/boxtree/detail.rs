use crate::{
    boxtree::{
        types::{
            Albedo, BoxTree, NodeChildren, NodeContent, NodeData, PaletteIndexValues, VoxelData,
        },
        BrickData, V3c, BOX_NODE_CHILDREN_COUNT, BOX_NODE_DIMENSION,
    },
    object_pool::empty_marker,
    spatial::{lut::SECTANT_OFFSET_LUT, math::flat_projection},
};
use bendy::{decoding::FromBencode, encoding::ToBencode};
use num_traits::Zero;
use std::{
    hash::Hash,
    ops::{Add, Div},
};

impl<T: Zero + PartialEq> VoxelData for T {
    fn is_empty(&self) -> bool {
        *self == T::zero()
    }
}

//####################################################################################
//     █████████   █████       ███████████  ██████████ ██████████      ███████
//   ███░░░░░███ ░░███       ░░███░░░░░███░░███░░░░░█░░███░░░░███   ███░░░░░███
//  ░███    ░███  ░███        ░███    ░███ ░███  █ ░  ░███   ░░███ ███     ░░███
//  ░███████████  ░███        ░██████████  ░██████    ░███    ░███░███      ░███
//  ░███░░░░░███  ░███        ░███░░░░░███ ░███░░█    ░███    ░███░███      ░███
//  ░███    ░███  ░███      █ ░███    ░███ ░███ ░   █ ░███    ███ ░░███     ███
//  █████   █████ ███████████ ███████████  ██████████ ██████████   ░░░███████░
// ░░░░░   ░░░░░ ░░░░░░░░░░░ ░░░░░░░░░░░  ░░░░░░░░░░ ░░░░░░░░░░      ░░░░░░░
//####################################################################################

impl Albedo {
    pub fn with_red(mut self, r: u8) -> Self {
        self.r = r;
        self
    }

    pub fn with_green(mut self, g: u8) -> Self {
        self.g = g;
        self
    }

    pub fn with_blue(mut self, b: u8) -> Self {
        self.b = b;
        self
    }

    pub fn with_alpha(mut self, a: u8) -> Self {
        self.a = a;
        self
    }

    pub fn is_transparent(&self) -> bool {
        self.a == 0
    }

    pub fn distance_from(&self, other: &Albedo) -> f32 {
        let distance_r = self.r as f32 - other.r as f32;
        let distance_g = self.g as f32 - other.g as f32;
        let distance_b = self.b as f32 - other.b as f32;
        let distance_a = self.a as f32 - other.a as f32;
        (distance_r.powf(2.) + distance_g.powf(2.) + distance_b.powf(2.) + distance_a.powf(2.))
            .sqrt()
    }
}

impl From<u32> for Albedo {
    fn from(value: u32) -> Self {
        let a = (value & 0x000000FF) as u8;
        let b = ((value & 0x0000FF00) >> 8) as u8;
        let g = ((value & 0x00FF0000) >> 16) as u8;
        let r = ((value & 0xFF000000) >> 24) as u8;

        Albedo::default()
            .with_red(r)
            .with_green(g)
            .with_blue(b)
            .with_alpha(a)
    }
}

impl Add for Albedo {
    type Output = Albedo;
    fn add(self, other: Albedo) -> Albedo {
        Albedo {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
            a: self.a + other.a,
        }
    }
}

impl Div<f32> for Albedo {
    type Output = Albedo;
    fn div(self, divisor: f32) -> Albedo {
        Albedo {
            r: (self.r as f32 / divisor).round() as u8,
            g: (self.g as f32 / divisor).round() as u8,
            b: (self.b as f32 / divisor).round() as u8,
            a: (self.a as f32 / divisor).round() as u8,
        }
    }
}

impl Zero for Albedo {
    fn zero() -> Self {
        Self {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        }
    }
    fn is_zero(&self) -> bool {
        self.is_empty()
    }
}

//####################################################################################
//  ███████████     ███████    █████ █████ ███████████ ███████████   ██████████ ██████████
// ░░███░░░░░███  ███░░░░░███ ░░███ ░░███ ░█░░░███░░░█░░███░░░░░███ ░░███░░░░░█░░███░░░░░█
//  ░███    ░███ ███     ░░███ ░░███ ███  ░   ░███  ░  ░███    ░███  ░███  █ ░  ░███  █ ░
//  ░██████████ ░███      ░███  ░░█████       ░███     ░██████████   ░██████    ░██████
//  ░███░░░░░███░███      ░███   ███░███      ░███     ░███░░░░░███  ░███░░█    ░███░░█
//  ░███    ░███░░███     ███   ███ ░░███     ░███     ░███    ░███  ░███ ░   █ ░███ ░   █
//  ███████████  ░░░███████░   █████ █████    █████    █████   █████ ██████████ ██████████
// ░░░░░░░░░░░     ░░░░░░░    ░░░░░ ░░░░░    ░░░░░    ░░░░░   ░░░░░ ░░░░░░░░░░ ░░░░░░░░░░
//####################################################################################
impl<T> BoxTree<T>
where
    T: Default + Clone + Eq + Hash + VoxelData,
{
    /// The root node is always the first item
    pub(crate) const ROOT_NODE_KEY: u32 = 0;
}

impl<
        #[cfg(all(feature = "bytecode", feature = "serialization"))] T: FromBencode
            + ToBencode
            + Serialize
            + DeserializeOwned
            + Default
            + Eq
            + Clone
            + Hash
            + VoxelData,
        #[cfg(all(feature = "bytecode", not(feature = "serialization")))] T: FromBencode + ToBencode + Default + Eq + Clone + Hash + VoxelData,
        #[cfg(all(not(feature = "bytecode"), feature = "serialization"))] T: Serialize + DeserializeOwned + Default + Eq + Clone + Hash + VoxelData,
        #[cfg(all(not(feature = "bytecode"), not(feature = "serialization")))] T: Default + Eq + Clone + Hash + VoxelData,
    > BoxTree<T>
{
    /// Provides the child key if there is a valid child under the given sectant
    #[cfg(feature = "bevy_wgpu")]
    pub(crate) fn valid_child_for(&self, node_key: usize, sectant: u8) -> Option<usize> {
        let child_key = self.nodes.get(node_key).child(sectant);
        if self.nodes.key_is_valid(child_key) {
            Some(child_key)
        } else {
            None
        }
    }

    /// Returns with true if Node is empty at the given target sectant
    pub(crate) fn node_empty_at(&self, node_key: usize, target_sectant: u8) -> bool {
        match &self.nodes.get(node_key).content {
            NodeContent::Nothing => true,
            NodeContent::Leaf(bricks) => match &bricks[target_sectant as usize] {
                BrickData::Empty => true,
                BrickData::Solid(voxel) => NodeContent::pix_points_to_empty(
                    voxel,
                    &self.voxel_color_palette,
                    &self.voxel_data_palette,
                ),
                BrickData::Parted(_brick) => {
                    if let Some(data) = bricks[target_sectant as usize].get_homogeneous_data() {
                        NodeContent::pix_points_to_empty(
                            data,
                            &self.voxel_color_palette,
                            &self.voxel_data_palette,
                        )
                    } else {
                        false
                    }
                }
            },
            NodeContent::UniformLeaf(brick) => match brick {
                BrickData::Empty => true,
                BrickData::Solid(voxel) => NodeContent::pix_points_to_empty(
                    voxel,
                    &self.voxel_color_palette,
                    &self.voxel_data_palette,
                ),
                BrickData::Parted(brick) => {
                    let check_start = V3c::from(
                        (SECTANT_OFFSET_LUT[target_sectant as usize] * self.brick_dim as f32)
                            .floor(),
                    );
                    let check_size =
                        (self.brick_dim as f32 / BOX_NODE_DIMENSION as f32).max(1.) as usize;
                    for x in check_start.x..(check_start.x + check_size) {
                        for y in check_start.y..(check_start.y + check_size) {
                            for z in check_start.z..(check_start.z + check_size) {
                                if !NodeContent::pix_points_to_empty(
                                    &brick[flat_projection(x, y, z, self.brick_dim as usize)],
                                    &self.voxel_color_palette,
                                    &self.voxel_data_palette,
                                ) {
                                    return false;
                                }
                            }
                        }
                    }
                    true
                }
            },
            NodeContent::Internal => {
                let child_key = self.nodes.get(node_key).child(target_sectant);

                if !self.nodes.key_is_valid(child_key) {
                    return true; // Invalid child counts as empty
                }

                // Examine the child at every sectant
                for child_sectant in 0..BOX_NODE_CHILDREN_COUNT {
                    if !self.node_empty_at(child_key, child_sectant as u8) {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Compares the contents of the given node keys to see if they match
    /// Invalid keys count as empty content
    /// Returns with true if the 2 keys have equivalaent values
    pub(crate) fn compare_nodes(&self, node_key_left: usize, node_key_right: usize) -> bool {
        if self.nodes.key_is_valid(node_key_left) != self.nodes.key_is_valid(node_key_right) {
            return false;
        }

        if self.nodes.key_is_valid(node_key_left) {
            // both keys are valid, compare their contents
            return self
                .nodes
                .get(node_key_left)
                .content
                .compare(&self.nodes.get(node_key_right).content);
        }
        true
    }

    /// Subdivides the node into multiple nodes. It guarantees that there will be a child at the target sectant
    /// * `node_key` - The key of the node to subdivide. It must be a leaf
    /// * `target_sectant` - The sectant that must have a child
    pub(crate) fn subdivide_leaf_to_nodes(&mut self, node_key: usize, target_sectant: usize) {
        let mut node_content = NodeContent::Internal;
        std::mem::swap(&mut node_content, &mut self.nodes.get_mut(node_key).content);
        let mut node_new_children = [empty_marker(); BOX_NODE_CHILDREN_COUNT];
        match node_content {
            NodeContent::Nothing | NodeContent::Internal => {
                panic!("Non-leaf node expected to be Leaf")
            }
            NodeContent::Leaf(mut bricks) => {
                // All contained bricks shall be converted to leaf nodes
                for sectant in 0..BOX_NODE_CHILDREN_COUNT {
                    let mut brick = BrickData::Empty;
                    std::mem::swap(&mut brick, &mut bricks[sectant]);

                    if !brick.contains_nothing(&self.voxel_color_palette, &self.voxel_data_palette)
                        || sectant == target_sectant
                    // Push in a new child even if the brick is empty for the target sectant
                    {
                        // Push in the new(placeholder) child
                        node_new_children[sectant] = self.nodes.push(NodeData::empty_node()) as u32;
                    }

                    match brick {
                        BrickData::Empty => {}
                        BrickData::Solid(voxel) => {
                            self.nodes
                                .get_mut(node_new_children[sectant] as usize)
                                .occupied_bits = u64::MAX;
                            self.nodes
                                .get_mut(node_new_children[sectant] as usize)
                                .content = NodeContent::UniformLeaf(BrickData::Solid(voxel));
                        }
                        BrickData::Parted(brick) => {
                            // Calculcate the occupancy bitmap for the new leaf child node
                            // As it is a higher resolution, than the current bitmap, it needs to be bruteforced
                            self.nodes
                                .get_mut(node_new_children[sectant] as usize)
                                .occupied_bits = bricks[sectant].calculate_occupied_bits(
                                self.brick_dim as usize,
                                &self.voxel_color_palette,
                                &self.voxel_data_palette,
                            );
                            self.nodes
                                .get_mut(node_new_children[sectant] as usize)
                                .content =
                                NodeContent::UniformLeaf(BrickData::Parted(brick.clone()));
                        }
                    }
                }
            }
            NodeContent::UniformLeaf(brick) => {
                // The leaf will be divided into 64 bricks, and the contents will be mapped from the current brick
                match brick {
                    BrickData::Empty => {
                        // Push in an empty leaf child to the target sectant ( that will be populated later )
                        // But nothing else to do, as the Uniform leaf is empty!
                        node_new_children[target_sectant] =
                            self.nodes.push(NodeData::empty_node()) as u32;
                    }
                    BrickData::Solid(voxel) => {
                        // Push in all solid children for child sectants
                        for new_child in node_new_children.iter_mut().take(BOX_NODE_CHILDREN_COUNT)
                        {
                            *new_child =
                                self.nodes.push(NodeData::uniform_solid_node(voxel)) as u32;
                        }
                    }
                    BrickData::Parted(brick) => {
                        // Each brick is mapped to take up one subsection of the current data
                        let children_bricks = Self::dilute_brick_data(brick, self.brick_dim);
                        for (sectant, new_brick) in children_bricks.into_iter().enumerate() {
                            // Push in the new child
                            let child_occupied_bits = BrickData::calculate_brick_occupied_bits(
                                &new_brick,
                                self.brick_dim as usize,
                                &self.voxel_color_palette,
                                &self.voxel_data_palette,
                            );
                            node_new_children[sectant] =
                                self.nodes.push(NodeData::uniform_parted_node(
                                    BrickData::Parted(new_brick),
                                    child_occupied_bits,
                                )) as u32;
                        }
                    }
                }
            }
        }
        self.nodes.get_mut(node_key).children = NodeChildren::Children(node_new_children);
    }

    /// Tries to create a brick from the given node if possible. WARNING: Data loss may occur
    pub(crate) fn try_brick_from_node(&self, node_key: usize) -> BrickData<PaletteIndexValues> {
        if !self.nodes.key_is_valid(node_key) {
            return BrickData::Empty;
        }
        match &self.nodes.get(node_key).content {
            NodeContent::Nothing | NodeContent::Internal | NodeContent::Leaf(_) => BrickData::Empty,

            NodeContent::UniformLeaf(brick) => brick.clone(),
        }
    }

    /// Erase all children of the node under the given key, and set its children to "No children"
    pub(crate) fn deallocate_children_of(&mut self, node_key: usize) {
        if !self.nodes.key_is_valid(node_key) {
            return;
        }
        let mut to_deallocate = Vec::new();
        if let Some(children) = self.nodes.get(node_key).children_iter() {
            for child in children {
                if self.nodes.key_is_valid(*child as usize) {
                    to_deallocate.push(*child as usize);
                }
            }
            for child_key in to_deallocate {
                debug_assert_ne!(child_key, node_key, "Node referring to itself as child");

                self.deallocate_children_of(child_key); // Recursion should be fine as depth is not expceted to be more, than 32
                self.nodes.free(child_key);
            }
        }
    }
}
