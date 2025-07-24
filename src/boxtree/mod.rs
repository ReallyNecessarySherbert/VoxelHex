mod detail;
pub(crate) mod iterate;
pub(crate) mod mipmap;
mod node;

/// The inner structure of the container
pub mod types;

/// Utilities for data update ibnside the voxel container
pub mod update;

#[cfg(test)]
mod tests;

pub use crate::spatial::math::vector::{V3c, V3cf32};
pub use types::{
    Albedo, BoxTree, BoxTreeEntry, MIPMapStrategy, MIPResamplingMethods, StrategyUpdater, VoxelData,
};

use crate::{
    boxtree::types::{BrickData, NodeChildren, NodeContent, OctreeError, PaletteIndexValues},
    object_pool::{ObjectPool, empty_marker},
    spatial::{
        Cube,
        math::{flat_projection, matrix_index_for},
    },
};
use std::{collections::HashMap, path::Path};

#[cfg(feature = "bytecode")]
use bendy::{decoding::FromBencode, encoding::ToBencode};

#[cfg(feature = "bytecode")]
use std::{
    fs::File,
    io::{Error, Read, Write},
};

//####################################################################################
//     ███████      █████████  ███████████ ███████████   ██████████ ██████████
//   ███░░░░░███   ███░░░░░███░█░░░███░░░█░░███░░░░░███ ░░███░░░░░█░░███░░░░░█
//  ███     ░░███ ███     ░░░ ░   ░███  ░  ░███    ░███  ░███  █ ░  ░███  █ ░
// ░███      ░███░███             ░███     ░██████████   ░██████    ░██████
// ░███      ░███░███             ░███     ░███░░░░░███  ░███░░█    ░███░░█
// ░░███     ███ ░░███     ███    ░███     ░███    ░███  ░███ ░   █ ░███ ░   █
//  ░░░███████░   ░░█████████     █████    █████   █████ ██████████ ██████████
//    ░░░░░░░      ░░░░░░░░░     ░░░░░    ░░░░░   ░░░░░ ░░░░░░░░░░ ░░░░░░░░░░
//  ██████████ ██████   █████ ███████████ ███████████   █████ █████
// ░░███░░░░░█░░██████ ░░███ ░█░░░███░░░█░░███░░░░░███ ░░███ ░░███
//  ░███  █ ░  ░███░███ ░███ ░   ░███  ░  ░███    ░███  ░░███ ███
//  ░██████    ░███░░███░███     ░███     ░██████████    ░░█████
//  ░███░░█    ░███ ░░██████     ░███     ░███░░░░░███    ░░███
//  ░███ ░   █ ░███  ░░█████     ░███     ░███    ░███     ░███
//  ██████████ █████  ░░█████    █████    █████   █████    █████
// ░░░░░░░░░░ ░░░░░    ░░░░░    ░░░░░    ░░░░░   ░░░░░    ░░░░░
//####################################################################################
impl<'a, T: VoxelData> From<(&'a Albedo, &'a T)> for BoxTreeEntry<'a, T> {
    fn from((albedo, data): (&'a Albedo, &'a T)) -> Self {
        BoxTreeEntry::Complex(albedo, data)
    }
}

/// Helper macro to create voxel data entries
#[macro_export]
macro_rules! voxel_data {
    ($data:expr) => {
        BoxTreeEntry::Informative($data)
    };
    () => {
        BoxTreeEntry::Empty
    };
}

impl<'a, T: VoxelData> From<&'a Albedo> for BoxTreeEntry<'a, T> {
    fn from(albedo: &'a Albedo) -> Self {
        BoxTreeEntry::Visual(albedo)
    }
}

impl<'a, T: VoxelData> BoxTreeEntry<'a, T> {
    pub fn albedo(&self) -> Option<&'a Albedo> {
        match self {
            BoxTreeEntry::Empty => None,
            BoxTreeEntry::Visual(albedo) => Some(albedo),
            BoxTreeEntry::Informative(_) => None,
            BoxTreeEntry::Complex(albedo, _) => Some(albedo),
        }
    }

    pub fn data(&self) -> Option<&'a T> {
        match self {
            BoxTreeEntry::Empty => None,
            BoxTreeEntry::Visual(_) => None,
            BoxTreeEntry::Informative(data) => Some(data),
            BoxTreeEntry::Complex(_, data) => Some(data),
        }
    }

    pub fn is_none(&self) -> bool {
        match self {
            BoxTreeEntry::Empty => true,
            BoxTreeEntry::Visual(albedo) => albedo.is_transparent(),
            BoxTreeEntry::Informative(data) => data.is_empty(),
            BoxTreeEntry::Complex(albedo, data) => albedo.is_transparent() && data.is_empty(),
        }
    }

    pub fn is_some(&self) -> bool {
        !self.is_none()
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
pub(crate) const OOB_SECTANT: u8 = 64;
pub(crate) const BOX_NODE_DIMENSION: usize = 4;
pub(crate) const BOX_NODE_CHILDREN_COUNT: usize = 64;

/// Creates a boxtree with the given parameters, also sets defaults for brick_dimension and user data type if not given!
#[macro_export]
macro_rules! make_tree {
    ($size:expr) => {
        Boxtree::<u32>::new($size, 32)
    };

    ($size:expr, $brick_dim:expr) => {
        Boxtree::<u32>::new($size, $brick_dim)
    };

    (<$type:ty>, $size:expr) => {
        Boxtree::<$type>::new($size, 32)
    };

    (<$type:ty>, $size:expr, $brick_dim:expr) => {
        Boxtree::<$type>::new($size, $brick_dim)
    };
}

impl<T: VoxelData> BoxTree<T> {
    /// converts the data structure to a byte representation
    #[cfg(feature = "bytecode")]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_bencode()
            .expect("Failed to serialize Octree to Bytes")
    }

    /// parses the data structure from a byte string
    #[cfg(feature = "bytecode")]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self::from_bencode(&bytes).expect("Failed to de-serialize Octree from bytes")
    }

    #[cfg(feature = "bytecode")]
    pub fn version<P: AsRef<Path>>(path: P) -> Result<crate::Version, Error> {
        let mut file = File::open(path)?;
        let mut bytes = vec![0; Self::bytes_until_version()];
        file.read_exact(&mut bytes)?;
        Ok(Self::parse_version(&bytes).expect("Expected to be able to parse Boxtree vrsion"))
    }

    /// saves the data structure to the given file path
    #[cfg(feature = "bytecode")]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let mut file = File::create(path)?;
        file.write_all(&self.to_bytes())?;
        Ok(())
    }

    /// loads the data structure from the given file path
    #[cfg(feature = "bytecode")]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(Self::from_bytes(bytes))
    }

    /// creates an boxtree with the given size
    /// * `brick_dimension` - must be one of `(2^x)` and smaller than the size of the boxtree
    /// * `size` - must be `brick_dimension * (4^x)`, e.g: brick_dimension == 2 --> size can be 8,32,128...
    pub fn new(size: u32, brick_dimension: u32) -> Result<Self, OctreeError> {
        if 0 == size || (brick_dimension as f32).log(2.0).fract() != 0.0 {
            return Err(OctreeError::InvalidBrickDimension(brick_dimension));
        }
        if brick_dimension > size
            || 0 == size
            || (size as f32 / brick_dimension as f32).log(4.0).fract() != 0.0
        {
            return Err(OctreeError::InvalidSize(size));
        }
        if size < (brick_dimension * BOX_NODE_DIMENSION as u32) {
            return Err(OctreeError::InvalidStructure(
                "Octree size must be larger, than BOX_NODE_DIMENSION * brick dimension".into(),
            ));
        }
        let node_count_estimation = (size / brick_dimension).pow(3);
        let mut nodes = ObjectPool::with_capacity(node_count_estimation.min(1024) as usize);
        let root_node_key = nodes.push(NodeContent::Nothing); // The first element is the root Node
        assert!(root_node_key == 0);
        Ok(Self {
            auto_simplify: true,
            boxtree_size: size,
            brick_dim: brick_dimension,
            nodes,
            node_children: vec![NodeChildren::default()],
            node_mips: vec![BrickData::Empty],
            voxel_color_palette: vec![],
            voxel_data_palette: vec![],
            map_to_color_index_in_palette: HashMap::new(),
            map_to_data_index_in_palette: HashMap::new(),
            mip_map_strategy: MIPMapStrategy::default(),
        })
    }

    /// Getter function for the boxtree
    /// * Returns immutable reference to the data at the given position, if there is any
    pub fn get(&self, position: &V3c<u32>) -> BoxTreeEntry<'_, T> {
        NodeContent::pix_get_ref(
            &self.get_internal(
                Self::ROOT_NODE_KEY as usize,
                Cube::root_bounds(self.boxtree_size as f32),
                position,
            ),
            &self.voxel_color_palette,
            &self.voxel_data_palette,
        )
    }

    /// Tells the radius of the area covered by the boxtree
    pub fn get_size(&self) -> u32 {
        self.boxtree_size
    }

    /// Object to set the MIP map strategy for each MIP level inside the boxtree
    pub fn albedo_mip_map_resampling_strategy(&mut self) -> StrategyUpdater<'_, T> {
        StrategyUpdater(self)
    }

    /// Internal Getter function for the boxtree, to be able to call get from within the tree itself
    /// * Returns immutable reference to the data of the given node at the given position, if there is any
    fn get_internal(
        &self,
        current_node_key: usize,
        mut current_bounds: Cube,
        position: &V3c<u32>,
    ) -> PaletteIndexValues {
        let position_ = V3c::from(*position);
        if !current_bounds.contains(&position_) {
            return empty_marker();
        }

        let Some(current_node_key) =
            self.get_node_internal(current_node_key, &mut current_bounds, &position_)
        else {
            return empty_marker();
        };

        match self.nodes.get(current_node_key) {
            NodeContent::Nothing => empty_marker(),
            NodeContent::Leaf(bricks) => {
                // In case brick_dimension == boxtree size, the root node can not be a leaf...
                debug_assert!(self.brick_dim < self.boxtree_size);

                // Hash the position to the target child
                let child_sectant_at_position = current_bounds.sectant_for(&position_);

                // If the child exists, query it for the voxel
                match &bricks[child_sectant_at_position as usize] {
                    BrickData::Empty => empty_marker(),
                    BrickData::Parted(brick) => {
                        current_bounds =
                            Cube::child_bounds_for(&current_bounds, child_sectant_at_position);
                        let mat_index = matrix_index_for(&current_bounds, position, self.brick_dim);
                        let mat_index = flat_projection(
                            mat_index.x as usize,
                            mat_index.y as usize,
                            mat_index.z as usize,
                            self.brick_dim as usize,
                        );
                        if !NodeContent::pix_points_to_empty(
                            &brick[mat_index],
                            &self.voxel_color_palette,
                            &self.voxel_data_palette,
                        ) {
                            return brick[mat_index];
                        }
                        empty_marker()
                    }
                    BrickData::Solid(voxel) => *voxel,
                }
            }
            NodeContent::UniformLeaf(brick) => match brick {
                BrickData::Empty => empty_marker(),
                BrickData::Parted(brick) => {
                    let mat_index = matrix_index_for(&current_bounds, position, self.brick_dim);
                    let mat_index = flat_projection(
                        mat_index.x as usize,
                        mat_index.y as usize,
                        mat_index.z as usize,
                        self.brick_dim as usize,
                    );
                    brick[mat_index]
                }
                BrickData::Solid(voxel) => *voxel,
            },
            NodeContent::Internal(_occupied_bits) => {
                // Deepest child at given position is empty
                empty_marker()
            }
        }
    }
}
