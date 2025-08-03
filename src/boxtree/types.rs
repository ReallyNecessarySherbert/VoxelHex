use crate::{boxtree::BOX_NODE_CHILDREN_COUNT, object_pool::ObjectPool};
use std::{collections::HashMap, error::Error, hash::Hash, sync::Arc};

#[cfg(feature = "bytecode")]
use bendy::{decoding::FromBencode, encoding::ToBencode};

/// error types during usage or creation of the boxtree
#[derive(Debug)]
pub enum OctreeError {
    /// Octree creation was attempted with an invalid boxtree size
    InvalidSize(u32),

    /// Octree creation was attempted with an invalid brick dimension
    InvalidBrickDimension(u32),

    /// Octree creation was attempted with an invalid structure parameter ( refer to error )
    InvalidStructure(Box<dyn Error>),

    /// Octree query was attempted with an invalid position
    InvalidPosition { x: u32, y: u32, z: u32 },
}

/// An entry for stored voxel data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoxTreeEntry<'a, T: VoxelData> {
    /// No information available in boxtree query
    Empty,

    /// Albedo data is available in boxtree query
    Visual(&'a Albedo),

    /// User data is avaliable in boxtree query
    Informative(&'a T),

    /// Both user data and color information is available in boxtree query
    Complex(&'a Albedo, &'a T),
}

/// Data representation for a matrix of voxels
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) enum BrickData<T>
where
    T: Clone + PartialEq + Clone,
{
    /// Brick is empty
    #[default]
    Empty,

    /// Brick is an NxNxN matrix, size is determined by the parent entity
    Parted(Vec<T>),

    /// Brick is a single item T, which takes up the entirety of the brick
    Solid(T),
}

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) enum NodeContent<T>
where
    T: Clone + PartialEq + Clone,
{
    /// Node is empty
    #[default]
    Nothing,

    /// Internal node
    Internal,

    /// Node contains 8 children, each with their own brickdata
    Leaf([BrickData<T>; BOX_NODE_CHILDREN_COUNT]),

    /// Node has one child, which takes up the entirety of the node with its brick data
    UniformLeaf(BrickData<T>),
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub(crate) enum NodeChildren<T: Default> {
    #[default]
    NoChildren,
    Children([T; BOX_NODE_CHILDREN_COUNT]),
}

/// Trait for User Defined Voxel Data
pub trait VoxelData:
    Default + Eq + Clone + Hash + Send + Sync + 'static + SerializableVoxelData
{
    /// Determines if the voxel is to be hit by rays in the raytracing algorithms
    fn is_empty(&self) -> bool;
}

#[cfg(feature = "bytecode")]
pub trait SerializableVoxelData: FromBencode + ToBencode {}

#[cfg(not(feature = "bytecode"))]
pub trait SerializableVoxelData {}

#[cfg(feature = "bytecode")]
impl<T> SerializableVoxelData for T where T: FromBencode + ToBencode {}

#[cfg(not(feature = "bytecode"))]
impl<T> SerializableVoxelData for T {}

/// Color properties of a voxel
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Albedo {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

pub(crate) type PaletteIndexValues = u32;
pub type OctreeMIPMapStrategy = HashMap<usize, MIPResamplingMethods>;

/// Implemented methods for MIP sampling. Default is set for
/// all MIP leveles not mentioned in the strategy
#[derive(Debug, Default, Clone, PartialEq)]
pub enum MIPResamplingMethods {
    /// MIP sampled from the MIPs below it, each voxel is the gamma corrected
    /// average of the voxels the cell contains on the same space one level below.
    /// (Gamma is set to be 2.)
    /// Warning: this introduces a significant amount of new colors into the palette
    #[default]
    BoxFilter,

    /// MIP sampled from the MIPs below it, each voxel is chosen from
    /// voxels the cell contains on the same space one level below,
    /// Introduces no new colors as current colors are reused
    PointFilter,

    /// Same as @PointFilter, but the voxels are sampled from
    /// the lowest possible level, instead of MIPs.
    /// It takes the most dominant voxel from the bottom, thus "BD"
    /// --> Bottom Dominant (It's nothing kinky)
    /// On level 1 it behaves like the regular version of itself
    PointFilterBD,

    /// MIP sampled from the MIPs below it, similar voxels are grouped together
    /// the sampled voxel is the average of the largest group
    /// @Albedo color range is assumed(0-255)
    /// f32 parameter is the threshold for similarity with a 0.001 resolution
    Posterize(f32),

    /// Same as @Posterize, but the voxels are sampled from
    /// the lowest possible level, instead of MIPs.
    /// It takes the most dominant voxel from the bottom, thus "BD"
    /// f32 parameter is the threshold for similarity with a 0.001 resolution
    /// On level 1 it behaves like the regular version of itself
    PosterizeBD(f32),
}

/// A helper object for setting Octree MIP map resampling strategy
pub struct StrategyUpdater<'a, T: Default + Clone + Eq + Hash>(pub(crate) &'a mut BoxTree<T>);

/// Configuration object for storing MIP map strategy
/// Don't forget to @recalculate_mip after you've enabled it, as it is
/// only updated on boxtree updates otherwise.
/// Activating MIP maps will require a larger GPU view (see @OctreeGPUHost::create_new_view)
/// As the MIP bricks will take space from other bricks.
#[derive(Clone)]
pub struct MIPMapStrategy {
    /// Decides if the strategy is enabled, see @Octree/node_mips
    pub(crate) enabled: bool,

    /// The MIP resampling strategy for different MIP levels
    pub(crate) resampling_methods: HashMap<usize, MIPResamplingMethods>,

    /// Color similarity threshold to reduce adding
    /// new colors during MIP operations for each MIP level. Has a resolution of 0.001
    pub(crate) resampling_color_matching_thresholds: HashMap<usize, f32>,
}

/// Data of nodes within a BoxTree
#[derive(Debug, Default, Clone)]
pub(crate) struct NodeData {
    /// Type and content information of the node
    pub(crate) content: NodeContent<PaletteIndexValues>,

    /// Node Child Connections
    pub(crate) children: NodeChildren<u32>,

    /// Brick data for each node containing a simplified representation, or all empties if the feature is disabled
    pub(crate) mip: BrickData<PaletteIndexValues>,

    /// Occupancy information of children on the bit-level
    pub(crate) occupied_bits: u64,

    /// Occlusion information for node sides
    ///  _===============================================================_
    /// | Byte 0   | node occlusion bits for sides:                      |
    /// |----------------------------------------------------------------|
    /// |  bit 0   | set if back side of the node is occluded            |
    /// |  bit 1   | set if front side of the node is occluded           |
    /// |  bit 2   | set if top of the node is occluded                  |
    /// |  bit 3   | set if bottom side of the node is occluded          |
    /// |  bit 4   | set if left side of the node is occluded            |
    /// |  bit 5   | set if right side of the node is occluded           |
    /// |  bit 6   | unused                                              |
    /// |  bit 7   | unused                                              |
    /// `================================================================`
    pub(crate) occlusion_bits: u8,
}

/// Data forwarded when a BoxTree is being updated
pub(crate) type BoxTreeUpdatedSignalParams = (BoxTreeNodeAccessStack, Vec<u8>);

/// A sequence of node_key, child sectant pairs describing the access path to a node
/// where each pair is inside the previous element(parent node, target sectant)
pub(crate) type BoxTreeNodeAccessStack = Vec<(usize, u8)>;

/// A function being called when the data in the tree is being updated
/// Fn(node_access_stack: Vec<(usize, u8)>, updated_sectants: Vec<u8>)
pub(crate) type BoxTreeUpdatedSignal = dyn Fn(BoxTreeNodeAccessStack, Vec<u8>) + Send + Sync;

/// Sparse 64Tree of Voxel Bricks, where each leaf node contains a brick of voxels.
/// A Brick is a 3 dimensional matrix, each element of it containing a voxel.
/// A Brick can be indexed directly, as opposed to the boxtree which is essentially a
/// tree-graph where each node has 64 children.
#[derive(Clone)]
pub struct BoxTree<T = u32>
where
    T: Default + Clone + Eq + Hash,
{
    /// Feature flag to enable/disable simplification attempts during boxtree update operations
    pub auto_simplify: bool,

    /// Size of one brick in a leaf node (dim^3)
    pub(crate) brick_dim: u32,

    /// Extent of the boxtree
    pub(crate) boxtree_size: u32,

    /// Storing data at each position through palette index values
    pub(crate) nodes: ObjectPool<NodeData>,

    /// The albedo colors used by the boxtree. Maximum 65535 colors can be used at once
    /// because of a limitation on GPU raytracing, to spare space index values refering
    /// the palettes are stored on 2 Bytes
    pub(crate) voxel_color_palette: Vec<Albedo>, // referenced by @nodes

    /// The different instances of user data stored within the boxtree
    /// Not sent to GPU
    pub(crate) voxel_data_palette: Vec<T>, // referenced by @nodes

    /// Cache variable to help find colors inside the color palette
    pub(crate) map_to_color_index_in_palette: HashMap<Albedo, usize>,

    /// Cache variable to help find user data in the palette
    pub(crate) map_to_data_index_in_palette: HashMap<T, usize>,

    /// The stored MIP map strategy
    pub(crate) mip_map_strategy: MIPMapStrategy,

    /// The signals to be called whenever the tree is updated
    pub(crate) update_triggers: Vec<Arc<BoxTreeUpdatedSignal>>,
}
