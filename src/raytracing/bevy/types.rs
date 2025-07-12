use crate::{
    boxtree::{BoxTree, V3c, V3cf32, VoxelData, types::PaletteIndexValues},
    spatial::Cube,
};
use bevy::{
    asset::Handle,
    ecs::resource::Resource,
    math::{UVec2, Vec4},
    prelude::Image,
    reflect::TypePath,
    render::{
        extract_resource::ExtractResource,
        render_graph::RenderLabel,
        render_resource::{
            BindGroup, BindGroupLayout, Buffer, CachedComputePipelineId, ShaderType,
        },
        renderer::RenderQueue,
    },
};
use bimap::BiHashMap;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    sync::{Arc, RwLock},
};

#[derive(Debug, Clone, ShaderType)]
pub struct BoxTreeMetaData {
    /// Color of the ambient light in the render
    pub ambient_light_color: V3cf32,

    /// Position of the ambient light in the render
    pub ambient_light_position: V3cf32,

    /// Size of the boxtree to display
    pub(crate) boxtree_size: u32,

    /// Contains the properties of the Octree
    ///  _===================================================================_
    /// | Byte 0-1 | Voxel Brick Dimension                                    |
    /// |=====================================================================|
    /// | Byte 2   | Features                                                 |
    /// |---------------------------------------------------------------------|
    /// |  bit 0   | 1 if MIP maps are enabled                                |
    /// |  bit 1   | unused                                                   |
    /// |  bit 2   | unused                                                   |
    /// |  bit 3   | unused                                                   |
    /// |  bit 4   | unused                                                   |
    /// |  bit 5   | unused                                                   |
    /// |  bit 6   | unused                                                   |
    /// |  bit 7   | unused                                                   |
    /// |=====================================================================|
    /// | Byte 3   | unused                                                   |
    /// `=====================================================================`
    pub(crate) tree_properties: u32,
}

#[derive(Debug, Clone, Copy, ShaderType)]
pub struct Viewport {
    /// The origin of the viewport, think of it as the position the eye
    pub(crate) origin: V3cf32,

    /// Delta position in case the viewport origin is displaced
    pub(crate) origin_delta: V3cf32,

    /// The direction the raycasts are based upon, think of it as wherever the eye looks
    pub direction: V3cf32,

    /// The volume the viewport reaches to
    /// * `x` - looking glass width
    /// * `y` - looking glass height
    /// * `z` - the max depth of the viewport
    pub frustum: V3cf32,

    /// Field of View: how scattered will the rays in the viewport are
    pub fov: f32,
}

#[derive(Resource, Clone, TypePath, ExtractResource)]
#[type_path = "shocovox::gpu::OctreeGPUHost"]
pub struct BoxTreeGPUHost<T = u32>
where
    T: Default + Clone + Eq + VoxelData + Send + Sync + Hash + 'static,
{
    pub tree: BoxTree<T>,
}

#[derive(Debug, Resource, Clone, TypePath)]
#[type_path = "shocovox::gpu::VhxViewSet"]
pub struct VhxViewSet {
    pub(crate) changed: bool,
    pub(crate) views: Vec<Arc<RwLock<BoxTreeGPUView>>>,
}

/// The Camera responsible for storing frustum and view related data
#[derive(Debug, Clone)]
pub struct BoxTreeSpyGlass {
    // The texture used to store depth information in the scene
    pub(crate) depth_texture: Handle<Image>,

    /// The currently used output texture
    pub(crate) output_texture: Handle<Image>,

    // Set to true, if the viewport changed
    pub(crate) viewport_changed: bool,

    // The viewport containing display information
    pub(crate) viewport: Viewport,
}

/// A View of an Octree
#[derive(Debug, Resource, Clone)]
pub struct BoxTreeGPUView {
    /// Buffers, layouts and bind groups for the view
    pub(crate) resources: Option<BoxTreeRenderDataResources>,

    /// The data handler responsible for uploading data to the GPU
    pub data_handler: BoxTreeGPUDataHandler,

    /// The plane for the basis of the raycasts
    pub spyglass: BoxTreeSpyGlass,

    /// Set to true if the view needs to be reloaded
    pub(crate) reload: bool,

    /// Set to true if the buffers in the view need to be resized
    pub(crate) resize: bool,

    /// Set to true if the view needs to be refreshed, e.g. by a resolution change
    pub(crate) rebuild: bool,

    /// Sets to true if new pipeline textures are ready
    pub(crate) new_images_ready: bool,

    /// The currently used resolution the raycasting dimensions are based for the base ray
    pub(crate) resolution: [u32; 2],

    /// The new resolution to be set if any
    pub(crate) new_resolution: Option<[u32; 2]>,

    /// The new depth texture to be used, if any
    pub(crate) new_depth_texture: Option<Handle<Image>>,

    /// The new output texture to be used, if any
    pub(crate) new_output_texture: Option<Handle<Image>>,

    pub(crate) brick_slot: Cube,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum BrickOwnedBy {
    None,
    NodeAsChild(u32, u8),
    NodeAsMIP(u32),
}

#[derive(Debug, Clone)]
pub(crate) struct NodeUploadRequest {
    pub(crate) node_key: usize,
    pub(crate) parent_key: usize,
    pub(crate) sectant: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct BrickUploadRequest {
    pub(crate) ownership: BrickOwnedBy,
    pub(crate) min_position: V3cf32,
}

#[derive(Debug, Clone)]
pub(crate) struct UploadQueueTargets {
    /// The nodes to upload to the GPU in ascending sorted order.
    /// This is a complete list of all the nodes required to be on the GPU.
    pub(crate) node_upload_queue: Vec<NodeUploadRequest>,

    /// This is a list of the bricks to upload to the GPU (bricks not yet uploaded)
    pub(crate) brick_upload_queue: Vec<BrickUploadRequest>,

    /// Map to connect brick indexes in GPU data to their counterparts in the tree
    pub(crate) brick_ownership: BiHashMap<usize, BrickOwnedBy>,

    /// Centerpoint of each brick; Valid only if the brick is owned!
    pub(crate) brick_positions: Vec<V3c<f32>>,

    /// Map to connect tree node keys to node meta indexes
    pub(crate) node_key_vs_meta_index: BiHashMap<usize, usize>,

    /// Map to connect nodes index values inside the GPU to their parents
    /// Mapping is as following: node_index -> (parent_index, child_sectant)
    pub(crate) node_index_vs_parent: HashMap<usize, (usize, u8)>,

    /// A set containing all nodes which should be on the GPU
    pub(crate) nodes_to_see: HashSet<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct UploadQueueStatus {
    /// The number of nodes already uploaded into the GPU
    /// from @node_upload_queue
    pub(crate) node_upload_progress: usize,

    /// The number of bricks already uploaded into the GPU
    /// from @node_upload_queue
    pub(crate) brick_upload_progress: usize,

    /// Index pointing inside GPU data where the search will start
    /// for the next brick to be overwritten
    pub(crate) victim_brick: usize,

    /// Index pointing inside GPU data where the search will start
    /// for the next node to be overwritten
    pub(crate) victim_node: usize,

    /// The number of colors uploaded to the GPU
    pub(crate) uploaded_color_palette_size: usize,
}

#[derive(Debug, Resource, Clone)]
pub struct BoxTreeGPUDataHandler {
    /// Tells the handler how many nodes to upload in one frame
    pub node_uploads_per_frame: usize,

    /// Tells the handler how many bricks to upload in one frame
    pub brick_uploads_per_frame: usize,

    /// Tells the handler how far away to look for bricks to find
    /// the furthest to unload
    pub brick_unload_search_perimeter: usize,

    /// The capacity for nodes within the view
    pub(crate) nodes_in_view: usize,

    /// The capacity for bricks within the view
    pub(crate) bricks_in_view: usize,

    /// The area on which node and brick upload is based on
    pub(crate) upload_range: Cube,

    /// The data the GPU displays
    pub(crate) render_data: BoxTreeRenderData,

    /// Target and progress data for GPU uploads
    pub(crate) upload_targets: UploadQueueTargets,

    /// Target and progress data for GPU uploads
    pub(crate) upload_state: UploadQueueStatus,
}

#[derive(Debug, Clone)]
pub(crate) struct BoxTreeRenderDataResources {
    pub(crate) render_stage_prepass_bind_group: BindGroup,
    pub(crate) render_stage_main_bind_group: BindGroup,

    // Spyglass group
    // --{
    pub(crate) spyglass_bind_group: BindGroup,
    pub(crate) viewport_buffer: Buffer,
    // }--

    // Octree render data group
    // --{
    pub(crate) tree_bind_group: BindGroup,
    pub(crate) boxtree_meta_buffer: Buffer,
    pub(crate) node_metadata_buffer: Buffer,
    pub(crate) node_children_buffer: Buffer,
    pub(crate) node_mips_buffer: Buffer,

    /// Buffer of Node occupancy bitmaps. Each node has a 64 bit bitmap,
    /// which is stored in 2 * u32 values. only available in GPU, to eliminate needles redundancy
    pub(crate) node_ocbits_buffer: Buffer,

    /// Buffer of Voxel Bricks. Each brick contains voxel_brick_dim^3 elements.
    /// Each Brick has a corresponding 64 bit occupancy bitmap in the @voxel_maps buffer.
    /// Only available in GPU, to eliminate needles redundancy
    pub(crate) voxels_buffer: Buffer,
    pub(crate) color_palette_buffer: Buffer,
    // }--
}

/// An update to a single brick inside the GPU cache in a view
#[derive(Default)]
pub(crate) struct BrickUpdate<'a> {
    pub(crate) brick_index: usize,
    pub(crate) data: &'a [PaletteIndexValues],
}

/// An update generated by a request to insert a node, brick or MIP
#[derive(Default)]
pub(crate) struct CacheUpdatePackage<'a> {
    /// true if the view needs a resize
    pub(crate) allocation_failed: bool,

    /// The bricks updated during the request
    pub(crate) brick_update: Option<BrickUpdate<'a>>,

    /// The list of modified nodes during the operation
    pub(crate) modified_nodes: Vec<usize>,
}

#[derive(Debug, Clone, TypePath)]
#[type_path = "shocovox::gpu::ShocoVoxRenderData"]
pub struct BoxTreeRenderData {
    /// CPU only field, contains stored MIP feature enabled state
    pub(crate) mips_enabled: bool,

    /// Contains the properties of the Octree
    pub(crate) boxtree_meta: BoxTreeMetaData,

    /// Node Property descriptors
    ///  _===============================================================_
    /// | Byte 0   | 8x 1 bit: 1 in case node is a leaf                  |
    /// |----------------------------------------------------------------|
    /// | Byte 1   | 8x 1 bit: 1 in case node is uniform                 |
    /// |----------------------------------------------------------------|
    /// | Byte 2   | unused                                              |
    /// |----------------------------------------------------------------|
    /// | Byte 3   | unused                                              |
    /// `================================================================`
    pub(crate) node_metadata: Vec<u32>,

    /// Composite field: Children information
    /// In case of Internal Nodes
    /// -----------------------------------------
    /// Index values for Nodes, 64 value per @SizedNode entry.
    /// Each value points to one of 64 children of the node,
    /// either pointing to a node in metadata, or marked empty
    /// when there are no children in the given sectant
    ///
    /// In case of Leaf Nodes:
    /// -----------------------------------------
    /// Contains 64 bricks pointing to the child of the node for the relevant sectant
    /// according to @node_metadata ( Uniform/Non-uniform ) a node may have 1
    /// or 64 children, in that case only the first index is used.
    /// Structure is as follows:
    ///  _===============================================================_
    /// | bit 0-30 | index of where the voxel brick starts               |
    /// |          | inside the @voxels_buffer(when parted)              |
    /// |          | or inside the @color_palette(when solid)            |
    /// |----------------------------------------------------------------|
    /// |   bit 31 | 0 if brick is parted, 1 if solid                    |
    /// `================================================================`
    pub(crate) node_children: Vec<u32>,

    /// Index values for node MIPs stored inside the bricks, each node has one MIP index, or marked empty
    /// Structure is the same as one child in @node_children
    pub(crate) node_mips: Vec<u32>,

    /// Buffer of Node occupancy bitmaps. Each node has a 64 bit bitmap,
    /// which is stored in 2 * u32 values
    pub(crate) node_ocbits: Vec<u32>,

    /// Stores each unique color, it is references in @voxels
    /// and in @children_buffer as well( in case of solid bricks )
    pub(crate) color_palette: Vec<Vec4>,
}

pub struct RenderBevyPlugin<T = u32>
where
    T: Default + Clone + Eq + VoxelData + Send + Sync + 'static,
{
    pub(crate) dummy: std::marker::PhantomData<T>,
}

pub(crate) const VHX_PREPASS_STAGE_ID: u32 = 0x01;
pub(crate) const VHX_RENDER_STAGE_ID: u32 = 0x02;

#[derive(Debug, Clone, Copy, ShaderType)]
pub(crate) struct RenderStageData {
    pub(crate) stage: u32,
    pub(crate) output_resolution: UVec2,
}

#[derive(Resource)]
pub(crate) struct VhxRenderPipeline {
    pub(crate) render_queue: RenderQueue,
    pub(crate) update_pipeline: CachedComputePipelineId,
    pub(crate) render_stage_bind_group_layout: BindGroupLayout,
    pub(crate) spyglass_bind_group_layout: BindGroupLayout,
    pub(crate) render_data_bind_group_layout: BindGroupLayout,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct VhxLabel;

pub(crate) struct VhxRenderNode {
    pub(crate) ready: bool,
}

#[cfg(test)]
mod types_wgpu_byte_compatibility_tests {
    use super::{BoxTreeMetaData, Viewport};
    use bevy::render::render_resource::encase::ShaderType;

    #[test]
    fn test_wgpu_compatibility() {
        Viewport::assert_uniform_compat();
        BoxTreeMetaData::assert_uniform_compat();
    }
}
