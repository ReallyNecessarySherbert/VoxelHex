pub use crate::raytracing::bevy::types::{
    BoxTreeGPUDataHandler, BoxTreeGPUHost, BoxTreeGPUView, BoxTreeMetaData, BoxTreeSpyGlass,
    VhxViewSet, Viewport,
};
use crate::{
    boxtree::{V3c, V3cf32, VoxelData, BOX_NODE_CHILDREN_COUNT},
    object_pool::empty_marker,
    raytracing::{
        bevy::{
            data::boxtree_properties,
            types::{UploadQueueStatus, UploadQueueTargets},
        },
        BoxTreeRenderData,
    },
    spatial::Cube,
};
use bendy::{decoding::FromBencode, encoding::ToBencode};
use bevy::{
    prelude::{Assets, Handle, Image, Res, ResMut, Vec4},
    render::{
        render_asset::RenderAssetUsages,
        render_asset::RenderAssets,
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        texture::GpuImage,
    },
};
use bimap::BiHashMap;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    sync::{Arc, RwLock},
};

impl<
        #[cfg(all(feature = "bytecode", feature = "serialization"))] T: FromBencode
            + ToBencode
            + Serialize
            + DeserializeOwned
            + Default
            + Eq
            + Clone
            + Hash
            + VoxelData
            + Send
            + Sync
            + 'static,
        #[cfg(all(feature = "bytecode", not(feature = "serialization")))] T: FromBencode + ToBencode + Default + Eq + Clone + Hash + VoxelData + Send + Sync + 'static,
        #[cfg(all(not(feature = "bytecode"), feature = "serialization"))] T: Serialize
            + DeserializeOwned
            + Default
            + Eq
            + Clone
            + Hash
            + VoxelData
            + Send
            + Sync
            + 'static,
        #[cfg(all(not(feature = "bytecode"), not(feature = "serialization")))] T: Default + Eq + Clone + Hash + VoxelData + Send + Sync + 'static,
    > BoxTreeGPUHost<T>
{
    /// Creates GPU compatible data renderable on the GPU from an BoxTree
    pub fn create_new_view(
        &mut self,
        viewset: &mut VhxViewSet,
        viewport: Viewport,
        resolution: [u32; 2],
        mut images: ResMut<Assets<Image>>,
    ) -> usize {
        let tree = &self
            .tree
            .read()
            .expect("Expected to be able to read tree from GPU host");

        // This is an estimation for the required number of nodes in the given view
        // which sums the number of nodes within the viewport on every level
        let nodes_in_view: usize = {
            let tree_max_levels = (tree.boxtree_size as f32 / tree.brick_dim as f32)
                .log(4.)
                .ceil() as u32;
            (1..=tree_max_levels)
                .map(|level| {
                    let view_distance = viewport.frustum.z;
                    let cube_size_at_level = tree.brick_dim * 4_u32.pow(level);
                    (view_distance / cube_size_at_level as f32).ceil().powf(3.) as usize
                })
                .sum()
        };

        let bricks_in_view = (
            // Bricks in view
            ((viewport.frustum.z / tree.brick_dim as f32).ceil() as usize).pow(3)
                // MIPs in view
                + nodes_in_view
        ) / 4;

        let gpu_data_handler = BoxTreeGPUDataHandler {
            upload_range: Cube {
                min_position: viewport.origin - V3c::unit(viewport.frustum.z / 2.),
                size: viewport.frustum.z,
            },
            render_data: BoxTreeRenderData {
                mips_enabled: tree.mip_map_strategy.is_enabled(),
                boxtree_meta: BoxTreeMetaData {
                    boxtree_size: tree.boxtree_size,
                    tree_properties: boxtree_properties(&tree),
                    ambient_light_color: V3c::new(1., 1., 1.),
                    ambient_light_position: V3c::new(
                        tree.boxtree_size as f32,
                        tree.boxtree_size as f32,
                        tree.boxtree_size as f32,
                    ),
                },
                node_metadata: vec![0; (nodes_in_view as f32 / 8.).ceil() as usize],
                node_ocbits: vec![0; nodes_in_view * 2],
                node_children: vec![empty_marker(); nodes_in_view * BOX_NODE_CHILDREN_COUNT],
                node_mips: vec![empty_marker(); nodes_in_view],
                color_palette: vec![Vec4::ZERO; u16::MAX as usize],
            },
            upload_targets: UploadQueueTargets {
                nodes_to_see: HashSet::new(),
                brick_ownership: Arc::default(),
                node_key_vs_meta_index: BiHashMap::new(),
                node_index_vs_parent: HashMap::new(),
            },
            upload_state: UploadQueueStatus {
                victim_node: 0,
                victim_brick: 0,
                bricks_to_upload: vec![],
                target_node_stack: vec![],
                uploaded_color_palette_size: 0,
            },
            pending_upload_queue_update: None,
            nodes_in_view,
            bricks_in_view,
            node_uploads_per_frame: 50,
            brick_uploads_per_frame: 50,
            brick_unload_search_perimeter: 8,
        };
        let output_texture = create_output_texture(resolution, &mut images);

        viewset.views.push(Arc::new(RwLock::new(BoxTreeGPUView {
            resolution,
            reload: true,
            rebuild: false,
            resize: false,
            new_images_ready: true,
            new_resolution: None,
            new_output_texture: None,
            new_depth_texture: None,
            data_handler: gpu_data_handler,
            resources: None,
            brick_slot: Cube::brick_slot_for(&viewport.origin, tree.brick_dim),
            spyglass: BoxTreeSpyGlass {
                depth_texture: create_depth_texture(resolution, &mut images),
                output_texture,
                viewport_changed: true,
                viewport,
            },
        })));
        viewset.changed = true;
        viewset.views.len() - 1
    }
}

impl BoxTreeGPUView {
    /// Erases the whole view to be uploaded to the GPU again
    pub fn reload(&mut self) {
        self.data_handler.upload_targets.reset();
        self.reload = true;
    }

    /// Provides the handle to the output texture
    /// Warning! Handle will no longer being updated after resolution change
    pub fn output_texture(&self) -> &Handle<Image> {
        &self.spyglass.output_texture
    }

    /// Updates the resolution on which the view operates on.
    /// It will make a new output texture if size is larger, than the current output texture
    pub fn set_resolution(
        &mut self,
        resolution: [u32; 2],
        images: &mut ResMut<Assets<Image>>,
    ) -> Handle<Image> {
        if self.resolution != resolution {
            self.new_resolution = Some(resolution);
            self.new_output_texture = Some(create_output_texture(resolution, images));
            self.new_depth_texture = Some(create_depth_texture(resolution, images));
            self.rebuild = true;
            self.new_images_ready = false;
            self.new_output_texture.as_ref().unwrap().clone_weak()
        } else {
            self.spyglass.output_texture.clone_weak()
        }
    }

    /// Provides currently used resolution for the view
    pub fn resolution(&self) -> [u32; 2] {
        self.resolution
    }
}

impl BoxTreeSpyGlass {
    pub fn viewport(&self) -> &Viewport {
        &self.viewport
    }
    pub fn view_frustum(&self) -> &V3cf32 {
        &self.viewport.frustum
    }
    pub fn view_fov(&self) -> f32 {
        self.viewport.fov
    }
    pub fn viewport_mut(&mut self) -> &mut Viewport {
        self.viewport_changed = true;
        &mut self.viewport
    }
}

impl Viewport {
    /// Creates a viewport based on the given parameters
    pub fn new(origin: V3cf32, direction: V3cf32, frustum: V3cf32, fov: f32) -> Self {
        Self {
            origin,
            origin_delta: V3c::unit(0.),
            direction,
            frustum,
            fov,
        }
    }

    /// Provides the point the viewport originates rays from. All rays point away from this point.
    pub const fn origin(&self) -> V3cf32 {
        self.origin
    }

    /// Moves the viewports origin with the given delta position
    pub fn move_viewport(&mut self, delta: V3cf32) {
        self.origin_delta += delta;
        self.origin += delta;
    }

    /// Sets the VIewports origin to the given position
    pub fn set_viewport_origin(&mut self, new_origin: V3cf32) {
        self.origin_delta += self.origin - new_origin;
        self.origin = new_origin;
    }
}

pub(crate) fn create_output_texture(
    resolution: [u32; 2],
    images: &mut ResMut<Assets<Image>>,
) -> Handle<Image> {
    let mut output_texture = Image::new_fill(
        Extent3d {
            width: resolution[0],
            height: resolution[1],
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    output_texture.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::RENDER_ATTACHMENT;
    images.add(output_texture)
}

/// Create a depth texture for the given output resolutions
/// Depth texture resolution should cover a single voxel
pub(crate) fn create_depth_texture(
    resolution: [u32; 2],
    images: &mut ResMut<Assets<Image>>,
) -> Handle<Image> {
    let mut depth_texture = Image::new_fill(
        Extent3d {
            width: resolution[0] / 2,
            height: resolution[1] / 2,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    depth_texture.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    images.add(depth_texture)
}

pub(crate) fn handle_resolution_updates_main_world(mut viewset: Option<ResMut<VhxViewSet>>) {
    if let Some(viewset) = viewset.as_mut() {
        if viewset.is_empty() {
            return; // Nothing to do without views..
        }
        let mut current_view = viewset.views[0].write().unwrap();
        if current_view.new_images_ready && current_view.new_resolution.is_some() {
            current_view.resolution = current_view.new_resolution.take().unwrap();
            current_view.spyglass.output_texture = current_view.new_output_texture.clone().unwrap();
            current_view.spyglass.depth_texture = current_view.new_depth_texture.clone().unwrap();
        }
    }
}

pub(crate) fn handle_resolution_updates_render_world(
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut viewset: Option<ResMut<VhxViewSet>>,
) {
    if let Some(viewset) = viewset.as_mut() {
        let Some(mut view) = viewset.view_mut(0) else {
            return; // Nothing to do without views..
        };
        if view.new_resolution.is_some() {
            debug_assert!(
                !view.new_images_ready,
                "Expected images ready flag to be false before images are taken over"
            );
            view.new_images_ready = gpu_images
                .get(view.new_output_texture.as_ref().unwrap())
                .is_some()
                && gpu_images
                    .get(view.new_depth_texture.as_ref().unwrap())
                    .is_some();

            if view.new_images_ready && view.new_resolution.is_some() {
                view.resolution = view.new_resolution.take().unwrap();
                view.spyglass.output_texture = view.new_output_texture.clone().unwrap();
                view.spyglass.depth_texture = view.new_depth_texture.clone().unwrap();
            }
        }
    }
}
