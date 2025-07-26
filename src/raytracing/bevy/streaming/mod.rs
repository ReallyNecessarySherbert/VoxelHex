mod cache;
pub(crate) mod types;
pub(crate) mod upload_queue;

use crate::{
    boxtree::{BoxTree, VoxelData, BOX_NODE_CHILDREN_COUNT},
    object_pool::empty_marker,
    raytracing::bevy::{
        streaming::types::{BrickOwnedBy, CacheUpdatePackage, UploadQueueUpdateTask},
        types::{BoxTreeGPUHost, BoxTreeGPUView, VhxRenderPipeline, VhxViewSet},
    },
};
use bendy::{decoding::FromBencode, encoding::ToBencode};
use bevy::{
    prelude::{Commands, Res, ResMut},
    render::{
        render_resource::{
            encase::{internal::WriteInto, UniformBuffer},
            Buffer, ShaderSize,
        },
        renderer::RenderQueue,
    },
};
use std::{
    hash::{Hash, Hasher},
    ops::Range,
};

pub(crate) fn boxtree_properties<
    #[cfg(all(feature = "bytecode", feature = "serialization"))] T: FromBencode
        + ToBencode
        + Serialize
        + DeserializeOwned
        + Default
        + Eq
        + Clone
        + Hash
        + VoxelData,
    #[cfg(all(feature = "bytecode", not(feature = "serialization")))] T: Default + Eq + Clone + Hash + VoxelData,
    #[cfg(all(not(feature = "bytecode"), feature = "serialization"))] T: Serialize + DeserializeOwned + Default + Eq + Clone + Hash + VoxelData,
    #[cfg(all(not(feature = "bytecode"), not(feature = "serialization")))] T: Default + Eq + Clone + Hash + VoxelData,
>(
    tree: &BoxTree<T>,
) -> u32 {
    (tree.brick_dim & 0x0000FFFF) | ((tree.mip_map_strategy.is_enabled() as u32) << 16)
}

/// Invalidates view to be rebuilt on the size needed by bricks and nodes
pub(crate) fn re_evaluate_view_size(view: &mut BoxTreeGPUView) {
    // Decide if there's enough space to host the required number of nodes
    let nodes_needed_overall = view
        .data_handler
        .upload_targets
        .nodes_to_see
        .read()
        .expect("Expected to be able to read list of nodes in view!")
        .len();
    let rebuild_nodes = nodes_needed_overall > view.data_handler.nodes_in_view;

    if rebuild_nodes {
        let new_node_count = (nodes_needed_overall as f32 * 1.1) as usize;
        let render_data = &mut view.data_handler.render_data;

        // Extend render data
        render_data
            .node_metadata
            .resize((new_node_count as f32 / 8.).ceil() as usize, 0);
        render_data
            .node_children
            .resize(new_node_count * BOX_NODE_CHILDREN_COUNT, empty_marker());
        render_data.node_mips.resize(new_node_count, empty_marker());
        render_data.node_ocbits.resize(new_node_count * 2, 0);
        view.data_handler.nodes_in_view = new_node_count;
    }

    // Decide if there's enough space to host the required number of bricks
    let bricks_needed_overall = view.data_handler.upload_state.bricks_to_upload.len()
        + view
            .data_handler
            .upload_targets
            .brick_ownership
            .read()
            .expect("Expected to be able to read brick ownership entries")
            .len()
        + nodes_needed_overall;
    let rebuild_bricks = bricks_needed_overall > view.data_handler.bricks_in_view;
    if rebuild_bricks {
        let new_brick_count = (bricks_needed_overall as f32 * 1.1) as usize;
        view.data_handler.bricks_in_view = new_brick_count;
    }

    debug_assert!(
        rebuild_nodes || rebuild_bricks,
        "Expected view to be too small while calling size evaluation!",
    );
    view.resize = true;
}

/// Converts the given array to `&[u8]` on the given range,
/// and schedules it to be written to the given buffer in the GPU
fn write_range_to_buffer<U>(
    array: &[U],
    index_range: Range<usize>,
    buffer: &Buffer,
    render_queue: &RenderQueue,
) where
    U: Send + Sync + 'static + ShaderSize + WriteInto,
{
    if !index_range.is_empty() {
        let element_size = std::mem::size_of_val(&array[0]);
        let byte_offset = (index_range.start * element_size) as u64;
        let slice = array.get(index_range.clone()).unwrap_or_else(|| {
            panic!(
                "{}",
                format!(
                    "Expected range {:?} to be in bounds of {:?}",
                    index_range,
                    array.len(),
                )
                .to_owned()
            )
        });
        unsafe {
            render_queue.write_buffer(buffer, byte_offset, slice.align_to::<u8>().1);
        }
    }
}

impl Hash for BrickOwnedBy {
    fn hash<H: Hasher>(&self, state: &mut H) -> () {
        match self {
            BrickOwnedBy::NodeAsChild(node_key, child_sectant, _brick_position) => {
                0.hash(state);
                node_key.hash(state);
                child_sectant.hash(state)
            }
            BrickOwnedBy::NodeAsMIP(node_key) => {
                1.hash(state);
                node_key.hash(state);
            }
            BrickOwnedBy::None => {
                2.hash(state);
            }
        }
    }
}

impl CacheUpdatePackage<'_> {
    /// Error state when memory allocation failed for an item within the GPU buffers
    pub(crate) fn allocation_failed() -> Self {
        CacheUpdatePackage {
            allocation_failed: true,
            added_node: None,
            brick_update: None,
            modified_nodes: vec![],
        }
    }
}

/// Continually upload node and brick data to GPU
pub(crate) fn upload<
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
    #[cfg(all(not(feature = "bytecode"), feature = "serialization"))] T: Serialize + DeserializeOwned + Default + Eq + Clone + Hash + VoxelData + Send + Sync + 'static,
    #[cfg(all(not(feature = "bytecode"), not(feature = "serialization")))] T: Default + Eq + Clone + Hash + VoxelData + Send + Sync + 'static,
>(
    mut commands: Commands,
    mut viewset: Option<ResMut<VhxViewSet>>,
    tree_gpu_host: Option<Res<BoxTreeGPUHost<T>>>,
    mut vhx_pipeline: Option<ResMut<VhxRenderPipeline>>,
    upload_queue_update: Option<ResMut<UploadQueueUpdateTask>>,
) {
    let (Some(pipeline), Some(tree_host), Some(viewset)) = (
        vhx_pipeline.as_mut(),
        tree_gpu_host.as_ref(),
        viewset.as_mut(),
    ) else {
        return; // Nothing to do without the required resources
    };

    if viewset.is_empty() {
        return; // Nothing to do without views..
    }

    let tree = tree_host
        .tree
        .read()
        .expect("Expected to be able to read tree from GPU host");
    let mut view = viewset.view_mut(0).unwrap();
    if view.resources.is_none() {
        return; // Can't write to buffers as there are not created
    }
    // Decide target nodes/bricks to upload
    let tree_binding = tree_host
        .tree
        .read()
        .expect("Expected to be able to read BoxTree");
    let cache_updates = crate::raytracing::bevy::streaming::upload_queue::process(
        &mut commands,
        &tree_binding,
        tree_host,
        &mut view,
        upload_queue_update,
    );

    // Apply writes to GPU
    let render_queue = &pipeline.render_queue;

    // Data updates for spyglass viewport
    if view.spyglass.viewport_changed {
        view.spyglass.viewport_changed = false;

        let mut buffer = UniformBuffer::new(Vec::<u8>::new());
        buffer.write(&view.spyglass.viewport).unwrap();
        render_queue.write_buffer(
            &view.resources.as_ref().unwrap().viewport_buffer,
            0,
            &buffer.into_inner(),
        );
    }

    // Data updates for BoxTree MIP map feature
    if view.data_handler.render_data.mips_enabled != tree.mip_map_strategy.is_enabled() {
        // Regenerate feature bits
        view.data_handler.render_data.boxtree_meta.tree_properties = boxtree_properties(&tree);

        // Write to GPU
        let mut buffer = UniformBuffer::new(Vec::<u8>::new());
        buffer
            .write(&view.data_handler.render_data.boxtree_meta)
            .unwrap();
        pipeline.render_queue.write_buffer(
            &view.resources.as_ref().unwrap().node_metadata_buffer,
            0,
            &buffer.into_inner(),
        );
        view.data_handler.render_data.mips_enabled = tree.mip_map_strategy.is_enabled()
    }

    // Data updates for color palette
    let host_color_count = tree.map_to_color_index_in_palette.keys().len();
    let color_palette_size_diff =
        host_color_count - view.data_handler.upload_state.uploaded_color_palette_size;

    debug_assert!(
        host_color_count >= view.data_handler.upload_state.uploaded_color_palette_size,
        "Expected host color palette({:?}), to be larger, than colors stored on the GPU({:?})",
        host_color_count,
        view.data_handler.upload_state.uploaded_color_palette_size
    );

    if 0 < color_palette_size_diff {
        for i in view.data_handler.upload_state.uploaded_color_palette_size..host_color_count {
            view.data_handler.render_data.color_palette[i] = tree.voxel_color_palette[i].into();
        }

        // Upload color palette delta to GPU
        write_range_to_buffer(
            &view.data_handler.render_data.color_palette,
            (host_color_count - color_palette_size_diff)..(host_color_count),
            &view.resources.as_ref().unwrap().color_palette_buffer,
            render_queue,
        );
    }
    view.data_handler.upload_state.uploaded_color_palette_size =
        tree.map_to_color_index_in_palette.keys().len();

    // compile cache updates into write batches
    #[allow(clippy::reversed_empty_ranges)]
    let mut node_meta_updated = usize::MAX..0;

    #[allow(clippy::reversed_empty_ranges)]
    let mut node_children_updated = usize::MAX..0;

    #[allow(clippy::reversed_empty_ranges)]
    let mut ocbits_updated = usize::MAX..0;

    #[allow(clippy::reversed_empty_ranges)]
    let mut node_mips_updated = usize::MAX..0; // Any brick upload could invalidate node_mips values
    for cache_update in cache_updates.into_iter() {
        if let Some(added_node_index) = cache_update.added_node {
            ocbits_updated.start = ocbits_updated.start.min(added_node_index * 2);
            ocbits_updated.end = ocbits_updated.end.max(added_node_index * 2 + 2);
            node_meta_updated.start = node_meta_updated.start.min(added_node_index / 8);
            node_meta_updated.end = node_meta_updated.end.max(added_node_index / 8 + 1);
        }

        for node_index in cache_update.modified_nodes {
            node_mips_updated.start = node_mips_updated.start.min(node_index);
            node_mips_updated.end = node_mips_updated.end.max(node_index + 1);
            node_children_updated.start = node_children_updated
                .start
                .min(node_index * BOX_NODE_CHILDREN_COUNT);
            node_children_updated.end = node_children_updated
                .end
                .max(node_index * BOX_NODE_CHILDREN_COUNT + BOX_NODE_CHILDREN_COUNT);
        }

        // Upload Voxel data
        if let Some(modified_brick_data) = cache_update.brick_update {
            let voxel_start_index =
                modified_brick_data.brick_index * modified_brick_data.data.len();
            debug_assert_eq!(
                modified_brick_data.data.len(),
                tree.brick_dim.pow(3) as usize,
                "Expected Brick slice to align to tree brick dimension"
            );
            unsafe {
                render_queue.write_buffer(
                    &view.resources.as_ref().unwrap().voxels_buffer,
                    (voxel_start_index * std::mem::size_of_val(&modified_brick_data.data[0]))
                        as u64,
                    modified_brick_data.data.align_to::<u8>().1,
                );
            }
        }
    }

    write_range_to_buffer(
        &view.data_handler.render_data.node_metadata,
        node_meta_updated,
        &view.resources.as_ref().unwrap().node_metadata_buffer,
        render_queue,
    );
    write_range_to_buffer(
        &view.data_handler.render_data.node_children,
        node_children_updated,
        &view.resources.as_ref().unwrap().node_children_buffer,
        render_queue,
    );
    write_range_to_buffer(
        &view.data_handler.render_data.node_ocbits,
        ocbits_updated,
        &view.resources.as_ref().unwrap().node_ocbits_buffer,
        render_queue,
    );
    write_range_to_buffer(
        &view.data_handler.render_data.node_mips,
        node_mips_updated,
        &view.resources.as_ref().unwrap().node_mips_buffer,
        render_queue,
    );
}
