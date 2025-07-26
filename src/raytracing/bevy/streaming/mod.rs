mod cache;
pub(crate) mod types;
pub(crate) mod upload_queue;

use crate::{
    boxtree::{
        types::{BrickData, NodeContent},
        BoxTree, V3c, VoxelData, BOX_NODE_CHILDREN_COUNT,
    },
    object_pool::empty_marker,
    raytracing::bevy::{
        streaming::types::{BrickOwnedBy, BrickUpdate, CacheUpdatePackage, UploadQueueUpdateTask},
        types::{BoxTreeGPUHost, BoxTreeGPUView, VhxRenderPipeline, VhxViewSet},
    },
    spatial::Cube,
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

/// Process updates made to the Boxtree inside the given tree host
pub(crate) fn handle_tree_updates<
    'a,
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
    tree: &'a BoxTree<T>,
    tree_host: &'a BoxTreeGPUHost<T>,
    view: &mut BoxTreeGPUView,
    nodes_to_process: usize,
) -> Vec<CacheUpdatePackage<'a>> {
    let mut cache_updates = vec![];

    for _ in 0..nodes_to_process {
        let Some((node_access_stack, updated_sectants)) = tree_host
            .changes_buffer
            .write()
            .expect("Expected to be able to update BoxTree updates buffer")
            .pop_front()
        else {
            break;
        };

        debug_assert_eq!(
            BoxTree::<T>::ROOT_NODE_KEY as usize,
            node_access_stack.first().unwrap().0
        );

        // Upload changes to root node
        let data_handler = &mut view.data_handler;
        cache_updates.push(data_handler.add_node(
            &tree,
            BoxTree::<T>::ROOT_NODE_KEY as usize,
            BOX_NODE_CHILDREN_COUNT as u8,
        ));
        if let Some(existing_brick) = data_handler
            .upload_targets
            .brick_ownership
            .read()
            .expect("Expected to be able to read brick ownership entries")
            .get_by_right(&BrickOwnedBy::NodeAsMIP(BoxTree::<T>::ROOT_NODE_KEY))
        {
            match &tree.nodes.get(BoxTree::<T>::ROOT_NODE_KEY as usize).mip {
                BrickData::Empty | BrickData::Solid(_) => {}
                BrickData::Parted(brick) => {
                    cache_updates.push(CacheUpdatePackage {
                        allocation_failed: false,
                        added_node: None,
                        brick_updates: vec![BrickUpdate {
                            brick_index: *existing_brick,
                            data: &brick[..],
                        }],
                        modified_nodes: vec![],
                    });
                }
            }
        } else {
            let mip_update =
                data_handler.add_brick(&tree, BrickOwnedBy::NodeAsMIP(BoxTree::<T>::ROOT_NODE_KEY));
            if mip_update.allocation_failed {
                // Can't fit new MIP brick into buffers, need to rebuild the pipeline
                re_evaluate_view_size(view);
                return cache_updates; // voxel data still needs to be written out
            }
            cache_updates.push(mip_update);
        }

        // Upload rest of the update chain into GPU
        let parent_key = node_access_stack.last().unwrap().0;
        let mut node_bounds = Cube::root_bounds(tree.get_size() as f32);

        for (parent_key, _child_sectant) in node_access_stack.iter() {
            data_handler
                .upload_targets
                .nodes_to_see
                .write()
                .expect("Expected to be able to update list of nodes to display")
                .insert(*parent_key);
        }

        for (parent_key, child_sectant) in node_access_stack.into_iter() {
            // Also put the nodes edited into the visible nodes set
            // nodes outside the viewport will be removed at the next evaluation

            if let Some(node_key) = tree.valid_child_for(parent_key, child_sectant) {
                // Upload child Node to GPU
                let new_node_update = data_handler.add_node(&tree, parent_key, child_sectant);

                if new_node_update.allocation_failed {
                    // Can't fit new brick into buffers, need to rebuild the pipeline
                    re_evaluate_view_size(view);
                    return cache_updates; // voxel data still needs to be written out
                }
                cache_updates.push(new_node_update);

                // Upload MIP to GPU
                if let Some(existing_brick) = data_handler
                    .upload_targets
                    .brick_ownership
                    .read()
                    .expect("Expected to be able to read brick ownership entries")
                    .get_by_right(&BrickOwnedBy::NodeAsMIP(node_key as u32))
                {
                    match &tree.nodes.get(node_key).mip {
                        BrickData::Empty | BrickData::Solid(_) => {}
                        BrickData::Parted(brick) => {
                            cache_updates.push(CacheUpdatePackage {
                                allocation_failed: false,
                                added_node: None,
                                brick_updates: vec![BrickUpdate {
                                    brick_index: *existing_brick,
                                    data: &brick[..],
                                }],
                                modified_nodes: vec![],
                            });
                        }
                    }
                } else {
                    let mip_update =
                        data_handler.add_brick(&tree, BrickOwnedBy::NodeAsMIP(node_key as u32));
                    if mip_update.allocation_failed {
                        // Can't fit new MIP brick into buffers, need to rebuild the pipeline
                        re_evaluate_view_size(view);
                        return cache_updates; // voxel data still needs to be written out
                    }
                    cache_updates.push(mip_update);
                }
            }
            node_bounds = node_bounds.child_bounds_for(child_sectant);
        }

        // Upload child bricks
        match &tree.nodes.get(parent_key).content {
            NodeContent::Nothing | NodeContent::Internal => {
                // Update might have been a clear operation
            }
            NodeContent::UniformLeaf(brick) => match brick {
                BrickData::Empty | BrickData::Solid(_) => {}
                BrickData::Parted(brick) => {
                    debug_assert_eq!(updated_sectants, vec![0]);
                    let brick_ownership_entry = BrickOwnedBy::NodeAsChild(
                        parent_key as u32,
                        0,
                        V3c::from(node_bounds.min_position),
                    );
                    if let Some(brick_index) = data_handler
                        .upload_targets
                        .brick_ownership
                        .read()
                        .expect("Expected to be able to read brick ownership entries")
                        .get_by_right(&brick_ownership_entry)
                    {
                        cache_updates.push(CacheUpdatePackage {
                            allocation_failed: false,
                            added_node: None,
                            brick_updates: vec![BrickUpdate {
                                brick_index: *brick_index,
                                data: &brick[..],
                            }],
                            modified_nodes: vec![],
                        });
                    } else {
                        cache_updates.push(data_handler.add_brick(tree, brick_ownership_entry));
                    }
                }
            },
            NodeContent::Leaf(bricks) => {
                let mut new_brick_requests = vec![];
                let brick_update_requests = updated_sectants
                    .into_iter()
                    .map(|sec| {
                        let brick_ownership_entry = BrickOwnedBy::NodeAsChild(
                            parent_key as u32,
                            sec,
                            V3c::from(node_bounds.child_bounds_for(sec).min_position),
                        );
                        // If target child brick is not uploaded to GPU already, it needs to be added
                        if let Some(brick_index) = data_handler
                            .upload_targets
                            .brick_ownership
                            .read()
                            .expect("Expected to be able to read brick ownership entries")
                            .get_by_right(&brick_ownership_entry)
                        {
                            Some((*brick_index, sec))
                        } else {
                            new_brick_requests.push(brick_ownership_entry);
                            None
                        }
                    })
                    .flatten()
                    .collect::<Vec<_>>();

                // Re-upload relevant child bricks already inside the to GPU one more time
                for (brick_index, target_sectant) in brick_update_requests {
                    debug_assert!(matches!(
                        bricks[target_sectant as usize],
                        BrickData::Parted(_)
                    ));
                    match &bricks[target_sectant as usize] {
                        BrickData::Empty | BrickData::Solid(_) => {}
                        BrickData::Parted(brick) => {
                            cache_updates.push(CacheUpdatePackage {
                                allocation_failed: false,
                                added_node: None,
                                brick_updates: vec![BrickUpdate {
                                    brick_index: brick_index,
                                    data: &brick[..],
                                }],
                                modified_nodes: vec![],
                            });
                        }
                    }
                }

                // Upload new bricks
                for brick_request in new_brick_requests {
                    let brick_update = data_handler.add_brick(&tree, brick_request);
                    if brick_update.allocation_failed {
                        // Can't fit new brick brick into buffers, need to rebuild the pipeline
                        re_evaluate_view_size(view);
                        return cache_updates; // voxel data still needs to be written out
                    }
                    cache_updates.push(brick_update);
                }
            }
        }

        // If there are no more updates left to process, restart tree scan
        if tree_host
            .changes_buffer
            .read()
            .expect("Expected to be able to read BoxTree updates buffer")
            .is_empty()
        {
            // Set target_node_stack to the start of the tree
            view.data_handler.upload_state.target_node_stack = vec![(
                BoxTree::<T>::ROOT_NODE_KEY as usize,
                0,
                Cube::root_bounds(
                    tree_host
                        .tree
                        .read()
                        .expect("Expected to be able to read BoxTree size")
                        .get_size() as f32,
                ),
            )];
        }
    }
    cache_updates
}

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
        let new_node_count = (nodes_needed_overall as f32 * 1.2) as usize;
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
        let new_brick_count = (bricks_needed_overall as f32 * 1.2) as usize;
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
                BOX_NODE_CHILDREN_COUNT.hash(state)
            }
            BrickOwnedBy::None => {
                2.hash(state);
                u32::MAX.hash(state);
                BOX_NODE_CHILDREN_COUNT.hash(state)
            }
        }
    }
}

impl PartialEq for BrickOwnedBy {
    fn eq(&self, other: &BrickOwnedBy) -> bool {
        match (self, other) {
            (
                BrickOwnedBy::NodeAsChild(node_key, child_sectant, _),
                BrickOwnedBy::NodeAsChild(other_node_key, other_child_sectant, _),
            ) => node_key == other_node_key && child_sectant == other_child_sectant,
            (BrickOwnedBy::NodeAsMIP(node_key), BrickOwnedBy::NodeAsMIP(other_node_key)) => {
                node_key == other_node_key
            }
            _ => false,
        }
    }
}

impl CacheUpdatePackage<'_> {
    /// Error state when memory allocation failed for an item within the GPU buffers
    pub(crate) fn allocation_failed() -> Self {
        CacheUpdatePackage {
            allocation_failed: true,
            added_node: None,
            brick_updates: vec![],
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

    let cache_updates;
    if view.reload {
        cache_updates = upload_queue::process(
            &mut commands,
            &tree_binding,
            tree_host,
            &mut view,
            upload_queue_update,
        );
    } else {
        let nodes_to_process = view.data_handler.node_uploads_per_frame;
        let tree_updates =
            handle_tree_updates(&tree_binding, tree_host, &mut view, nodes_to_process);
        cache_updates = if tree_updates.is_empty() {
            upload_queue::process(
                &mut commands,
                &tree_binding,
                tree_host,
                &mut view,
                upload_queue_update,
            )
        } else {
            tree_updates
        };
    }

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
        for (node_index, modified_children_bitfield) in cache_update.modified_nodes {
            node_meta_updated.start = node_meta_updated.start.min(node_index / 8);
            node_meta_updated.end = node_meta_updated.end.max(node_index / 8 + 1);
            ocbits_updated.start = ocbits_updated.start.min(node_index * 2);
            ocbits_updated.end = ocbits_updated.end.max(node_index * 2 + 2);
            node_mips_updated.start = node_mips_updated.start.min(node_index);
            node_mips_updated.end = node_mips_updated.end.max(node_index + 1);
            for sectant in 0..BOX_NODE_CHILDREN_COUNT {
                if 0 != (modified_children_bitfield & (0x01 << sectant)) {
                    node_children_updated.start = node_children_updated
                        .start
                        .min(node_index * BOX_NODE_CHILDREN_COUNT + sectant);
                    node_children_updated.end = node_children_updated
                        .end
                        .max(node_index * BOX_NODE_CHILDREN_COUNT + sectant + 1);
                }
            }
        }

        // Upload Voxel data
        for modified_brick_data in cache_update.brick_updates {
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
