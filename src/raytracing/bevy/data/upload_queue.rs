use crate::{
    boxtree::{
        iterate::execute_for_relevant_sectants,
        types::{BrickData, NodeChildren, NodeContent},
        BoxTree, V3c, V3cf32, VoxelData, BOX_NODE_CHILDREN_COUNT, BOX_NODE_DIMENSION,
    },
    object_pool::empty_marker,
    raytracing::bevy::{
        data::{boxtree_properties, re_evaluate_view_size, write_range_to_buffer},
        types::{
            BoxTreeGPUHost, BrickOwnedBy, BrickOwnership, UploadQueueTargets,
            UploadQueueUpdateTask, VhxRenderPipeline, VhxViewSet,
        },
    },
    spatial::{raytracing::viewport_contains_target, Cube},
};
use bendy::{decoding::FromBencode, encoding::ToBencode};
use bevy::{
    ecs::system::{Res, ResMut},
    prelude::Commands,
    render::render_resource::encase::UniformBuffer,
    tasks::{block_on, futures_lite::future, AsyncComputeTaskPool},
};
use std::{collections::HashSet, hash::Hash};

impl UploadQueueTargets {
    pub(crate) fn reset(&mut self) {
        self.brick_ownership
            .write()
            .expect("Expected to be able to clear brick ownersip entries")
            .clear();
        self.node_key_vs_meta_index.clear();
        self.nodes_to_see.clear();
    }
}

//##############################################################################
//  ███████████   ██████████ ███████████  █████  █████ █████ █████       ██████████
// ░░███░░░░░███ ░░███░░░░░█░░███░░░░░███░░███  ░░███ ░░███ ░░███       ░░███░░░░███
//  ░███    ░███  ░███  █ ░  ░███    ░███ ░███   ░███  ░███  ░███        ░███   ░░███
//  ░██████████   ░██████    ░██████████  ░███   ░███  ░███  ░███        ░███    ░███
//  ░███░░░░░███  ░███░░█    ░███░░░░░███ ░███   ░███  ░███  ░███        ░███    ░███
//  ░███    ░███  ░███ ░   █ ░███    ░███ ░███   ░███  ░███  ░███      █ ░███    ███
//  █████   █████ ██████████ ███████████  ░░████████   █████ ███████████ ██████████
// ░░░░░   ░░░░░ ░░░░░░░░░░ ░░░░░░░░░░░    ░░░░░░░░   ░░░░░ ░░░░░░░░░░░ ░░░░░░░░░░
//##############################################################################
/// Recreates the list of nodes and bricks to upload based on the current position and view distance
pub(crate) fn rebuild<
    #[cfg(all(feature = "bytecode", feature = "serialization"))] T: FromBencode
        + ToBencode
        + Serializ
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
    tree: &BoxTree<T>,
    viewport_center_: V3cf32,
    view_distance: f32,
    brick_ownership: BrickOwnership,
) -> HashSet<usize> {
    let mut queue_population = HashSet::new();

    // Determine view center range
    let viewport_center = V3c::new(
        viewport_center_.x.clamp(0., tree.boxtree_size as f32),
        viewport_center_.y.clamp(0., tree.boxtree_size as f32),
        viewport_center_.z.clamp(0., tree.boxtree_size as f32),
    );
    let viewport_bl_ = viewport_center_ - V3c::unit(view_distance / 2.);
    let viewport_bl = V3c::new(
        viewport_bl_.x.clamp(0., tree.boxtree_size as f32),
        viewport_bl_.y.clamp(0., tree.boxtree_size as f32),
        viewport_bl_.z.clamp(0., tree.boxtree_size as f32),
    );
    let viewport_tr = viewport_bl + V3c::unit(view_distance);
    let viewport_tr = V3c::new(
        viewport_tr.x.clamp(0., tree.boxtree_size as f32),
        viewport_tr.y.clamp(0., tree.boxtree_size as f32),
        viewport_tr.z.clamp(0., tree.boxtree_size as f32),
    );

    // Decide the level boundaries to work within
    let max_mip_level = (tree.boxtree_size as f32 / tree.brick_dim as f32)
        .log(4.)
        .ceil() as u32;
    let deepest_mip_level_to_upload = ((viewport_bl_ - viewport_bl).length() / view_distance)
        .ceil()
        .min(max_mip_level as f32) as u32;

    // Look for the smallest node covering the entirety of the viewing distance
    let mut center_node_parent = None;
    let mut node_bounds = Cube::root_bounds(tree.boxtree_size as f32);
    let mut node_mip_level = max_mip_level;
    let mut node_stack = vec![(BoxTree::<T>::ROOT_NODE_KEY as usize)];
    loop {
        let node_key = node_stack.last().unwrap();
        if
        // current node is either leaf or empty
        matches!(tree.nodes.get(*node_key).content, NodeContent::Nothing | NodeContent::Leaf(_) | NodeContent::UniformLeaf(_))
        // or target child boundaries don't cover view distance
        || (node_bounds.size / BOX_NODE_DIMENSION as f32) <= view_distance
        || !node_bounds.contains(&viewport_bl)
        || !node_bounds.contains(&viewport_tr)
        {
            break;
        }

        // Hash the position to the target child
        let child_sectant_at_position = node_bounds.sectant_for(&viewport_center);
        let child_key_at_position = tree.nodes.get(*node_key).child(child_sectant_at_position);

        // There is a valid child at the given position inside the node, recurse into it
        if tree.nodes.key_is_valid(child_key_at_position) {
            debug_assert!(node_mip_level >= 1);
            center_node_parent = Some((*node_key, node_bounds, node_mip_level));
            node_stack.push(child_key_at_position);
            node_bounds = Cube::child_bounds_for(&node_bounds, child_sectant_at_position);
            node_mip_level -= 1;
        } else {
            break;
        }
    }

    // Add parent and children nodes into the upload queue and view set
    let center_node_key = node_stack.last().unwrap();
    queue_population.insert(*center_node_key);

    // add center node together with children inside the viewport into the queue
    add_children_to_upload_queue(
        center_node_parent.unwrap_or((*center_node_key, node_bounds, node_mip_level)),
        tree,
        &viewport_center,
        view_distance,
        deepest_mip_level_to_upload,
        &mut queue_population,
        brick_ownership,
    );

    queue_population
}

fn add_children_to_upload_queue<
    #[cfg(all(feature = "bytecode", feature = "serialization"))] T: FromBencode
        + ToBencode
        + Serializ
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
    (node_key, node_bounds, node_mip_level): (usize, Cube, u32),
    tree: &BoxTree<T>,
    viewport_center: &V3c<f32>,
    view_distance: f32,
    min_mip_level: u32,
    queue_population: &mut HashSet<usize>,
    brick_ownership: BrickOwnership,
) {
    debug_assert!(min_mip_level >= 1);
    if node_mip_level < min_mip_level {
        return;
    }

    debug_assert!(
        queue_population.contains(&node_key),
        "Expected node to be already included in the upload queue"
    );

    // the view distance has to be used for brick inclusion, but MIP should have
    // an extended inclusion range.
    let current_include_distance =
        view_distance * (BOX_NODE_DIMENSION as f32).powf(node_mip_level as f32 - 1.);
    let current_bl = V3c::from(*viewport_center - V3c::unit(current_include_distance / 2.));
    match &tree.nodes.get(node_key).content {
        NodeContent::Nothing | NodeContent::UniformLeaf(_) | NodeContent::Leaf(_) => {}
        NodeContent::Internal => {
            execute_for_relevant_sectants(
                &node_bounds,
                &current_bl,
                current_include_distance as u32,
                |position_in_target,
                 update_size_in_target,
                 target_child_sectant,
                 &target_bounds| {
                    if let Some(child_key) = tree.valid_child_for(node_key, target_child_sectant) {
                        if viewport_contains_target(
                            &current_bl,
                            current_include_distance,
                            &position_in_target,
                            &update_size_in_target,
                        ) {
                            queue_population.insert(child_key);
                            add_children_to_upload_queue(
                                (child_key, target_bounds, node_mip_level - 1),
                                tree,
                                viewport_center,
                                view_distance,
                                min_mip_level,
                                queue_population,
                                brick_ownership.clone(), // It's cloning an Arc
                            );
                        }
                    }
                },
            );
        }
    }
}

//##############################################################################
//  █████  █████ ███████████  █████          ███████      █████████   ██████████
// ░░███  ░░███ ░░███░░░░░███░░███         ███░░░░░███   ███░░░░░███ ░░███░░░░███
//  ░███   ░███  ░███    ░███ ░███        ███     ░░███ ░███    ░███  ░███   ░░███
//  ░███   ░███  ░██████████  ░███       ░███      ░███ ░███████████  ░███    ░███
//  ░███   ░███  ░███░░░░░░   ░███       ░███      ░███ ░███░░░░░███  ░███    ░███
//  ░███   ░███  ░███         ░███      █░░███     ███  ░███    ░███  ░███    ███
//  ░░████████   █████        ███████████ ░░░███████░   █████   █████ ██████████
//   ░░░░░░░░   ░░░░░        ░░░░░░░░░░░    ░░░░░░░    ░░░░░   ░░░░░ ░░░░░░░░░░
//##############################################################################
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
    mut upload_queue_update: Option<ResMut<UploadQueueUpdateTask>>,
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
    let mut updates = vec![];

    #[allow(clippy::reversed_empty_ranges)]
    let mut ocbits_updated = usize::MAX..0;

    // Decide target nodes/bricks to upload
    'upload_queue_update: {
        if view.reload {
            // rebuild upload queue if not already in progress
            if upload_queue_update.is_none() {
                let thread_pool = AsyncComputeTaskPool::get();
                let viewport_center = view.spyglass.viewport.origin.clone();
                let viewing_distance = view.spyglass.viewport.frustum.z;
                let brick_ownership = view.data_handler.upload_targets.brick_ownership.clone();
                let tree_arc = tree_host.tree.clone();
                commands.insert_resource(UploadQueueUpdateTask(thread_pool.spawn(async move {
                    rebuild::<T>(
                        &tree_arc
                            .read()
                            .expect("Expected to be able to read tree from GPU host"),
                        viewport_center,
                        viewing_distance,
                        brick_ownership,
                    )
                })));
            }

            // Set target_node_stack to the start of the tree
            view.data_handler.upload_state.target_node_stack = vec![(
                BoxTree::<T>::ROOT_NODE_KEY as usize,
                0,
                Cube::root_bounds(tree.get_size() as f32),
            )];

            // Upload root node to scene again
            let (new_node_index, new_node_update) = view.data_handler.add_node(
                &tree,
                BoxTree::<T>::ROOT_NODE_KEY as usize,
                BOX_NODE_CHILDREN_COUNT as u8,
            );
            debug_assert_eq!(0, new_node_index);
            updates.push(new_node_update);
            let mip_update = view
                .data_handler
                .add_brick(&tree, BrickOwnedBy::NodeAsMIP(BoxTree::<T>::ROOT_NODE_KEY));
            updates.push(mip_update);

            view.reload = false;
        }

        // If the upload queue update task is finished apply it!
        if let Some(ref mut upload_queue_update) = upload_queue_update {
            if let Some(population) = block_on(future::poll_once(&mut upload_queue_update.0)) {
                view.data_handler.upload_targets.nodes_to_see = population;
                commands.remove_resource::<UploadQueueUpdateTask>();
                view.data_handler.upload_state.target_node_stack = vec![(
                    BoxTree::<T>::ROOT_NODE_KEY as usize,
                    0,
                    Cube::root_bounds(
                        tree_host
                            .tree
                            .read()
                            .expect("Expected to be able to read BoxTree")
                            .get_size() as f32,
                    ),
                )];
            }
        }

        // Initiate pending update opertaion if there's any, but continue with uploads
        if let Some((viewport_center, viewing_distance)) =
            view.data_handler.pending_upload_queue_update
        {
            let thread_pool = AsyncComputeTaskPool::get();
            let brick_ownership = view.data_handler.upload_targets.brick_ownership.clone();
            let tree_arc = tree_host.tree.clone();
            commands.insert_resource(UploadQueueUpdateTask(thread_pool.spawn(async move {
                rebuild::<T>(
                    &tree_arc
                        .read()
                        .expect("Expected to be able to read tree from GPU host"),
                    viewport_center,
                    viewing_distance,
                    brick_ownership,
                )
            })));
            view.data_handler.pending_upload_queue_update = None;
        }

        // Decide on targets to upload this loop
        let viewing_distance = view.spyglass.viewport.frustum.z;
        let viewport_bl =
            V3c::<u32>::from(view.spyglass.viewport.origin - V3c::unit(viewing_distance / 2.));
        let data_handler = &mut view.data_handler;

        // Handle node uploads, if there are any
        'node_uploads: {
            if data_handler.upload_state.target_node_stack.is_empty() {
                break 'node_uploads;
            }

            for _ in 0..data_handler.node_uploads_per_frame {
                // Get next node to check
                let Some((parent_key, target_sectant, node_key, node_bounds)) = next_valid_node(
                    &tree,
                    &mut data_handler.upload_state.target_node_stack,
                    &data_handler.upload_targets.nodes_to_see,
                ) else {
                    break 'node_uploads;
                };
                debug_assert!(tree.nodes.key_is_valid(node_key));


                // Evaluate if target node is already uploaded
                if data_handler
                    .upload_targets
                    .node_key_vs_meta_index
                    .contains_left(&node_key)
                {
                    // Upload MIP again, if not present already
                    let mip_update =
                        data_handler.add_brick(&tree, BrickOwnedBy::NodeAsMIP(node_key as u32));
                    if mip_update.allocation_failed {
                        // Can't fit new mip brick into buffers, need to rebuild the pipeline
                        re_evaluate_view_size(&mut view);
                        break 'upload_queue_update; // voxel data still needs to be written out
                    }
                    updates.push(mip_update);
                } else {
                    // Upload Selected Node to GPU
                    let (new_node_index, new_node_update) =
                        data_handler.add_node(&tree, parent_key, target_sectant);

                    if new_node_update.allocation_failed {
                        // Can't fit new brick into buffers, need to rebuild the pipeline
                        re_evaluate_view_size(&mut view);
                        break 'upload_queue_update; // voxel data still needs to be written out
                    }
                    updates.push(new_node_update);

                    // Upload MIP to GPU
                    let mip_update =
                        data_handler.add_brick(&tree, BrickOwnedBy::NodeAsMIP(node_key as u32));

                    if mip_update.allocation_failed {
                        // Can't fit new MIP brick into buffers, need to rebuild the pipeline
                        re_evaluate_view_size(&mut view);
                        break 'upload_queue_update; // voxel data still needs to be written out
                    }
                    updates.push(mip_update);

                    // Also set the ocbits updated range
                    ocbits_updated.start = ocbits_updated.start.min(new_node_index * 2);
                    ocbits_updated.end = ocbits_updated.end.max(new_node_index * 2 + 2);
                }

                // Push the children into the brick upload list
                match &tree.nodes.get(node_key).content {
                    NodeContent::Nothing | NodeContent::Internal => {}
                    NodeContent::UniformLeaf(brick) => {
                        match &brick {
                            BrickData::Empty | BrickData::Solid(_) => {
                                // Empty brickdata is not uploaded,
                                // while solid brickdata should be present in the nodes data
                            }
                            BrickData::Parted(_brick) => {
                                let brick_ownership_entry = BrickOwnedBy::NodeAsChild(
                                    node_key as u32,
                                    0,
                                    V3c::from(node_bounds.min_position),
                                );
                                if viewport_contains_target(
                                    &viewport_bl,
                                    viewing_distance,
                                    &V3c::from(node_bounds.min_position),
                                    &V3c::unit(node_bounds.size as u32),
                                ) && !data_handler
                                    .upload_targets
                                    .brick_ownership
                                    .read()
                                    .expect("Expected to be able to read brick ownership entries")
                                    .contains_right(&brick_ownership_entry)
                                {
                                    data_handler
                                        .upload_state
                                        .bricks_to_upload
                                        .push(brick_ownership_entry);
                                }
                            }
                        };
                    }
                    NodeContent::Leaf(bricks) => {
                        execute_for_relevant_sectants(
                            &node_bounds,
                            &viewport_bl,
                            viewing_distance as u32,
                            |position_in_target,
                             update_size_in_target,
                             target_child_sectant,
                             &target_bounds| {
                                match &bricks[target_child_sectant as usize] {
                                    BrickData::Empty | BrickData::Solid(_) => {
                                        // Empty brickdata is not uploaded,
                                        // while solid brickdata should be present in the nodes data
                                    }
                                    BrickData::Parted(_brick) => {
                                        let brick_ownership_entry = BrickOwnedBy::NodeAsChild(
                                            node_key as u32,
                                            target_child_sectant,
                                            V3c::from(target_bounds.min_position),
                                        );

                                        if viewport_contains_target(
                                            &viewport_bl,
                                            viewing_distance,
                                            &position_in_target,
                                            &update_size_in_target,
                                        )
                                        && data_handler
                                            .upload_targets
                                            .brick_ownership
                                            .read()
                                            .expect("Expected to be able to read brick ownership entries")
                                            .get_by_right(&brick_ownership_entry)
                                            .is_none()
                                        {
                                            data_handler
                                                .upload_state
                                                .bricks_to_upload
                                                .push(brick_ownership_entry);
                                        }
                                    }
                                };
                            },
                        );
                    }
                }
            }
        }

        // upload bricks from the list
        if 0 == data_handler.upload_state.bricks_to_upload.len() {
            break 'upload_queue_update; // No bricks to upload!
        }
        let brick_requests = data_handler
            .upload_state
            .bricks_to_upload
            .drain(
                (data_handler
                    .upload_state
                    .bricks_to_upload
                    .len()
                    .saturating_sub(data_handler.brick_uploads_per_frame))..,
            )
            .collect::<Vec<_>>();
        for brick_request in brick_requests {
            if data_handler
                .upload_targets
                .brick_ownership
                .read()
                .expect("Expected to be able to read brick ownership entries")
                .contains_right(&brick_request)
            {
                continue;
            }

            let brick_update = data_handler.add_brick(&tree, brick_request.clone());

            if brick_update.allocation_failed {
                // Can't fit new brick brick into buffers, need to rebuild the pipeline
                re_evaluate_view_size(&mut view);
                break 'upload_queue_update; // voxel data still needs to be written out
            }
            updates.push(brick_update);
        }
    }

    if view.resources.is_none() {
        return; // Can't write to buffers as there are not created
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
    let mut node_mips_updated = usize::MAX..0; // Any brick upload could invalidate node_mips values
    for cache_update in updates.into_iter() {
        for node_index in cache_update.modified_nodes {
            node_meta_updated.start = node_meta_updated.start.min(node_index / 8);
            node_meta_updated.end = node_meta_updated.end.max(node_index / 8 + 1);
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

/// Provides the next valid node in the tree based on the given node_stack
/// If possible, returns with (parent_node_key, target_sectant, child_node_key, child_node_bounds)
fn next_valid_node<
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
    tree: &BoxTree<T>,
    node_stack: &mut Vec<(usize, u8, Cube)>,
    nodes_to_see: &HashSet<usize>,
) -> Option<(usize, u8, usize, Cube)> {
    // println!("==================================");
    let Some((current_node_key, mut target_sectant, current_node_bounds)) =
        node_stack.last().cloned()
    else {
        return None;
    };
    debug_assert!(tree.nodes.key_is_valid(current_node_key));
    loop {
        if (target_sectant as usize) >= BOX_NODE_CHILDREN_COUNT
            || tree.nodes.get(current_node_key).children == NodeChildren::NoChildren
        {
            node_stack.pop();
            if let Some((parent_key, target_of_parent, parent_bounds)) = node_stack.last_mut() {
                debug_assert_eq!(
                    tree.nodes.get(*parent_key).child(*target_of_parent),
                    current_node_key
                );
                debug_assert_eq!(
                    current_node_bounds,
                    parent_bounds.child_bounds_for(*target_of_parent)
                );
                *target_of_parent += 1;
                return Some((
                    *parent_key,
                    *target_of_parent - 1,
                    current_node_key,
                    current_node_bounds,
                ));
            }
            return None;
        }

        // Find next valid child
        let mut child_key = tree.nodes.get(current_node_key).child(target_sectant);
        while (target_sectant as usize) < BOX_NODE_CHILDREN_COUNT
            && (!tree.nodes.key_is_valid(child_key) || !nodes_to_see.contains(&child_key))
        {
            target_sectant += 1;
            if (target_sectant as usize) < BOX_NODE_CHILDREN_COUNT {
                child_key = tree.nodes.get(current_node_key).child(target_sectant);
            }
        }

        if (target_sectant as usize) >= BOX_NODE_CHILDREN_COUNT {
            continue;
        }
        // If child is a leaf node, or occluded
        if tree.nodes.get(child_key).children == NodeChildren::NoChildren
            || tree.nodes.get(child_key).is_occluded()
        {
            let result_target_sectant = target_sectant;
            let result_child_key = child_key;

            // Find next valid node ( be it child or parent )
            target_sectant += 1;
            if (target_sectant as usize) < BOX_NODE_CHILDREN_COUNT {
                child_key = tree.nodes.get(current_node_key).child(target_sectant);
            } else {
                child_key = empty_marker::<u32>() as usize;
            }
            while (target_sectant as usize) < BOX_NODE_CHILDREN_COUNT
                && !tree.nodes.key_is_valid(child_key)
            {
                target_sectant += 1;
                if (target_sectant as usize) < BOX_NODE_CHILDREN_COUNT {
                    child_key = tree.nodes.get(current_node_key).child(target_sectant);
                }
                // println!("----");
                // println!("target_sectant: {target_sectant}");
                // println!("child_key: {child_key}");
            }
            node_stack.last_mut().unwrap().1 = target_sectant;
            return Some((
                current_node_key,
                result_target_sectant,
                result_child_key,
                current_node_bounds,
            ));
        }

        // Return with current node, but go deeper in stack
        node_stack.last_mut().unwrap().1 = target_sectant;
        node_stack.push((
            child_key,
            0,
            current_node_bounds.child_bounds_for(target_sectant),
        ));
        return Some((
            current_node_key,
            target_sectant,
            child_key,
            current_node_bounds,
        ));
    }
}
