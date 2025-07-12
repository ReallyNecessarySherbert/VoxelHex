use crate::{
    boxtree::{
        iterate::execute_for_relevant_sectants,
        types::{BrickData, NodeContent},
        BoxTree, VoxelData, V3c, V3cf32, BOX_NODE_CHILDREN_COUNT, BOX_NODE_DIMENSION,
    },
    object_pool::empty_marker,
    raytracing::bevy::{
        data::{boxtree_properties, re_evaluate_view_size, write_range_to_buffer},
        types::{
            BoxTreeGPUHost, BrickOwnedBy, BrickUploadRequest, NodeUploadRequest,
            UploadQueueTargets, VhxRenderPipeline, VhxViewSet,
        },
    },
    spatial::Cube,
};
use bevy::{
    ecs::system::{Res, ResMut},
    render::render_resource::encase::UniformBuffer,
};

impl UploadQueueTargets {
    pub(crate) fn reset(&mut self) {
        self.node_upload_queue.clear();
        self.brick_upload_queue.clear();
        self.brick_ownership.clear();
        self.node_key_vs_meta_index.clear();
        self.nodes_to_see.clear();
    }
}

/// Recreates the list of nodes and bricks to upload based on the current position and view distance
pub(crate) fn rebuild<T: VoxelData>(
    tree: &BoxTree<T>,
    viewport_center_: &V3cf32,
    view_distance: f32,
    upload_targets: &mut UploadQueueTargets,
) {
    upload_targets.nodes_to_see.clear();
    upload_targets.brick_upload_queue.clear();
    upload_targets.node_upload_queue.clear();

    // Determine view center range
    let viewport_center = V3c::new(
        viewport_center_.x.clamp(0., tree.boxtree_size as f32),
        viewport_center_.y.clamp(0., tree.boxtree_size as f32),
        viewport_center_.z.clamp(0., tree.boxtree_size as f32),
    );
    let viewport_bl_ = *viewport_center_ - V3c::unit(view_distance / 2.);
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
    let mut node_stack = vec![NodeUploadRequest {
        node_key: BoxTree::<T>::ROOT_NODE_KEY as usize,
        parent_key: BoxTree::<T>::ROOT_NODE_KEY as usize,
        sectant: 0,
    }];
    loop {
        let node_key = node_stack.last().unwrap().node_key;
        if
        // current node is either leaf or empty
        matches!(tree.nodes.get(node_key), NodeContent::Nothing | NodeContent::Leaf(_) | NodeContent::UniformLeaf(_))
        // or target child boundaries don't cover view distance
        || (node_bounds.size / BOX_NODE_DIMENSION as f32) <= view_distance
        || !node_bounds.contains(&viewport_bl)
        || !node_bounds.contains(&viewport_tr)
        {
            break;
        }

        // Hash the position to the target child
        let child_sectant_at_position = node_bounds.sectant_for(&viewport_center);
        let child_key_at_position = tree.node_children[node_key].child(child_sectant_at_position);

        // There is a valid child at the given position inside the node, recurse into it
        if tree.nodes.key_is_valid(child_key_at_position) {
            debug_assert!(node_mip_level >= 1);
            center_node_parent = Some((node_key, node_bounds, node_mip_level));
            node_stack.push(NodeUploadRequest {
                node_key: child_key_at_position,
                parent_key: node_key,
                sectant: child_sectant_at_position,
            });
            node_bounds = Cube::child_bounds_for(&node_bounds, child_sectant_at_position);
            node_mip_level -= 1;
        } else {
            break;
        }
    }

    // Add parent and children nodes into the upload queue and view set
    let center_node_key = node_stack.last().unwrap().node_key;
    upload_targets.node_upload_queue.append(
        &mut node_stack
            .drain(..)
            .inspect(|v| {
                upload_targets.nodes_to_see.insert(v.node_key);
            })
            .collect(),
    );

    // add center node together with children inside the viewport into the queue
    add_children_to_upload_queue(
        center_node_parent.unwrap_or((center_node_key, node_bounds, node_mip_level)),
        tree,
        &viewport_center,
        view_distance,
        upload_targets,
        deepest_mip_level_to_upload,
    );
}

fn add_children_to_upload_queue<T: VoxelData>(
    (node_key, node_bounds, node_mip_level): (usize, Cube, u32),
    tree: &BoxTree<T>,
    viewport_center: &V3c<f32>,
    view_distance: f32,
    upload_targets: &mut UploadQueueTargets,
    min_mip_level: u32,
) {
    debug_assert!(min_mip_level >= 1);
    if node_mip_level < min_mip_level {
        return;
    }

    let viewport_contains_target_fn = |viewport_bl: &V3c<u32>,
                                       view_distance: f32,
                                       start_position_in_target: &V3c<u32>,
                                       update_size_in_target: &V3c<u32>|
     -> bool {
        !((viewport_bl.x + view_distance as u32) <= start_position_in_target.x
            || (start_position_in_target.x + update_size_in_target.x) <= viewport_bl.x
            || (viewport_bl.y + view_distance as u32) <= start_position_in_target.y
            || (start_position_in_target.y + update_size_in_target.y) <= viewport_bl.y
            || (viewport_bl.z + view_distance as u32) <= start_position_in_target.z
            || (start_position_in_target.z + update_size_in_target.z) <= viewport_bl.z)
    };

    debug_assert!(
        upload_targets.nodes_to_see.contains(&node_key),
        "Expected node to be already included in the upload queue"
    );

    // the view distance has to be used for brick inclusion, but MIP should have
    // an extended inclusion range.
    let current_include_distance =
        view_distance * (BOX_NODE_DIMENSION as f32).powf(node_mip_level as f32 - 1.);
    let current_bl = V3c::from(*viewport_center - V3c::unit(current_include_distance / 2.));
    match tree.nodes.get(node_key) {
        NodeContent::Nothing => {}
        NodeContent::UniformLeaf(brick) => {
            match &brick {
                BrickData::Empty | BrickData::Solid(_) => {
                    // Empty brickdata is not uploaded,
                    // while solid brickdata should be present in the nodes data
                }
                BrickData::Parted(_brick) => {
                    let brick_ownership = BrickOwnedBy::NodeAsChild(node_key as u32, 0);
                    if viewport_contains_target_fn(
                        &current_bl,
                        current_include_distance,
                        &V3c::from(node_bounds.min_position),
                        &V3c::unit(node_bounds.size as u32),
                    ) && upload_targets
                        .brick_ownership
                        .get_by_right(&brick_ownership)
                        .is_none()
                    {
                        upload_targets.brick_upload_queue.push(BrickUploadRequest {
                            ownership: brick_ownership,
                            min_position: node_bounds.min_position,
                        });
                    }
                }
            };
        }
        NodeContent::Leaf(bricks) => {
            execute_for_relevant_sectants(
                &node_bounds,
                &current_bl,
                current_include_distance as u32,
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
                            let brick_ownership =
                                BrickOwnedBy::NodeAsChild(node_key as u32, target_child_sectant);

                            if viewport_contains_target_fn(
                                &current_bl,
                                current_include_distance,
                                &position_in_target,
                                &update_size_in_target,
                            ) && upload_targets
                                .brick_ownership
                                .get_by_right(&brick_ownership)
                                .is_none()
                            {
                                upload_targets.brick_upload_queue.push(BrickUploadRequest {
                                    ownership: brick_ownership,
                                    min_position: target_bounds.min_position,
                                });
                            }
                        }
                    };
                },
            );
        }
        NodeContent::Internal(_ocbits) => {
            execute_for_relevant_sectants(
                &node_bounds,
                &current_bl,
                current_include_distance as u32,
                |position_in_target,
                 update_size_in_target,
                 target_child_sectant,
                 &target_bounds| {
                    if let Some(child_key) = tree.valid_child_for(node_key, target_child_sectant) {
                        if viewport_contains_target_fn(
                            &current_bl,
                            current_include_distance,
                            &position_in_target,
                            &update_size_in_target,
                        ) {
                            upload_targets.node_upload_queue.push(NodeUploadRequest {
                                node_key: child_key,
                                parent_key: node_key,
                                sectant: target_child_sectant,
                            });
                            upload_targets.nodes_to_see.insert(child_key);
                            add_children_to_upload_queue(
                                (child_key, target_bounds, node_mip_level - 1),
                                tree,
                                viewport_center,
                                view_distance,
                                upload_targets,
                                min_mip_level,
                            );
                        }
                    }
                },
            );
        }
    }
}

pub(crate) fn handle_changes<T: VoxelData>(
    tree_gpu_host: Option<Res<BoxTreeGPUHost<T>>>,
    mut vhx_pipeline: Option<ResMut<VhxRenderPipeline>>,
    mut viewset: Option<ResMut<VhxViewSet>>,
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

    let mut view = viewset.view_mut(0).unwrap();
    let mut updates = vec![];

    #[allow(clippy::reversed_empty_ranges)]
    let mut ocbits_updated = usize::MAX..0;

    'uploading_buffers: {
        // Decide upload targets
        if view.reload {
            rebuild::<T>(
                &tree_host.tree,
                &view.spyglass.viewport.origin.clone(),
                view.spyglass.viewport.frustum.z,
                &mut view.data_handler.upload_targets,
            );

            view.data_handler.upload_state.node_upload_progress = 0;
            view.data_handler.upload_state.brick_upload_progress = 0;

            view.reload = false;
        }

        // Upload targets if view is ready
        if !view.reload {
            let data_handler = &mut view.data_handler;

            debug_assert!(
                data_handler.upload_state.node_upload_progress
                    <= data_handler.upload_targets.node_upload_queue.len()
            );

            // Handle node uploads
            for _ in 0..data_handler.node_uploads_per_frame.min(
                data_handler.upload_targets.node_upload_queue.len()
                    - data_handler.upload_state.node_upload_progress,
            ) {
                // find first node to upload
                while data_handler.upload_state.node_upload_progress
                    < data_handler.upload_targets.node_upload_queue.len()
                {
                    let node_upload_request = data_handler.upload_targets.node_upload_queue
                        [data_handler.upload_state.node_upload_progress]
                        .clone();

                    if let Some(node_meta_index) = data_handler
                        .upload_targets
                        .node_key_vs_meta_index
                        .get_by_left(&node_upload_request.node_key)
                        .cloned()
                    {
                        // Skip to next node if the current node is already uploaded
                        if matches!(
                            tree_host.tree.node_mips[node_upload_request.node_key],
                            BrickData::Parted(_)
                        ) && data_handler.render_data.node_mips[node_meta_index]
                            == empty_marker::<u32>()
                        {
                            // Upload MIP again, if not present already
                            let mip_update = data_handler.add_brick(
                                &tree_host.tree,
                                BrickUploadRequest {
                                    ownership: BrickOwnedBy::NodeAsMIP(
                                        node_upload_request.node_key as u32,
                                    ),
                                    min_position: V3c::default(),
                                },
                            );
                            if mip_update.allocation_failed {
                                // Can't fit new mip brick into buffers, need to rebuild the pipeline
                                re_evaluate_view_size(&mut view);
                                break 'uploading_buffers; // voxel data still needs to be written out
                            }
                            updates.push(mip_update);
                        }

                        data_handler.upload_state.node_upload_progress += 1;
                        continue;
                    }

                    // Upload Node to GPU
                    let (new_node_index, new_node_update) =
                        data_handler.add_node(&tree_host.tree, &node_upload_request);

                    if new_node_update.allocation_failed {
                        // Can't fit new brick into buffers, need to rebuild the pipeline
                        re_evaluate_view_size(&mut view);
                        break 'uploading_buffers; // voxel data still needs to be written out
                    }
                    updates.push(new_node_update);

                    // Upload MIP to GPU
                    let mip_update = data_handler.add_brick(
                        &tree_host.tree,
                        BrickUploadRequest {
                            ownership: BrickOwnedBy::NodeAsMIP(node_upload_request.node_key as u32),
                            min_position: V3c::unit(0.), // min_position not used for MIPs
                        },
                    );

                    if mip_update.allocation_failed {
                        // Can't fit new MIP brick into buffers, need to rebuild the pipeline
                        re_evaluate_view_size(&mut view);
                        break 'uploading_buffers; // voxel data still needs to be written out
                    }
                    updates.push(mip_update);

                    // Also set the ocbits updated range
                    ocbits_updated.start = ocbits_updated.start.min(new_node_index * 2);
                    ocbits_updated.end = ocbits_updated.end.max(new_node_index * 2 + 2);

                    data_handler.upload_state.node_upload_progress += 1;
                    break;
                }
                if data_handler.upload_state.node_upload_progress
                    == data_handler.upload_targets.node_upload_queue.len()
                {
                    // No more nodes to upload!
                    break;
                }
            }

            // Handle brick uploads
            if data_handler.upload_state.brick_upload_progress
                == data_handler.upload_targets.brick_upload_queue.len()
            {
                break 'uploading_buffers; // All bricks are already uploaded
            }
            for _ in 0..data_handler.brick_uploads_per_frame {
                debug_assert!(
                    data_handler.upload_state.brick_upload_progress
                        < data_handler.upload_targets.brick_upload_queue.len()
                );

                // find a brick to upload
                for brick_request_index in data_handler.upload_state.brick_upload_progress
                    ..data_handler.upload_targets.brick_upload_queue.len()
                {
                    let brick_request =
                        data_handler.upload_targets.brick_upload_queue[brick_request_index].clone();
                    let brick_ownership = brick_request.ownership.clone();

                    if
                    // current brick is not uploaded
                    data_handler
                        .upload_targets
                        .brick_ownership
                        .get_by_right(&brick_request.ownership)
                        .is_none()
                        // current brick can be uploaded
                        && match brick_request.ownership {
                            BrickOwnedBy::None => panic!("Request to upload unowned brick?!"),
                            BrickOwnedBy::NodeAsMIP(node_key) |BrickOwnedBy::NodeAsChild(node_key, _) => {
                                // Brick can be uploaded if its parent is already uploaded to the GPU
                                 data_handler.upload_targets.node_key_vs_meta_index.contains_left(&(node_key as usize))
                                 // Brick should be uploaded if its parent also needs to be uploaded to GPU
                                 && data_handler.upload_targets.nodes_to_see.contains(&(node_key as usize))
                            }
                        }
                    {
                        let brick_update = data_handler.add_brick(&tree_host.tree, brick_request);
                        if brick_update.allocation_failed {
                            // Can't fit new brick into buffers, need to rebuild the pipeline
                            re_evaluate_view_size(&mut view);
                            break 'uploading_buffers; // voxel data still needs to be written out
                        }
                        updates.push(brick_update);
                    } else {
                        if brick_request_index
                            == data_handler.upload_targets.brick_upload_queue.len()
                        {
                            break 'uploading_buffers; // Can't upload more bricks this loop
                        }
                        continue;
                    }

                    // In case current brick request is uploaded already, just increase progress
                    debug_assert!(data_handler
                        .upload_targets
                        .brick_ownership
                        .get_by_right(&brick_ownership)
                        .is_some());
                    if brick_request_index == data_handler.upload_state.brick_upload_progress {
                        data_handler.upload_state.brick_upload_progress += 1;
                        if data_handler.upload_state.brick_upload_progress
                            == data_handler.upload_targets.brick_upload_queue.len()
                        {
                            break 'uploading_buffers; // No more bricks to upload
                        }
                    }
                    break;
                }
            }
        }
    }

    if view.resources.is_none() {
        return; // Can't write to buffers as there are not created
    }

    // Apply writes to GPU
    let render_queue = &pipeline.render_queue;

    // Data updates for spyglass viewport
    if view.spyglass.viewport_changed {
        // Update viewport matrices before uploading to GPU
        let resolution = view.resolution;
        view.spyglass.viewport.update_matrices(resolution);
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
    let tree = &tree_host.tree;
    if view.data_handler.render_data.mips_enabled != tree.mip_map_strategy.is_enabled() {
        // Regenerate feature bits
        view.data_handler.render_data.boxtree_meta.tree_properties = boxtree_properties(tree);

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
