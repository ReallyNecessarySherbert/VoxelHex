use crate::{
    boxtree::{
        iterate::execute_for_relevant_sectants,
        types::{BrickData, NodeChildren, NodeContent},
        BoxTree, V3c, V3cf32, VoxelData, BOX_NODE_CHILDREN_COUNT, BOX_NODE_DIMENSION,
    },
    object_pool::empty_marker,
    raytracing::{
        bevy::{
            streaming::{
                re_evaluate_view_size,
                types::{
                    BrickOwnedBy, BrickOwnership, CacheUpdatePackage, UploadQueueTargets,
                    UploadQueueUpdateTask,
                },
            },
            types::BoxTreeGPUHost,
        },
        BoxTreeGPUView,
    },
    spatial::{raytracing::viewport_contains_target, Cube},
};
use bendy::{decoding::FromBencode, encoding::ToBencode};
use bevy::{
    ecs::system::ResMut,
    prelude::Commands,
    tasks::{block_on, futures_lite::future, AsyncComputeTaskPool},
};
use bimap::BiHashMap;
use std::{
    collections::HashSet,
    hash::Hash,
    sync::{Arc, RwLock},
};

impl UploadQueueTargets {
    pub(crate) fn reset(&mut self) {
        self.brick_ownership
            .write()
            .expect("Expected to be able to clear brick ownersip entries")
            .clear();
        self.node_key_vs_meta_index.clear();
        self.node_index_vs_parent.clear();
        self.nodes_to_see
            .write()
            .expect("Expected to be able to reset list of nodes in view!")
            .clear();
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
    nodes_to_see: Arc<RwLock<HashSet<usize>>>,
) {
    nodes_to_see
        .write()
        .expect("Expected to be able to reset list of nodes in view!")
        .clear();

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
    let deepest_mip_level_to_upload = (((viewport_bl_ - viewport_bl).length() / view_distance)
        .ceil() as u32)
        .min(max_mip_level)
        .max(1);

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
    nodes_to_see
        .write()
        .expect("Expected to be able to update list of nodes in view!")
        .insert(*center_node_key);

    // add center node together with children inside the viewport into the queue
    add_children_to_upload_queue(
        center_node_parent.unwrap_or((*center_node_key, node_bounds, node_mip_level)),
        tree,
        &viewport_center,
        view_distance,
        deepest_mip_level_to_upload,
        nodes_to_see,
        brick_ownership,
    );
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
    nodes_to_see: Arc<RwLock<HashSet<usize>>>,
    brick_ownership: BrickOwnership,
) {
    debug_assert!(min_mip_level >= 1);
    if node_mip_level < min_mip_level {
        return;
    }

    debug_assert!(
        nodes_to_see
            .read()
            .expect("Expected to be able to read list of nodes in view!")
            .contains(&node_key),
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
                            nodes_to_see
                                .write()
                                .expect("Expected to be able to update list of nodes in view!")
                                .insert(child_key);
                            add_children_to_upload_queue(
                                (child_key, target_bounds, node_mip_level - 1),
                                tree,
                                viewport_center,
                                view_distance,
                                min_mip_level,
                                nodes_to_see.clone(),
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
//  ███████████  ███████████      ███████      █████████  ██████████  █████████   █████████
// ░░███░░░░░███░░███░░░░░███   ███░░░░░███   ███░░░░░███░░███░░░░░█ ███░░░░░███ ███░░░░░███
//  ░███    ░███ ░███    ░███  ███     ░░███ ███     ░░░  ░███  █ ░ ░███    ░░░ ░███    ░░░
//  ░██████████  ░██████████  ░███      ░███░███          ░██████   ░░█████████ ░░█████████
//  ░███░░░░░░   ░███░░░░░███ ░███      ░███░███          ░███░░█    ░░░░░░░░███ ░░░░░░░░███
//  ░███         ░███    ░███ ░░███     ███ ░░███     ███ ░███ ░   █ ███    ░███ ███    ░███
//  █████        █████   █████ ░░░███████░   ░░█████████  ██████████░░█████████ ░░█████████
// ░░░░░        ░░░░░   ░░░░░    ░░░░░░░      ░░░░░░░░░  ░░░░░░░░░░  ░░░░░░░░░   ░░░░░░░░░
//##############################################################################
/// Recreate the list of nodes currently required to be inside the viewport
pub(crate) fn process<
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
    commands: &mut Commands,
    tree: &'a BoxTree<T>,
    tree_host: &BoxTreeGPUHost<T>,
    view: &mut BoxTreeGPUView,
    mut upload_queue_update: Option<ResMut<UploadQueueUpdateTask>>,
) -> Vec<CacheUpdatePackage<'a>> {
    let mut cache_updates = vec![];
    let view_distance = view.spyglass.viewport.frustum.z;

    if view.reload {
        // rebuild upload queue if not already in progress
        if upload_queue_update.is_none() {
            let thread_pool = AsyncComputeTaskPool::get();
            let viewport_center = view.spyglass.viewport.origin;
            let brick_ownership = view.data_handler.upload_targets.brick_ownership.clone();
            let tree_arc = tree_host.tree.clone();
            let nodes_to_see = view.data_handler.upload_targets.nodes_to_see.clone();
            commands.insert_resource(UploadQueueUpdateTask(thread_pool.spawn(async move {
                rebuild::<T>(
                    &tree_arc
                        .read()
                        .expect("Expected to be able to read tree from GPU host"),
                    viewport_center,
                    view_distance,
                    brick_ownership,
                    nodes_to_see,
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
        let new_node_update = view.data_handler.add_node(
            tree,
            BoxTree::<T>::ROOT_NODE_KEY as usize,
            BOX_NODE_CHILDREN_COUNT as u8,
        );
        debug_assert_eq!(
            0,
            new_node_update
                .added_node
                .expect("Expected update package to contain an added node")
        );
        cache_updates.push(new_node_update);
        let mip_update = view
            .data_handler
            .add_brick(tree, BrickOwnedBy::NodeAsMIP(BoxTree::<T>::ROOT_NODE_KEY));
        cache_updates.push(mip_update);

        view.reload = false;
    }

    // If the upload queue update task is finished apply it!
    if let Some(ref mut upload_queue_update) = upload_queue_update {
        if block_on(future::poll_once(&mut upload_queue_update.0)).is_some() {
            commands.remove_resource::<UploadQueueUpdateTask>();
            view.data_handler.upload_state.target_node_stack = vec![(
                BoxTree::<T>::ROOT_NODE_KEY as usize,
                0,
                Cube::root_bounds(tree.get_size() as f32),
            )];
        }
    }

    // Initiate pending update opertaion if there's any, but continue with uploads
    if let Some((viewport_center, viewing_distance)) = view.data_handler.pending_upload_queue_update
    {
        let thread_pool = AsyncComputeTaskPool::get();
        let brick_ownership = view.data_handler.upload_targets.brick_ownership.clone();
        let tree_arc = tree_host.tree.clone();
        let nodes_to_see = view.data_handler.upload_targets.nodes_to_see.clone();

        commands.insert_resource(UploadQueueUpdateTask(thread_pool.spawn(async move {
            rebuild::<T>(
                &tree_arc
                    .read()
                    .expect("Expected to be able to read tree from GPU host"),
                viewport_center,
                viewing_distance,
                brick_ownership,
                nodes_to_see,
            )
        })));
        view.data_handler.pending_upload_queue_update = None;
    }

    // Decide on targets to upload this loop
    let data_handler = &mut view.data_handler;

    // Handle node uploads, if there are any
    'node_uploads: {
        if data_handler.upload_state.target_node_stack.is_empty() {
            break 'node_uploads;
        }

        for _ in 0..data_handler.node_uploads_per_frame {
            // Get next node to check
            let Some((parent_key, target_sectant, node_key, node_bounds)) = next_valid_node(
                tree,
                &mut data_handler.upload_state.target_node_stack,
                &data_handler
                    .upload_targets
                    .nodes_to_see
                    .read()
                    .expect("Expected to be able to read list of nodes in view!"),
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
                    data_handler.add_brick(tree, BrickOwnedBy::NodeAsMIP(node_key as u32));
                if mip_update.allocation_failed {
                    // Can't fit new mip brick into buffers, need to rebuild the pipeline
                    re_evaluate_view_size(view);
                    return cache_updates; // voxel data still needs to be written out
                }
                cache_updates.push(mip_update);
            } else {
                // Upload Selected Node to GPU
                let new_node_update = data_handler.add_node(tree, parent_key, target_sectant);

                if new_node_update.allocation_failed {
                    // Can't fit new brick into buffers, need to rebuild the pipeline
                    re_evaluate_view_size(view);
                    return cache_updates; // voxel data still needs to be written out
                }
                cache_updates.push(new_node_update);

                // Upload MIP to GPU
                let mip_update =
                    data_handler.add_brick(tree, BrickOwnedBy::NodeAsMIP(node_key as u32));

                if mip_update.allocation_failed {
                    // Can't fit new MIP brick into buffers, need to rebuild the pipeline
                    re_evaluate_view_size(view);
                    return cache_updates; // voxel data still needs to be written out
                }
                cache_updates.push(mip_update);
            }

            // Push the children into the brick upload list
            data_handler
                .upload_state
                .bricks_to_upload
                .append(&mut process_node_children(
                    tree,
                    node_key,
                    &node_bounds,
                    &V3c::<u32>::from(
                        view.spyglass.viewport.origin - V3c::unit(view_distance / 2.),
                    ),
                    view_distance,
                    &data_handler
                        .upload_targets
                        .brick_ownership
                        .read()
                        .expect("Expected to be able to read brick ownership entries"),
                ));
        }
    }

    // upload bricks from the upload list if there is any
    let data_handler = &mut view.data_handler;
    let brick_requests = if data_handler.upload_state.bricks_to_upload.is_empty() {
        vec![]
    } else {
        data_handler
            .upload_state
            .bricks_to_upload
            .drain(
                (data_handler
                    .upload_state
                    .bricks_to_upload
                    .len()
                    .saturating_sub(data_handler.brick_uploads_per_frame))..,
            )
            .collect::<Vec<_>>()
    };
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

        let brick_update = data_handler.add_brick(tree, brick_request.clone());
        if brick_update.allocation_failed {
            // Can't fit new brick brick into buffers, need to rebuild the pipeline
            re_evaluate_view_size(view);
            return cache_updates; // voxel data still needs to be written out
        }
        cache_updates.push(brick_update);
    }

    cache_updates
}

fn process_node_children<
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
    node_key: usize,
    node_bounds: &Cube,
    viewport_bl: &V3c<u32>,
    view_distance: f32,
    brick_ownership: &BiHashMap<usize, BrickOwnedBy>,
) -> Vec<BrickOwnedBy> {
    let mut result = vec![];
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
                        viewport_bl,
                        view_distance,
                        &V3c::from(node_bounds.min_position),
                        &V3c::unit(node_bounds.size as u32),
                    ) && !brick_ownership.contains_right(&brick_ownership_entry)
                    {
                        result.push(brick_ownership_entry);
                    }
                }
            };
        }
        NodeContent::Leaf(bricks) => {
            execute_for_relevant_sectants(
                node_bounds,
                viewport_bl,
                view_distance as u32,
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
                                viewport_bl,
                                view_distance,
                                &position_in_target,
                                &update_size_in_target,
                            ) && brick_ownership
                                .get_by_right(&brick_ownership_entry)
                                .is_none()
                            {
                                result.push(brick_ownership_entry);
                            }
                        }
                    };
                },
            );
        }
    }
    result
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
    let (current_node_key, mut target_sectant, current_node_bounds) = node_stack.last().cloned()?;
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
