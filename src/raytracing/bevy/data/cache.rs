use crate::{
    boxtree::{
        BOX_NODE_CHILDREN_COUNT, BoxTree, OOB_SECTANT, V3c, VoxelData,
        types::{BrickData, NodeContent},
    },
    object_pool::empty_marker,
    raytracing::bevy::types::{
        BoxTreeGPUDataHandler, BrickOwnedBy, BrickUpdate, BrickUploadRequest, CacheUpdatePackage,
        NodeUploadRequest,
    },
};
use std::hash::Hash;

impl BoxTreeGPUDataHandler {
    //##############################################################################
    //  ██████████     █████████   ███████████   █████████
    // ░░███░░░░███   ███░░░░░███ ░█░░░███░░░█  ███░░░░░███
    //  ░███   ░░███ ░███    ░███ ░   ░███  ░  ░███    ░███
    //  ░███    ░███ ░███████████     ░███     ░███████████
    //  ░███    ░███ ░███░░░░░███     ░███     ░███░░░░░███
    //  ░███    ███  ░███    ░███     ░███     ░███    ░███
    //  ██████████   █████   █████    █████    █████   █████
    // ░░░░░░░░░░   ░░░░░   ░░░░░    ░░░░░    ░░░░░   ░░░░░

    //  ██████████   ██████████  █████████  █████   █████████  ██████   █████
    // ░░███░░░░███ ░░███░░░░░█ ███░░░░░███░░███   ███░░░░░███░░██████ ░░███
    //  ░███   ░░███ ░███  █ ░ ░███    ░░░  ░███  ███     ░░░  ░███░███ ░███
    //  ░███    ░███ ░██████   ░░█████████  ░███ ░███          ░███░░███░███
    //  ░███    ░███ ░███░░█    ░░░░░░░░███ ░███ ░███    █████ ░███ ░░██████
    //  ░███    ███  ░███ ░   █ ███    ░███ ░███ ░░███  ░░███  ░███  ░░█████
    //  ██████████   ██████████░░█████████  █████ ░░█████████  █████  ░░█████
    // ░░░░░░░░░░   ░░░░░░░░░░  ░░░░░░░░░  ░░░░░   ░░░░░░░░░  ░░░░░    ░░░░░
    //##############################################################################
    /// Creates the descriptor bytes for the given node
    fn inject_node_properties<T>(
        meta_array: &mut [u32],
        node_index: usize,
        tree: &BoxTree<T>,
        node_key: usize,
    ) where
        T: Default + Clone + Eq + VoxelData + Hash,
    {
        // set node type
        match tree.nodes.get(node_key) {
            NodeContent::Internal(_) | NodeContent::Nothing => {
                meta_array[node_index / 8] &= !(0x01 << (node_index % 8));
                meta_array[node_index / 8] &= !(0x01 << (8 + (node_index % 8)));
            }
            NodeContent::Leaf(_bricks) => {
                meta_array[node_index / 8] |= 0x01 << (node_index % 8);
                meta_array[node_index / 8] &= !(0x01 << (8 + (node_index % 8)));
            }
            NodeContent::UniformLeaf(_brick) => {
                meta_array[node_index / 8] |= 0x01 << (node_index % 8);
                meta_array[node_index / 8] |= 0x01 << (8 + (node_index % 8));
            }
        };
    }

    //##############################################################################
    //  ██████████ ███████████     █████████    █████████  ██████████
    // ░░███░░░░░█░░███░░░░░███   ███░░░░░███  ███░░░░░███░░███░░░░░█
    //  ░███  █ ░  ░███    ░███  ░███    ░███ ░███    ░░░  ░███  █ ░
    //  ░██████    ░█████████    ░███████████ ░░█████████  ░██████
    //  ░███░░█    ░███░░░░░███  ░███░░░░░███  ░░░░░░░░███ ░███░░█
    //  ░███ ░   █ ░███    ░███  ░███    ░███  ███    ░███ ░███ ░   █
    //  ██████████ █████   █████ █████   █████░░█████████  ██████████
    // ░░░░░░░░░░ ░░░░░   ░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░
    //  ██████   █████    ███████    ██████████   ██████████
    // ░░██████ ░░███   ███░░░░░███ ░░███░░░░███ ░░███░░░░░█
    //  ░███░███ ░███  ███     ░░███ ░███   ░░███ ░███  █ ░
    //  ░███░░███░███ ░███      ░███ ░███    ░███ ░██████
    //  ░███ ░░██████ ░███      ░███ ░███    ░███ ░███░░█
    //  ░███  ░░█████ ░░███     ███  ░███    ███  ░███ ░   █
    //  █████  ░░█████ ░░░███████░   ██████████   ██████████
    // ░░░░░    ░░░░░    ░░░░░░░    ░░░░░░░░░░   ░░░░░░░░░░
    //    █████████  █████   █████ █████ █████       ██████████
    //   ███░░░░░███░░███   ░░███ ░░███ ░░███       ░░███░░░░███
    //  ███     ░░░  ░███    ░███  ░███  ░███        ░███   ░░███
    // ░███          ░███████████  ░███  ░███        ░███    ░███
    // ░███          ░███░░░░░███  ░███  ░███        ░███    ░███
    // ░░███     ███ ░███    ░███  ░███  ░███      █ ░███    ███
    //  ░░█████████  █████   █████ █████ ███████████ ██████████
    //   ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░ ░░░░░░░░░░░ ░░░░░░░░░░
    //##############################################################################
    /// Erases the child node pointed by the given victim pointer
    /// returns with the vector of nodes modified
    fn erase_node_child<T>(
        &mut self,
        meta_index: usize,
        child_sectant: usize,
        tree: &BoxTree<T>,
    ) -> Vec<usize>
    where
        T: Default + Clone + Eq + VoxelData + Hash,
    {
        let mut modified_nodes = vec![meta_index];
        debug_assert!(
            self.upload_targets
                .node_key_vs_meta_index
                .contains_right(&meta_index),
            "Expected parent node to be in metadata index hash! (meta: {meta_index})"
        );
        let parent_key = self
            .upload_targets
            .node_key_vs_meta_index
            .get_by_right(&meta_index)
            .unwrap();

        debug_assert!(
            tree.nodes.key_is_valid(*parent_key),
            "Expected parent node({:?}) to be valid",
            parent_key
        );

        // Erase connection to parent
        let parent_first_child_index = meta_index * BOX_NODE_CHILDREN_COUNT;
        let parent_children_offset = parent_first_child_index + child_sectant;
        let child_descriptor = self.render_data.node_children[parent_children_offset] as usize;
        debug_assert_ne!(
            child_descriptor,
            empty_marker::<u32>() as usize,
            "Expected erased child[{}] of node[{}] meta[{}] to be an erasable node/brick",
            child_sectant,
            parent_key,
            meta_index
        );
        self.upload_targets
            .node_index_vs_parent
            .remove(&child_descriptor);
        match tree.nodes.get(*parent_key) {
            NodeContent::Nothing => {
                panic!("HOW DO I ERASE NOTHING. AMERICA EXPLAIN")
            }
            NodeContent::Internal(_) | NodeContent::Leaf(_) | NodeContent::UniformLeaf(_) => {
                self.render_data.node_children[parent_children_offset] = empty_marker::<u32>();
            }
        }

        match tree.nodes.get(*parent_key) {
            NodeContent::Nothing => {
                panic!("HOW DO I ERASE NOTHING. AMERICA EXPLAIN")
            }
            NodeContent::Internal(_occupied_bits) => {
                debug_assert!(
                    self.upload_targets
                        .node_key_vs_meta_index
                        .contains_right(&child_descriptor),
                    "Expected erased child node index[{child_descriptor}] to be in metadata index hash!"
                );
                let child_key = self
                    .upload_targets
                    .node_key_vs_meta_index
                    .get_by_right(&child_descriptor)
                    .unwrap();
                debug_assert!(
                    tree.nodes.key_is_valid(*child_key),
                    "Expected erased child node({child_key}) to be valid"
                );

                // Erase MIP connection, Erase ownership as well
                let child_mip = self.render_data.node_mips[child_descriptor];
                if child_mip != empty_marker::<u32>() {
                    self.render_data.node_mips[child_descriptor] = empty_marker();
                    if matches!(tree.node_mips[*child_key], BrickData::Parted(_)) {
                        self.upload_targets
                            .brick_ownership
                            .remove_by_left(&(child_mip as usize));
                    }
                }
                modified_nodes.push(child_descriptor);
            }
            NodeContent::UniformLeaf(_) | NodeContent::Leaf(_) => {
                let brick_index = child_descriptor & 0x7FFFFFFF;
                debug_assert!(
                    (0 == child_sectant)
                        || matches!(tree.nodes.get(*parent_key), NodeContent::Leaf(_)),
                    "Expected child sectant in uniform leaf to be 0 in: {:?}",
                    (meta_index, child_sectant)
                );
                if child_descriptor != empty_marker::<u32>() as usize {
                    self.upload_targets
                        .brick_ownership
                        .remove_by_left(&{ brick_index });
                }
            }
        }
        modified_nodes
    }

    //##############################################################################
    //  ██████   █████    ███████    ██████████   ██████████
    // ░░██████ ░░███   ███░░░░░███ ░░███░░░░███ ░░███░░░░░█
    //  ░███░███ ░███  ███     ░░███ ░███   ░░███ ░███  █ ░
    //  ░███░░███░███ ░███      ░███ ░███    ░███ ░██████
    //  ░███ ░░██████ ░███      ░███ ░███    ░███ ░███░░█
    //  ░███  ░░█████ ░░███     ███  ░███    ███  ░███ ░   █
    //  █████  ░░█████ ░░░███████░   ██████████   ██████████
    // ░░░░░    ░░░░░    ░░░░░░░    ░░░░░░░░░░   ░░░░░░░░░░
    //##############################################################################

    /// Provides the first available index in the metadata buffer which can be overwritten
    /// Optionally the source where the child can be taken from
    /// May fail to provide index, in case insufficient space
    fn first_available_node(&mut self) -> Option<(usize, Option<(usize, u8)>)> {
        // Iterate the buffer until either a node is found or the victim node loops back to itself
        let mut victim_node_index = (self.upload_state.victim_node + 1) % self.nodes_in_view;
        while victim_node_index != self.upload_state.victim_node {
            // query if the node key of the potential node victim
            let victim_node_key = self
                .upload_targets
                .node_key_vs_meta_index
                .get_by_right(&victim_node_index);

            if victim_node_key.is_none()
                || !self
                    .upload_targets
                    .nodes_to_see
                    .contains(victim_node_key.unwrap())
            {
                // Victim node is not in the node upload queue, it can be overwritten!
                self.upload_state.victim_node = victim_node_index;
                return Some((
                    victim_node_index,
                    self.upload_targets
                        .node_index_vs_parent
                        .get(&victim_node_index)
                        .copied(),
                ));
            }

            victim_node_index = (victim_node_index + 1) % self.nodes_in_view;
        }
        // Unable to select a single node to overwrite
        None
    }

    /// Writes most of the data of the given node to the first available index
    /// Writes: metadata, available child information, occupied bits and parent connections
    /// It will try to collecty MIP information if still available, but will not upload a MIP
    /// * `returns` - Returns the meta index of the added node, the modified nodes and bricks updates for the insertion
    pub(crate) fn add_node<'a, T: VoxelData>(
        &mut self,
        tree: &'a BoxTree<T>,
        node_upload_request: &NodeUploadRequest,
    ) -> (usize, CacheUpdatePackage<'a>) {
        let node_key = node_upload_request.node_key;
        let mut modifications = CacheUpdatePackage {
            allocation_failed: false,
            brick_update: None,
            modified_nodes: vec![],
        };

        // Determine the new node index in meta
        debug_assert!(
            !self
                .upload_targets
                .node_key_vs_meta_index
                .contains_left(&node_key)
                || BoxTree::<T>::ROOT_NODE_KEY == node_key as u32,
            "Trying to add already available node twice!"
        );
        let (node_index, robbed_parent) = if BoxTree::<T>::ROOT_NODE_KEY == node_key as u32 {
            (0, None)
        } else {
            let Some((node_index, robbed_parent)) = self.first_available_node() else {
                modifications.allocation_failed = true;
                return (0, modifications);
            };
            (node_index, robbed_parent)
        };

        let robbed_node_key_in_meta = self
            .upload_targets
            .node_key_vs_meta_index
            .get_by_right(&node_index)
            .cloned();
        self.upload_targets
            .node_key_vs_meta_index
            .insert(node_key, node_index);

        if modifications.allocation_failed {
            // allocation failed! Can't upload node after all, undo ownership update
            if let Some(robbed_node_key_in_meta) = robbed_node_key_in_meta {
                self.upload_targets
                    .node_key_vs_meta_index
                    .insert(robbed_node_key_in_meta, node_index);
            } else {
                self.upload_targets
                    .node_key_vs_meta_index
                    .remove_by_right(&node_index);
            }
            return (0, modifications);
        }

        // overwrite a currently present node if needed
        if let Some(robbed_parent) = robbed_parent {
            debug_assert_eq!(
                (self.render_data.node_children
                    [robbed_parent.0 * BOX_NODE_CHILDREN_COUNT + robbed_parent.1 as usize])
                    as usize,
                node_index,
                "Expected child[{:?}] of node[{:?}] to be node[{:?}] instead of {:?}*!",
                robbed_parent.1,
                robbed_parent.0,
                node_index,
                self.render_data.node_children
                    [robbed_parent.0 * BOX_NODE_CHILDREN_COUNT + robbed_parent.1 as usize]
            );
            modifications
                .modified_nodes
                .append(&mut self.erase_node_child(
                    robbed_parent.0,
                    robbed_parent.1 as usize,
                    tree,
                ));
        } else {
            modifications.modified_nodes.push(node_index);
        };

        // Inject Node properties to render data
        Self::inject_node_properties(
            &mut self.render_data.node_metadata,
            node_index,
            tree,
            node_key,
        );

        // Update occupancy in ocbits
        let occupied_bits = tree.stored_occupied_bits(node_key);
        self.render_data.node_ocbits[node_index * 2] = (occupied_bits & 0x00000000FFFFFFFF) as u32;
        self.render_data.node_ocbits[node_index * 2 + 1] =
            ((occupied_bits & 0xFFFFFFFF00000000) >> 32) as u32;

        // Add empty children
        let child_children_offset = node_index * BOX_NODE_CHILDREN_COUNT;
        self.render_data.node_children.splice(
            (child_children_offset)..(child_children_offset + BOX_NODE_CHILDREN_COUNT),
            vec![empty_marker::<u32>(); BOX_NODE_CHILDREN_COUNT],
        );

        // Set parent conection
        debug_assert!(
            self.upload_targets
                .node_key_vs_meta_index
                .contains_left(&node_upload_request.parent_key),
            "Expected node parent to be in GPU render data at the time of upload"
        );
        if BoxTree::<T>::ROOT_NODE_KEY as usize != node_key {
            let parent_meta_index = self
                .upload_targets
                .node_key_vs_meta_index
                .get_by_left(&node_upload_request.parent_key)
                .unwrap();
            let parent_child_index = (parent_meta_index * BOX_NODE_CHILDREN_COUNT)
                + node_upload_request.sectant as usize;
            self.render_data.node_children[parent_child_index] = node_index as u32;
            self.upload_targets.node_index_vs_parent.insert(
                node_index,
                (*parent_meta_index, node_upload_request.sectant),
            );
            modifications.modified_nodes.push(*parent_meta_index);
        }

        // Add child nodes of new child if any is available
        let parent_first_child_index = node_index * BOX_NODE_CHILDREN_COUNT;
        match tree.nodes.get(node_key) {
            NodeContent::Nothing => {}
            NodeContent::Internal(_) => {
                for sectant in 0..BOX_NODE_CHILDREN_COUNT {
                    let child_key = tree.node_children[node_key].child(sectant as u8);
                    if child_key != empty_marker::<u32>() as usize {
                        self.render_data.node_children[parent_first_child_index + sectant] = *self
                            .upload_targets
                            .node_key_vs_meta_index
                            .get_by_left(&child_key)
                            .unwrap_or(&(empty_marker::<u32>() as usize))
                            as u32;
                    } else {
                        self.render_data.node_children[parent_first_child_index + sectant] =
                            empty_marker::<u32>();
                    }
                }
            }
            NodeContent::UniformLeaf(brick) => {
                if let BrickData::Solid(voxel) = brick {
                    self.render_data.node_children[parent_first_child_index] = 0x80000000 | *voxel;
                } else {
                    self.render_data.node_children[parent_first_child_index] =
                        empty_marker::<u32>();
                }
            }
            NodeContent::Leaf(bricks) => {
                for (sectant, brick) in bricks.iter().enumerate().take(BOX_NODE_CHILDREN_COUNT) {
                    if let BrickData::Solid(voxel) = brick {
                        self.render_data.node_children[parent_first_child_index + sectant] =
                            0x80000000 | voxel;
                    } else {
                        let node_entry = BrickOwnedBy::NodeAsChild(node_key as u32, sectant as u8);
                        let brick_ownership = self
                            .upload_targets
                            .brick_ownership
                            .get_by_right(&node_entry);
                        if let Some(brick_index) = brick_ownership {
                            self.render_data.node_children[parent_first_child_index + sectant] =
                                0x7FFFFFFF & *brick_index as u32;
                        } else {
                            self.render_data.node_children[parent_first_child_index + sectant] =
                                empty_marker::<u32>();
                        }
                    }
                }
            }
        }

        // Try to collect node MIP entry
        self.render_data.node_mips[node_index] = match tree.node_mips[node_key] {
            BrickData::Empty => empty_marker::<u32>(), // empty MIPS are stored with empty_marker
            BrickData::Solid(voxel) => 0x80000000 | voxel, // In case MIP is solid, it is pointing to the color palette
            BrickData::Parted(_) => {
                // Try to add MIP if it's parted, and not already available
                if let Some(brick_index) = self
                    .upload_targets
                    .brick_ownership
                    .get_by_right(&BrickOwnedBy::NodeAsMIP(node_key as u32))
                {
                    0x7FFFFFFF & *brick_index as u32
                } else {
                    empty_marker()
                }
            }
        };
        (node_index, modifications)
    }

    //##############################################################################
    //  ███████████  ███████████   █████   █████████  █████   ████
    // ░░███░░░░░███░░███░░░░░███ ░░███   ███░░░░░███░░███   ███░
    //  ░███    ░███ ░███    ░███  ░███  ███     ░░░  ░███  ███
    //  ░██████████  ░██████████   ░███ ░███          ░███████
    //  ░███░░░░░███ ░███░░░░░███  ░███ ░███          ░███░░███
    //  ░███    ░███ ░███    ░███  ░███ ░░███     ███ ░███ ░░███
    //  ███████████  █████   █████ █████ ░░█████████  █████ ░░████
    // ░░░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░░░░░  ░░░░░   ░░░░
    //##############################################################################
    /// Provides the index of the first brick available to be overwritten, through the second chance algorithm
    /// * `returns` - The index of the first erasable brick inside the cache and the range of bricks updated
    fn first_available_brick(&mut self, brick_size: f32) -> Option<usize> {
        let brick_outside_range_fn = |brick_index: usize, brick_size: f32| -> bool {
            let brick_bl = &self.upload_targets.brick_positions[brick_index];
            (brick_bl.x + brick_size) < self.upload_range.min_position.x
                || (self.upload_range.min_position.x + self.upload_range.size) < brick_bl.x
                || (brick_bl.y + brick_size) < self.upload_range.min_position.y
                || (self.upload_range.min_position.y + self.upload_range.size) < brick_bl.y
                || (brick_bl.z + brick_size) < self.upload_range.min_position.z
                || (self.upload_range.min_position.z + self.upload_range.size) < brick_bl.z
        };
        let victim_brick = &mut self.upload_state.victim_brick;
        let victim_search_start =
            (*victim_brick as i32 - self.brick_unload_search_perimeter as i32 / 2).max(0) as usize;
        let victim_search_end =
            (victim_search_start + self.brick_unload_search_perimeter).min(self.bricks_in_view);
        let mut priority_victim = None;
        let mut furthest_victim = None;
        let mut furthest_victim_distance = 0.;
        for victim_brick_index in victim_search_start..victim_search_end {
            let brick_ownership = self
                .upload_targets
                .brick_ownership
                .get_by_left(&victim_brick_index)
                .unwrap_or(&BrickOwnedBy::None)
                .clone();
            match brick_ownership {
                BrickOwnedBy::None => {
                    // Unused brick slots have priority over far away bricks
                    priority_victim = Some(victim_brick_index);
                    break;
                }
                BrickOwnedBy::NodeAsMIP(node_key) => {
                    debug_assert!(
                        self.upload_targets
                            .node_key_vs_meta_index
                            .contains_left(&(node_key as usize))
                    );
                    if
                        // in case the node is not inside the node upload list
                        !self.upload_targets.nodes_to_see.contains(&(node_key as usize))
                        // and the node have no children as bricks
                        && self.render_data.node_children.iter()
                            .skip(
                                self.upload_targets
                                    .node_key_vs_meta_index
                                    .get_by_left(&(node_key as usize))
                                    .expect("Expected node_key in brick ownership to be available inside the GPU")
                                    * BOX_NODE_CHILDREN_COUNT,
                            )
                            .take(BOX_NODE_CHILDREN_COUNT)
                            .all(|v| *v == empty_marker::<u32>())
                    {
                       // MIP can be discarded
                        priority_victim = Some(victim_brick_index);
                        break;
                    }
                }
                BrickOwnedBy::NodeAsChild(_node_key, _child_sectant) => {
                    let brick_bl = &self.upload_targets.brick_positions[victim_brick_index];
                    // in case the brick position is outside of the view distance, the brick can be overwritten
                    if brick_outside_range_fn(victim_brick_index, brick_size)
                        && (furthest_victim.is_none()
                            || (*brick_bl - self.upload_range.min_position
                                + V3c::unit(self.upload_range.size / 2.))
                            .length()
                                > furthest_victim_distance)
                    {
                        furthest_victim_distance = (*brick_bl - self.upload_range.min_position
                            + V3c::unit(self.upload_range.size / 2.))
                        .length();
                        furthest_victim = Some(victim_brick_index);
                    }
                }
            }
        }

        if let Some(result_brick_index) = priority_victim {
            *victim_brick = (result_brick_index + 1) % (self.bricks_in_view);
            return Some(result_brick_index);
        }

        if let Some(result_brick_index) = furthest_victim {
            *victim_brick = (result_brick_index + 1) % (self.bricks_in_view);
            return Some(result_brick_index);
        }

        // if there are no bricks within the range that can be safely overwritten, find one that is
        for check_range in [
            victim_search_end..self.bricks_in_view,
            0..victim_search_start,
        ]
        .into_iter()
        {
            for victim_brick_index in check_range {
                let brick_ownership = self
                    .upload_targets
                    .brick_ownership
                    .get_by_left(&victim_brick_index)
                    .unwrap_or(&BrickOwnedBy::None)
                    .clone();
                match brick_ownership {
                    BrickOwnedBy::None => {
                        // Unused brick slots have priority over far away bricks
                        *victim_brick = (victim_brick_index + 1) % (self.bricks_in_view);
                        return Some(victim_brick_index);
                    }
                    BrickOwnedBy::NodeAsMIP(node_key) => {
                        // in case the node is not inside the node upload list, MIP can be erased
                        if !self
                            .upload_targets
                            .nodes_to_see
                            .contains(&(node_key as usize))
                        {
                            *victim_brick = (victim_brick_index + 1) % (self.bricks_in_view);
                            return Some(victim_brick_index);
                        }
                    }
                    BrickOwnedBy::NodeAsChild(_node_key, _child_sectant) => {
                        if brick_outside_range_fn(victim_brick_index, brick_size) {
                            *victim_brick = (victim_brick_index + 1) % (self.bricks_in_view);
                            return Some(victim_brick_index);
                        }
                    }
                }
            }
        }
        None
    }

    /// Makes space for the requested brick and updates brick ownership if needed
    /// * `tree` - The boxtree where the brick is found
    /// * `node_key` - The key for the requested leaf node, whoose child needs to be uploaded
    /// * `target_sectant` - The sectant where the target brick lies
    /// * `returns` - brick updates applied and nodes updated during insertion
    pub(crate) fn add_brick<'a, T>(
        &mut self,
        tree: &'a BoxTree<T>,
        brick_request: BrickUploadRequest,
    ) -> CacheUpdatePackage<'a>
    where
        T: Default + Clone + Eq + Send + Sync + Hash + VoxelData + 'static,
    {
        let Some(brick_index) = self.first_available_brick(tree.brick_dim as f32) else {
            return CacheUpdatePackage {
                allocation_failed: true,
                brick_update: None,
                modified_nodes: vec![],
            };
        };

        let (brick, parent_node_key, target_sectant) = match brick_request.ownership {
            BrickOwnedBy::None => panic!("requesting brick upload with 'no ownership' for brick "),
            BrickOwnedBy::NodeAsChild(node_key, child_sectant) => {
                match tree.nodes.get(node_key as usize) {
                    NodeContent::UniformLeaf(brick) => {
                        debug_assert_eq!(
                            child_sectant, 0,
                            "Expected child of UniformLeaf to be requested as sectant 0!"
                        );
                        (brick, node_key as usize, child_sectant as usize)
                    }
                    NodeContent::Leaf(bricks) => (
                        &bricks[child_sectant as usize],
                        node_key as usize,
                        child_sectant as usize,
                    ),
                    NodeContent::Nothing | NodeContent::Internal(_) => {
                        unreachable!("Shouldn't add brick from Internal or empty node!")
                    }
                }
            }
            BrickOwnedBy::NodeAsMIP(node_key) => (
                &tree.node_mips[node_key as usize],
                node_key as usize,
                OOB_SECTANT as usize,
            ),
        };

        match brick {
            BrickData::Empty => CacheUpdatePackage::default(),
            BrickData::Solid(_voxel) => unreachable!("Shouldn't try to upload solid bricks"),
            BrickData::Parted(brick) => {
                let mut modified_nodes = match *self
                    .upload_targets
                    .brick_ownership
                    .get_by_left(&brick_index)
                    .unwrap_or(&BrickOwnedBy::None)
                {
                    BrickOwnedBy::NodeAsChild(key, sectant) => {
                        if self
                            .upload_targets
                            .node_key_vs_meta_index
                            .get_by_left(&(key as usize))
                            .is_some()
                        {
                            self.erase_node_child(
                                *self
                                    .upload_targets
                                    .node_key_vs_meta_index
                                    .get_by_left(&(key as usize))
                                    .unwrap(),
                                sectant as usize,
                                tree,
                            )
                        } else {
                            Vec::new()
                        }
                    }
                    BrickOwnedBy::NodeAsMIP(key) => {
                        // erase MIP from node if present
                        if self
                            .upload_targets
                            .node_key_vs_meta_index
                            .get_by_left(&(key as usize))
                            .is_some()
                        {
                            let robbed_meta_index = *self
                                .upload_targets
                                .node_key_vs_meta_index
                                .get_by_left(&(key as usize))
                                .unwrap();
                            self.render_data.node_mips[robbed_meta_index] = empty_marker();
                            vec![robbed_meta_index]
                        } else {
                            Vec::new()
                        }
                    }
                    BrickOwnedBy::None => Vec::new(),
                };

                // Set parent connection
                debug_assert!(
                    self.upload_targets
                        .node_key_vs_meta_index
                        .contains_left(&parent_node_key),
                    "Expected brick parent to be in GPU render data at the time of upload"
                );
                let parent_meta_index = self
                    .upload_targets
                    .node_key_vs_meta_index
                    .get_by_left(&parent_node_key)
                    .unwrap();
                modified_nodes.push(*parent_meta_index);

                if target_sectant as u8 != OOB_SECTANT {
                    let parent_child_index =
                        (parent_meta_index * BOX_NODE_CHILDREN_COUNT) + target_sectant;
                    self.render_data.node_children[parent_child_index] =
                        0x7FFFFFFF & brick_index as u32;
                } else {
                    self.render_data.node_mips[*parent_meta_index] =
                        0x7FFFFFFF & brick_index as u32;
                }

                // Set tracking data on CPU
                self.upload_targets.brick_positions[brick_index] = brick_request.min_position;
                self.upload_targets
                    .brick_ownership
                    .insert(brick_index, brick_request.ownership);

                debug_assert_eq!(
                    tree.brick_dim.pow(3) as usize,
                    brick.len(),
                    "Expected Brick slice to align to tree brick dimension"
                );
                CacheUpdatePackage {
                    allocation_failed: false,
                    brick_update: Some(BrickUpdate {
                        brick_index,
                        data: &brick[..],
                    }),
                    modified_nodes,
                }
            }
        }
    }
}
