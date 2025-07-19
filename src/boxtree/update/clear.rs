use crate::{
    boxtree::{
        iterate::execute_for_relevant_sectants,
        types::{BrickData, NodeChildren, NodeContent, OctreeError, PaletteIndexValues},
        BoxTree, VoxelData, BOX_NODE_CHILDREN_COUNT,
    },
    object_pool::empty_marker,
    spatial::{
        math::{flat_projection, vector::V3c},
        Cube, CubeSides,
    },
};
use std::hash::Hash;

#[cfg(feature = "bytecode")]
use bendy::{decoding::FromBencode, encoding::ToBencode};

impl<
        #[cfg(all(feature = "bytecode", feature = "serialization"))] T: FromBencode
            + ToBencode
            + Serialize
            + DeserializeOwned
            + Default
            + Eq
            + Clone
            + Hash
            + VoxelData,
        #[cfg(all(feature = "bytecode", not(feature = "serialization")))] T: FromBencode + ToBencode + Default + Eq + Clone + Hash + VoxelData,
        #[cfg(all(not(feature = "bytecode"), feature = "serialization"))] T: Serialize + DeserializeOwned + Default + Eq + Clone + Hash + VoxelData,
        #[cfg(all(not(feature = "bytecode"), not(feature = "serialization")))] T: Default + Eq + Clone + Hash + VoxelData,
    > BoxTree<T>
{
    /// clears the voxel at the given position
    pub fn clear(&mut self, position: &V3c<u32>) -> Result<(), OctreeError> {
        self.clear_at_lod(position, 1)
    }

    /// Clears the data at the given position and lod size
    /// * `position` - the position to insert data into, must be contained within the tree
    /// * `clear_size` - The size to update. The value `brick_dimension * (2^x)` is used instead, when size is higher, than brick_dimension
    pub fn clear_at_lod(
        &mut self,
        position_u32: &V3c<u32>,
        clear_size: u32,
    ) -> Result<(), OctreeError> {
        let position = V3c::<f32>::from(*position_u32);
        let root_bounds = Cube::root_bounds(self.boxtree_size as f32);
        if !root_bounds.contains(&position) {
            return Err(OctreeError::InvalidPosition {
                x: position_u32.x,
                y: position_u32.y,
                z: position_u32.z,
            });
        }

        // Nothing to do when no operations are requested
        if clear_size == 0 {
            return Ok(());
        }
        // A CPU stack does not consume significant relevant resources, e.g. a 4096*4096*4096 chunk has depth of 12
        let mut node_stack = vec![(
            Self::ROOT_NODE_KEY as usize,
            root_bounds.sectant_for(&position),
        )];
        let mut bounds_stack = vec![root_bounds];
        let mut erased_whole_sectants = vec![];
        let mut modified_bottom_sectants = vec![];
        let mut actual_update_size = V3c::unit(0);
        let mut updated = false;

        loop {
            let (current_node_key, target_child_sectant) = *node_stack.last().unwrap();
            let current_bounds = bounds_stack.last().unwrap();
            let target_bounds = current_bounds.child_bounds_for(target_child_sectant);
            let mut target_child_key = self.nodes.get(current_node_key).child(target_child_sectant);
            debug_assert!(
                target_bounds.size >= 1.
                    || matches!(
                        self.nodes.get(current_node_key).content,
                        NodeContent::UniformLeaf(_)
                    ),
                "Invalid target bounds(too small): {target_bounds:?}"
            );

            // Trying to clear whole nodes
            if clear_size > 1
                && target_bounds.size <= clear_size as f32
                && position <= target_bounds.min_position
                && matches!(
                    self.nodes.get(current_node_key).content,
                    NodeContent::Internal
                )
            {
                actual_update_size = execute_for_relevant_sectants(
                    current_bounds,
                    position_u32,
                    clear_size,
                    |position_in_target,
                     update_size_in_target,
                     child_sectant,
                     child_target_bounds| {
                        if position_in_target == child_target_bounds.min_position.into()
                            && update_size_in_target.x == child_target_bounds.size as u32
                            && update_size_in_target.y == child_target_bounds.size as u32
                            && update_size_in_target.z == child_target_bounds.size as u32
                        {
                            target_child_key =
                                self.nodes.get(current_node_key).child(child_sectant);

                            // Erase the whole child node
                            if self.nodes.key_is_valid(target_child_key) {
                                updated = true;
                                if self.nodes.key_is_valid(target_child_key) {
                                    self.deallocate_children_of(target_child_key);
                                    self.nodes.get_mut(target_child_key).content =
                                        NodeContent::Nothing;
                                    self.nodes.get_mut(target_child_key).children =
                                        NodeChildren::NoChildren;
                                }
                                erased_whole_sectants.push(child_sectant);
                            }
                            // If the target child is empty(invalid key), there's nothing to do as the targeted area is empty already
                        }
                    },
                );
                break;
            }

            if target_bounds.size > clear_size.max(self.brick_dim) as f32
                || self.nodes.key_is_valid(target_child_key)
            {
                // iteration needs to go deeper, as current Node size is still larger, than the requested clear size
                if self.nodes.key_is_valid(target_child_key) {
                    // iteration can go deeper , as target child is valid
                    node_stack.push((
                        self.nodes.get(current_node_key).child(target_child_sectant),
                        target_bounds.sectant_for(&position),
                    ));
                    bounds_stack.push(target_bounds);
                } else {
                    // no children are available for the target sectant
                    if matches!(
                        self.nodes.get(current_node_key).content,
                        NodeContent::Leaf(_) | NodeContent::UniformLeaf(_)
                    ) {
                        // The current Node is a leaf, representing the area under current_bounds
                        // filled with the data stored in NodeContent::*Leaf(_)
                        let target_match = match &self.nodes.get(current_node_key).content {
                            NodeContent::Nothing | NodeContent::Internal => {
                                panic!("Non-leaf node expected to be leaf!")
                            }
                            NodeContent::UniformLeaf(brick) => match brick {
                                BrickData::Empty => true,
                                BrickData::Solid(voxel) => NodeContent::pix_points_to_empty(
                                    voxel,
                                    &self.voxel_color_palette,
                                    &self.voxel_data_palette,
                                ),
                                BrickData::Parted(brick) => {
                                    let index_in_matrix = position - current_bounds.min_position;
                                    let index_in_matrix = flat_projection(
                                        index_in_matrix.x as usize,
                                        index_in_matrix.y as usize,
                                        index_in_matrix.z as usize,
                                        self.brick_dim as usize,
                                    );
                                    NodeContent::pix_points_to_empty(
                                        &brick[index_in_matrix],
                                        &self.voxel_color_palette,
                                        &self.voxel_data_palette,
                                    )
                                }
                            },
                            NodeContent::Leaf(bricks) => {
                                match &bricks[target_child_sectant as usize] {
                                    BrickData::Empty => true,
                                    BrickData::Solid(voxel) => NodeContent::pix_points_to_empty(
                                        voxel,
                                        &self.voxel_color_palette,
                                        &self.voxel_data_palette,
                                    ),
                                    BrickData::Parted(brick) => {
                                        let index_in_matrix =
                                            position - current_bounds.min_position;
                                        let index_in_matrix = flat_projection(
                                            index_in_matrix.x as usize,
                                            index_in_matrix.y as usize,
                                            index_in_matrix.z as usize,
                                            self.brick_dim as usize,
                                        );
                                        NodeContent::pix_points_to_empty(
                                            &brick[index_in_matrix],
                                            &self.voxel_color_palette,
                                            &self.voxel_data_palette,
                                        )
                                    }
                                }
                            }
                        };
                        if target_match
                            || self
                                .nodes
                                .get(current_node_key)
                                .content
                                .is_empty(&self.voxel_color_palette, &self.voxel_data_palette)
                        {
                            // the data stored equals the given data, at the requested position
                            // so no need to continue iteration as data already matches
                            break;
                        }

                        // The contained data does not match the given data at the given position,
                        // but the current node is a leaf, so it needs to be divided into separate nodes
                        // with its children having the same data as the current node, to keep integrity.
                        // It needs to be separated because it has an extent above DIM
                        debug_assert!(
                            current_bounds.size > self.brick_dim as f32,
                            "Expected Leaf node to have an extent({:?}) above DIM({:?})!",
                            current_bounds.size,
                            self.brick_dim
                        );
                        self.subdivide_leaf_to_nodes(
                            current_node_key,
                            target_child_sectant as usize,
                        );
                        //target_child_key = empty_marker(); // Note: target_child_key no longer valid
                        node_stack.push((
                            self.nodes.get(current_node_key).child(target_child_sectant),
                            target_bounds.sectant_for(&position),
                        ));
                        bounds_stack.push(target_bounds);
                    } else {
                        // current Node is a non-leaf Node, which doesn't have the child at the requested position.
                        // Nothing to do, because child didn't exist in the first place
                        break;
                    }
                }
            } else {
                // when clearing with size > DIM, Nodes are being cleared
                // current_bounds.size == min_node_size, which is the desired depth
                actual_update_size = execute_for_relevant_sectants(
                    current_bounds,
                    position_u32,
                    clear_size,
                    |position_in_target,
                     update_size_in_target,
                     child_sectant,
                     child_target_bounds| {
                        updated |= self.leaf_update(
                            true,
                            (current_node_key, current_bounds),
                            (child_target_bounds, child_sectant as usize),
                            (&position_in_target, &update_size_in_target),
                            empty_marker::<PaletteIndexValues>(),
                        );
                        modified_bottom_sectants.push(child_sectant);
                    },
                );
                break;
            }
        }

        if !updated {
            // No need to do post-processing operations if data wasn't updated..
            return Ok(());
        }

        // post-processing operations
        let mut simplifyable = self.auto_simplify; // Don't even start to simplify if it's disabled
        for modified_bottom_sectant in modified_bottom_sectants {
            let (node_key, original_sectant) = node_stack.last().cloned().unwrap();
            let node_bounds = bounds_stack.last().unwrap();
            let child_key = self.nodes.get(node_key).child(modified_bottom_sectant);

            if self.nodes.key_is_valid(child_key) {
                // Check bottom update as a node
                let child_bounds = node_bounds.child_bounds_for(modified_bottom_sectant);

                // Add child node into the stack
                node_stack.push((
                    child_key,
                    child_bounds.sectant_for(&V3c::new(
                        position.x.max(child_bounds.min_position.x),
                        position.y.max(child_bounds.min_position.y),
                        position.z.max(child_bounds.min_position.z),
                    )),
                ));
                self.post_process_node_clear(
                    &node_stack,
                    &child_bounds,
                    &actual_update_size,
                    position_u32,
                    clear_size,
                    vec![],
                );
                node_stack.pop();
            } else {
                node_stack.last_mut().unwrap().1 = modified_bottom_sectant;

                // Check bottom update as leaf
                self.post_process_node_clear(
                    &node_stack,
                    node_bounds,
                    &actual_update_size,
                    position_u32,
                    clear_size,
                    vec![],
                );
                node_stack.last_mut().unwrap().1 = original_sectant;
            }

            if simplifyable {
                simplifyable &= self.simplify(child_key, false);
            }
        }

        // processing higher level nodes
        while !node_stack.is_empty() {
            erased_whole_sectants = if self.post_process_node_clear(
                &node_stack,
                bounds_stack.last().unwrap(),
                &actual_update_size,
                position_u32,
                clear_size,
                erased_whole_sectants,
            ) {
                vec![bounds_stack.last().unwrap().sectant_for(&position)]
            } else {
                vec![]
            };

            // If any Nodes fail to simplify, no need to continue because their parents can not be simplified further
            if simplifyable {
                simplifyable = self.simplify(node_stack.last().unwrap().0, true);
            }

            node_stack.pop();
            bounds_stack.pop();
        }

        Ok(())
    }

    /// Node post-process for connections, content, mips, occupied bits and occlusion bits
    /// after data deletion
    /// Returns true if the whole node was delted or non-existent in the first place
    fn post_process_node_clear(
        &mut self,
        node_stack: &[(usize, u8)],
        node_bounds: &Cube,
        actual_update_size: &V3c<usize>,
        clear_position: &V3c<u32>,
        clear_size: u32,
        removed_children: Vec<u8>,
    ) -> bool {
        debug_assert_ne!(0, node_stack.len());
        let node_key = node_stack.last().unwrap().0;

        // node might already be removed
        if !self.nodes.key_is_valid(node_key) {
            return true;
        }

        // Any child cleared during this operation needs to be freed up
        // and parent connection needs to be updated as well
        if 0 < removed_children.len() {
            let mut node_stack = node_stack.to_vec();
            for child_sectant in removed_children {
                let child_key = self.nodes.get(node_key).child(child_sectant);

                // Set occlusion bits for deleted children
                //TODO: Except for leaf nodes..
                //TODO: for here and insert!!
                if self.nodes.key_is_valid(child_key) {
                    if self.nodes.get_mut(child_key).occupied_bits == u64::MAX {
                        let child_bounds = node_bounds.child_bounds_for(child_sectant);
                        node_stack.push((
                            child_key,
                            child_bounds.sectant_for(
                                &(&V3c::new(
                                    (clear_position.x as f32).max(child_bounds.min_position.x),
                                    (clear_position.y as f32).max(child_bounds.min_position.y),
                                    (clear_position.z as f32).max(child_bounds.min_position.z),
                                )),
                            ),
                        ));
                        for (direction, side) in [
                            (V3c::new(-1., 0., 0.), CubeSides::Right),
                            (V3c::new(1., 0., 0.), CubeSides::Left),
                            (V3c::new(0., -1., 0.), CubeSides::Top),
                            (V3c::new(0., 1., 0.), CubeSides::Bottom),
                            (V3c::new(0., 0., -1.), CubeSides::Front),
                            (V3c::new(0., 0., 1.), CubeSides::Back),
                        ]
                        .iter()
                        {
                            if let Some((sibling_node, _sibling_sectant)) =
                                self.get_sibling_by_stack(*direction, &node_stack)
                            {
                                self.nodes.get_mut(sibling_node).set_occlusion(*side, false);
                            }
                        }
                        node_stack.pop();
                    }
                    self.nodes.free(child_key);
                }

                self.nodes
                    .get_mut(node_key)
                    .clear_child(child_sectant as usize);
            }
        }

        let mut new_occupied_bits = self.nodes.get(node_key).occupied_bits;
        if node_bounds.size as usize == actual_update_size.x
            && node_bounds.size as usize == actual_update_size.y
            && node_bounds.size as usize == actual_update_size.z
            && V3c::from(node_bounds.min_position) == *clear_position
        {
            new_occupied_bits = 0;
        } else {
            execute_for_relevant_sectants(
                node_bounds,
                clear_position,
                clear_size,
                |_position_in_target,
                 _update_size_in_target,
                 child_sectant,
                 _child_target_bounds| {
                    if self.node_empty_at(node_key, child_sectant) {
                        new_occupied_bits &= !(0x01 << child_sectant);
                    }
                },
            );
        }

        // If Occupied bits depleted, deallocate children and unset node
        if 0 == new_occupied_bits {
            debug_assert_eq!(
                BOX_NODE_CHILDREN_COUNT,
                (0..BOX_NODE_CHILDREN_COUNT)
                    .filter(|sectant| { self.node_empty_at(node_key, *sectant as u8) })
                    .count(),
                "Expected empty node to have no valid children!"
            );
            self.deallocate_children_of(node_key);
            self.nodes.get_mut(node_key).children = NodeChildren::NoChildren;
            self.nodes.get_mut(node_key).content = NodeContent::Nothing;
        };

        debug_assert!(
            0 != new_occupied_bits
                || matches!(self.nodes.get(node_key).content, NodeContent::Nothing),
            "Occupied bits doesn't match node[{:?}]: {:?} <> {:?}\nnode children: {:?}",
            node_key,
            new_occupied_bits,
            self.nodes.get(node_key).content,
            self.nodes.get(node_key).children,
        );

        // Update sibling nodes occlusion bits ( direction paired with opposite side on sibling node )
        if self.nodes.get_mut(node_key).occupied_bits == u64::MAX && new_occupied_bits != u64::MAX {
            for (direction, side) in [
                (V3c::new(-1., 0., 0.), CubeSides::Right),
                (V3c::new(1., 0., 0.), CubeSides::Left),
                (V3c::new(0., -1., 0.), CubeSides::Top),
                (V3c::new(0., 1., 0.), CubeSides::Bottom),
                (V3c::new(0., 0., -1.), CubeSides::Front),
                (V3c::new(0., 0., 1.), CubeSides::Back),
            ]
            .iter()
            {
                if let Some((sibling_node, _sibling_sectant)) =
                    self.get_sibling_by_stack(*direction, node_stack)
                {
                    self.nodes.get_mut(sibling_node).set_occlusion(*side, false);
                }
            }
        }
        self.nodes.get_mut(node_key).occupied_bits = new_occupied_bits;
        self.update_mip(node_key, node_bounds, clear_position);

        0 == new_occupied_bits
    }
}
