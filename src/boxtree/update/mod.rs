pub mod clear;
pub mod insert;

#[cfg(test)]
mod tests;

use crate::{
    boxtree::{
        types::{BoxTreeEntry, BrickData, NodeChildren, NodeContent, PaletteIndexValues},
        Albedo, BoxTree, VoxelData, BOX_NODE_CHILDREN_COUNT, BOX_NODE_DIMENSION,
    },
    object_pool::empty_marker,
    spatial::{
        lut::SECTANT_OFFSET_LUT,
        math::{
            flat_projection, matrix_index_for, octant_in_sectants, offset_sectant, vector::V3c,
        },
        Cube,
    },
};
use num_traits::Zero;
use std::{fmt::Debug, hash::Hash};

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
    //####################################################################################
    // ███████████    █████████   █████       ██████████ ███████████ ███████████ ██████████
    // ░░███░░░░░███  ███░░░░░███ ░░███       ░░███░░░░░█░█░░░███░░░█░█░░░███░░░█░░███░░░░░█
    //  ░███    ░███ ░███    ░███  ░███        ░███  █ ░ ░   ░███  ░ ░   ░███  ░  ░███  █ ░
    //  ░██████████  ░███████████  ░███        ░██████       ░███        ░███     ░██████
    //  ░███░░░░░░   ░███░░░░░███  ░███        ░███░░█       ░███        ░███     ░███░░█
    //  ░███         ░███    ░███  ░███      █ ░███ ░   █    ░███        ░███     ░███ ░   █
    //  █████        █████   █████ ███████████ ██████████    █████       █████    ██████████
    // ░░░░░        ░░░░░   ░░░░░ ░░░░░░░░░░░ ░░░░░░░░░░    ░░░░░       ░░░░░    ░░░░░░░░░░
    //####################################################################################
    /// Updates the stored palette by adding the new colors and data in the given entry
    /// Since unused colors are not removed from the palette, possible "pollution" is possible,
    /// where unused colors remain in the palette.
    /// * Returns with the resulting PaletteIndexValues Entry
    pub(crate) fn add_to_palette(&mut self, entry: &BoxTreeEntry<T>) -> PaletteIndexValues {
        match entry {
            BoxTreeEntry::Empty => empty_marker::<PaletteIndexValues>(),
            BoxTreeEntry::Visual(albedo) => {
                if **albedo == Albedo::zero() {
                    return empty_marker();
                }
                let potential_new_albedo_index = self.map_to_color_index_in_palette.keys().len();
                let albedo_index = if let std::collections::hash_map::Entry::Vacant(e) =
                    self.map_to_color_index_in_palette.entry(**albedo)
                {
                    e.insert(potential_new_albedo_index);
                    self.voxel_color_palette.push(**albedo);
                    potential_new_albedo_index
                } else {
                    self.map_to_color_index_in_palette[albedo]
                };
                debug_assert!(
                    albedo_index < u16::MAX as usize,
                    "Albedo color palette overflow!"
                );
                NodeContent::pix_visual(albedo_index as u16)
            }
            BoxTreeEntry::Informative(data) => {
                if data.is_empty() {
                    return empty_marker();
                }
                let potential_new_data_index = self.map_to_data_index_in_palette.keys().len();
                let data_index = if let std::collections::hash_map::Entry::Vacant(e) =
                    self.map_to_data_index_in_palette.entry((*data).clone())
                {
                    e.insert(potential_new_data_index);
                    self.voxel_data_palette.push((*data).clone());
                    potential_new_data_index
                } else {
                    self.map_to_data_index_in_palette[data]
                };
                debug_assert!(
                    data_index < u16::MAX as usize,
                    "Data color palette overflow!"
                );
                NodeContent::pix_informal(data_index as u16)
            }
            BoxTreeEntry::Complex(albedo, data) => {
                if **albedo == Albedo::zero() {
                    return self.add_to_palette(&BoxTreeEntry::Informative(*data));
                } else if data.is_empty() {
                    return self.add_to_palette(&BoxTreeEntry::Visual(albedo));
                }
                let potential_new_albedo_index = self.map_to_color_index_in_palette.keys().len();
                let albedo_index = if let std::collections::hash_map::Entry::Vacant(e) =
                    self.map_to_color_index_in_palette.entry(**albedo)
                {
                    e.insert(potential_new_albedo_index);
                    self.voxel_color_palette.push(**albedo);
                    potential_new_albedo_index
                } else {
                    self.map_to_color_index_in_palette[albedo]
                };
                let potential_new_data_index = self.map_to_data_index_in_palette.keys().len();
                let data_index = if let std::collections::hash_map::Entry::Vacant(e) =
                    self.map_to_data_index_in_palette.entry((*data).clone())
                {
                    e.insert(potential_new_data_index);
                    self.voxel_data_palette.push((*data).clone());
                    potential_new_data_index
                } else {
                    self.map_to_data_index_in_palette[data]
                };
                debug_assert!(
                    albedo_index < u16::MAX as usize,
                    "Albedo color palette overflow!"
                );
                debug_assert!(
                    data_index < u16::MAX as usize,
                    "Data color palette overflow!"
                );
                NodeContent::pix_complex(albedo_index as u16, data_index as u16)
            }
        }
        // find color in the palette is present, add if not
    }

    //####################################################################################
    //  █████       ██████████   █████████   ███████████
    // ░░███       ░░███░░░░░█  ███░░░░░███ ░░███░░░░░░█
    //  ░███        ░███  █ ░  ░███    ░███  ░███   █ ░
    //  ░███        ░██████    ░███████████  ░███████
    //  ░███        ░███░░█    ░███░░░░░███  ░███░░░█
    //  ░███      █ ░███ ░   █ ░███    ░███  ░███  ░
    //  ███████████ ██████████ █████   █████ █████
    // ░░░░░░░░░░░ ░░░░░░░░░░ ░░░░░   ░░░░░ ░░░░░
    //  █████  █████ ███████████  ██████████     █████████   ███████████ ██████████
    // ░░███  ░░███ ░░███░░░░░███░░███░░░░███   ███░░░░░███ ░█░░░███░░░█░░███░░░░░█
    //  ░███   ░███  ░███    ░███ ░███   ░░███ ░███    ░███ ░   ░███  ░  ░███  █ ░
    //  ░███   ░███  ░██████████  ░███    ░███ ░███████████     ░███     ░██████
    //  ░███   ░███  ░███░░░░░░   ░███    ░███ ░███░░░░░███     ░███     ░███░░█
    //  ░███   ░███  ░███         ░███    ███  ░███    ░███     ░███     ░███ ░   █
    //  ░░████████   █████        ██████████   █████   █████    █████    ██████████
    //   ░░░░░░░░   ░░░░░        ░░░░░░░░░░   ░░░░░   ░░░░░    ░░░░░    ░░░░░░░░░░
    //####################################################################################
    /// Updates the given node to be a Leaf, and inserts the provided data for it.
    /// It will update a whole node, or maximum one brick. Brick update range is starting from the position,
    /// goes up to the extent of the brick. Does not set occupancy bitmap of the given node.
    /// * Returns with the size of the actual update
    pub(crate) fn leaf_update(
        &mut self,
        overwrite_if_empty: bool,
        (node_key, node_bounds): (usize, &Cube),
        (target_bounds, target_child_sectant): (&Cube, usize),
        (position, size): (&V3c<u32>, &V3c<u32>),
        target_content: PaletteIndexValues,
    ) -> bool {
        // Update the leaf node, if it is possible as is, and if it's even needed to update
        // and decide if the node content needs to be divided into bricks, and the update function to be called again
        match &mut self.nodes.get_mut(node_key).content {
            NodeContent::Leaf(bricks) => {
                // In case brick_dimension == boxtree size, the 0 can not be a leaf...
                debug_assert!(self.brick_dim < self.boxtree_size);
                match &mut bricks[target_child_sectant] {
                    //If there is no brick in the target position of the leaf, create one
                    BrickData::Empty => {
                        // Create a new empty brick at the given sectant
                        let mut new_brick = vec![
                            empty_marker::<PaletteIndexValues>();
                            self.brick_dim.pow(3) as usize
                        ];
                        // update the new empty brick at the given position
                        Self::update_brick(
                            overwrite_if_empty,
                            &mut new_brick,
                            target_bounds,
                            self.brick_dim,
                            *position,
                            *size,
                            &target_content,
                        );
                        bricks[target_child_sectant] = BrickData::Parted(new_brick);
                        true
                    }
                    BrickData::Solid(voxel) => {
                        // In case the data doesn't match the current contents of the node, it needs to be subdivided
                        if (NodeContent::pix_points_to_empty(
                            &target_content,
                            &self.voxel_color_palette,
                            &self.voxel_data_palette,
                        ) && !NodeContent::pix_points_to_empty(
                            voxel,
                            &self.voxel_color_palette,
                            &self.voxel_data_palette,
                        )) || (!NodeContent::pix_points_to_empty(
                            &target_content,
                            &self.voxel_color_palette,
                            &self.voxel_data_palette,
                        ) && *voxel != target_content)
                        {
                            // create new brick and update it at the given position
                            let mut new_brick = vec![*voxel; self.brick_dim.pow(3) as usize];
                            Self::update_brick(
                                overwrite_if_empty,
                                &mut new_brick,
                                target_bounds,
                                self.brick_dim,
                                *position,
                                *size,
                                &target_content,
                            );
                            bricks[target_child_sectant] = BrickData::Parted(new_brick);
                            true
                        } else {
                            // Since the Voxel already equals the data to be set, no need to update anything
                            false
                        }
                    }
                    BrickData::Parted(brick) => {
                        // Simply update the brick at the given position
                        Self::update_brick(
                            overwrite_if_empty,
                            brick,
                            target_bounds,
                            self.brick_dim,
                            *position,
                            *size,
                            &target_content,
                        );
                        true
                    }
                }
            }
            NodeContent::UniformLeaf(mat) => {
                match mat {
                    BrickData::Empty => {
                        debug_assert_eq!(
                            self.nodes.get(node_key).occupied_bits,
                            0,
                            "Expected Node OccupancyBitmap 0 for empty leaf node instead of {:?}",
                            self.nodes.get(node_key).occupied_bits
                        );
                        if !NodeContent::pix_points_to_empty(
                            &target_content,
                            &self.voxel_color_palette,
                            &self.voxel_data_palette,
                        ) {
                            let mut new_leaf_content: [BrickData<PaletteIndexValues>;
                                BOX_NODE_CHILDREN_COUNT] =
                                vec![BrickData::Empty; BOX_NODE_CHILDREN_COUNT]
                                    .try_into()
                                    .unwrap();

                            // Add a brick to the target sectant and update with the given data
                            let mut new_brick = vec![
                                self.add_to_palette(&BoxTreeEntry::Empty);
                                self.brick_dim.pow(3) as usize
                            ];
                            Self::update_brick(
                                overwrite_if_empty,
                                &mut new_brick,
                                target_bounds,
                                self.brick_dim,
                                *position,
                                *size,
                                &target_content,
                            );
                            new_leaf_content[target_child_sectant] = BrickData::Parted(new_brick);
                            self.nodes.get_mut(node_key).content =
                                NodeContent::Leaf(new_leaf_content);
                            return true;
                        }
                    }
                    BrickData::Solid(voxel) => {
                        // In case the data request doesn't match node content, it needs to be subdivided
                        if NodeContent::pix_points_to_empty(
                            &target_content,
                            &self.voxel_color_palette,
                            &self.voxel_data_palette,
                        ) && NodeContent::pix_points_to_empty(
                            voxel,
                            &self.voxel_color_palette,
                            &self.voxel_data_palette,
                        ) {
                            // Data request is to clear, it aligns with the voxel content,
                            // it's enough to update the node content in this case
                            self.nodes.get_mut(node_key).content = NodeContent::Nothing;
                            return false;
                        }

                        if !NodeContent::pix_points_to_empty(
                            &target_content,
                            &self.voxel_color_palette,
                            &self.voxel_data_palette,
                        ) && *voxel != target_content
                            || (NodeContent::pix_points_to_empty(
                                &target_content,
                                &self.voxel_color_palette,
                                &self.voxel_data_palette,
                            ) && !NodeContent::pix_points_to_empty(
                                voxel,
                                &self.voxel_color_palette,
                                &self.voxel_data_palette,
                            ))
                        {
                            // Data request doesn't align with the voxel data
                            // create a voxel brick and try to update with the given data
                            *mat = BrickData::Parted(vec![
                                *voxel;
                                (self.brick_dim * self.brick_dim * self.brick_dim)
                                    as usize
                            ]);

                            return self.leaf_update(
                                overwrite_if_empty,
                                (node_key, node_bounds),
                                (target_bounds, target_child_sectant),
                                (position, size),
                                target_content,
                            );
                        }

                        // data request aligns with node content
                        return false;
                    }
                    BrickData::Parted(brick) => {
                        // Check if the voxel at the target position matches with the data update request
                        // The target position index is to be calculated from the node bounds,
                        // instead of the target bounds because the position should cover the whole leaf
                        // not just one brick in it
                        let mat_index = matrix_index_for(node_bounds, position, self.brick_dim);
                        let mat_index = flat_projection(
                            mat_index.x,
                            mat_index.y,
                            mat_index.z,
                            self.brick_dim as usize,
                        );
                        if 1 < self.brick_dim // BrickData can only stay parted if brick_dimension is above 1
                            && (
                                (
                                    NodeContent::pix_points_to_empty(
                                        &target_content,
                                        &self.voxel_color_palette,
                                        &self.voxel_data_palette,
                                    )
                                    && NodeContent::pix_points_to_empty(
                                        &brick[mat_index],
                                        &self.voxel_color_palette,
                                        &self.voxel_data_palette
                                    )
                                )||(
                                    !NodeContent::pix_points_to_empty(
                                        &target_content,
                                        &self.voxel_color_palette,
                                        &self.voxel_data_palette,
                                    )
                                    && brick[mat_index] == target_content
                                )
                            )
                        {
                            // Target voxel matches with the data request, there's nothing to do!
                            return false;
                        }

                        // If uniform leaf is the size of one brick, the brick is updated as is
                        if node_bounds.size <= self.brick_dim as f32 && self.brick_dim > 1 {
                            Self::update_brick(
                                overwrite_if_empty,
                                brick,
                                node_bounds,
                                self.brick_dim,
                                *position,
                                *size,
                                &target_content,
                            );
                            return true;
                        }

                        // the data at the position inside the brick doesn't match the given data,
                        // so the leaf needs to be divided into a NodeContent::Leaf(bricks)
                        let mut leaf_data: [BrickData<PaletteIndexValues>;
                            BOX_NODE_CHILDREN_COUNT] =
                            vec![BrickData::Empty; BOX_NODE_CHILDREN_COUNT]
                                .try_into()
                                .unwrap();

                        // Each brick is mapped to take up one subsection of the current data
                        let child_bricks =
                            Self::dilute_brick_data(std::mem::take(brick), self.brick_dim);
                        let mut updated = false;
                        for (sectant, mut new_brick) in child_bricks.into_iter().enumerate() {
                            // Also update the brick if it is the target
                            if sectant == target_child_sectant {
                                Self::update_brick(
                                    overwrite_if_empty,
                                    &mut new_brick,
                                    target_bounds,
                                    self.brick_dim,
                                    *position,
                                    *size,
                                    &target_content,
                                );
                                updated |= true;
                            }
                            leaf_data[sectant] = BrickData::Parted(new_brick);
                        }

                        self.nodes.get_mut(node_key).content = NodeContent::Leaf(leaf_data);
                        debug_assert!(updated, "Expected Leaf node to be updated in operation");
                        return updated;
                    }
                }
                self.leaf_update(
                    overwrite_if_empty,
                    (node_key, node_bounds),
                    (target_bounds, target_child_sectant),
                    (position, size),
                    target_content,
                )
            }
            NodeContent::Internal => {
                // Warning: Calling leaf update to an internal node might induce data loss - see #69
                self.nodes.get_mut(node_key).children = NodeChildren::NoChildren;
                self.nodes.get_mut(node_key).content = NodeContent::Leaf(
                    (0..BOX_NODE_CHILDREN_COUNT)
                        .map(|sectant| {
                            self.try_brick_from_node(self.nodes.get(node_key).child(sectant as u8))
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                );
                self.deallocate_children_of(node_key);
                self.leaf_update(
                    overwrite_if_empty,
                    (node_key, node_bounds),
                    (target_bounds, target_child_sectant),
                    (position, size),
                    target_content,
                )
            }
            NodeContent::Nothing => {
                // Calling leaf update on Nothing is an odd thing to do..
                // But possible, if this call is mid-update
                // So let's try to gather all the information possible
                self.nodes.get_mut(node_key).content = NodeContent::Leaf(
                    (0..BOX_NODE_CHILDREN_COUNT)
                        .map(|sectant| {
                            self.try_brick_from_node(self.nodes.get(node_key).child(sectant as u8))
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                );
                self.deallocate_children_of(node_key);
                self.leaf_update(
                    overwrite_if_empty,
                    (node_key, node_bounds),
                    (target_bounds, target_child_sectant),
                    (position, size),
                    target_content,
                )
            }
        }
    }

    //####################################################################################
    //  ███████████  ███████████   █████   █████████  █████   ████
    // ░░███░░░░░███░░███░░░░░███ ░░███   ███░░░░░███░░███   ███░
    //  ░███    ░███ ░███    ░███  ░███  ███     ░░░  ░███  ███
    //  ░██████████  ░██████████   ░███ ░███          ░███████
    //  ░███░░░░░███ ░███░░░░░███  ░███ ░███          ░███░░███
    //  ░███    ░███ ░███    ░███  ░███ ░░███     ███ ░███ ░░███
    //  ███████████  █████   █████ █████ ░░█████████  █████ ░░████
    // ░░░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░░░░░  ░░░░░   ░░░░
    //####################################################################################
    /// Provides an array of bricks, based on the given brick data, with the same size of the original brick,
    /// each voxel mapped as the new bricks were the children of the given brick
    pub(crate) fn dilute_brick_data<B>(
        brick_data: Vec<B>,
        brick_dim: u32,
    ) -> [Vec<B>; BOX_NODE_CHILDREN_COUNT]
    where
        B: Debug + Clone + Copy + PartialEq,
    {
        debug_assert_eq!(brick_data.len(), brick_dim.pow(3) as usize);

        if 1 == brick_dim {
            debug_assert_eq!(brick_data.len(), 1);
            return vec![brick_data.clone(); BOX_NODE_CHILDREN_COUNT]
                .try_into()
                .unwrap();
        }

        if 2 == brick_dim {
            debug_assert_eq!(brick_data.len(), 8);
            return (0..BOX_NODE_CHILDREN_COUNT)
                .map(|sectant| {
                    vec![brick_data[octant_in_sectants(sectant)]; brick_dim.pow(3) as usize]
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
        };

        debug_assert!(brick_data.len() <= BOX_NODE_CHILDREN_COUNT);
        let mut result: [Vec<B>; BOX_NODE_CHILDREN_COUNT] = (0..BOX_NODE_CHILDREN_COUNT)
            .map(|sectant| vec![brick_data[sectant]; brick_dim.pow(3) as usize])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // in case one child can be mapped 1:1 to an element in the brick
        if 4 == brick_dim {
            debug_assert_eq!(brick_data.len(), BOX_NODE_CHILDREN_COUNT);
            return result;
        }

        // Generic case
        // Note: Each value in @result will be overwritten
        for sectant in 0..BOX_NODE_CHILDREN_COUNT {
            // Set the data of the new child
            let brick_offset: V3c<usize> =
                V3c::from(SECTANT_OFFSET_LUT[sectant] * brick_dim as f32);
            let new_brick_flat_offset = flat_projection(
                brick_offset.x,
                brick_offset.y,
                brick_offset.z,
                brick_dim as usize,
            );
            let mut new_brick_data =
                vec![brick_data[new_brick_flat_offset]; brick_dim.pow(3) as usize];
            for x in 0..brick_dim as usize {
                for y in 0..brick_dim as usize {
                    for z in 0..brick_dim as usize {
                        if x < BOX_NODE_DIMENSION
                            && y < BOX_NODE_DIMENSION
                            && z < BOX_NODE_DIMENSION
                        {
                            continue;
                        }
                        let new_brick_flat_offset = flat_projection(x, y, z, brick_dim as usize);
                        let brick_flat_offset = flat_projection(
                            brick_offset.x + x / BOX_NODE_DIMENSION,
                            brick_offset.y + y / BOX_NODE_DIMENSION,
                            brick_offset.z + z / BOX_NODE_DIMENSION,
                            brick_dim as usize,
                        );
                        new_brick_data[new_brick_flat_offset] = brick_data[brick_flat_offset];
                    }
                }
            }
            result[sectant] = new_brick_data;
        }
        result
    }

    /// Updates the content of the given brick and its occupancy bitmap. Each components of mat_index must be smaller, than the size of the brick.
    /// mat_index + size however need not be in bounds, the function will cut each component to fit inside the brick.
    /// * `brick` - mutable reference of the brick to update
    /// * `mat_index` - the first position to update with the given data
    /// * `size` - the number of elements in x,y,z to update with the given data
    /// * `data` - the data  to update the brick with. Erases data in case `None`
    /// * Returns with the size of the update
    fn update_brick(
        overwrite_if_empty: bool,
        brick: &mut [PaletteIndexValues],
        brick_bounds: &Cube,
        brick_dim: u32,
        position: V3c<u32>,
        size: V3c<u32>,
        data: &PaletteIndexValues,
    ) {
        debug_assert!(
            brick_bounds.contains(&(position.into())),
            "Expected position {:?} to be contained in brick bounds {:?}",
            position,
            brick_bounds
        );
        debug_assert!(
            brick_bounds.contains(&V3c::from(position + size - V3c::unit(1))),
            "Expected position {:?} and update_size {:?} to be contained in brick bounds {:?}",
            position,
            size,
            brick_bounds
        );

        let mat_index = matrix_index_for(brick_bounds, &position, brick_dim);

        for x in mat_index.x..(mat_index.x + size.x as usize).min(brick_dim as usize) {
            for y in mat_index.y..(mat_index.y + size.y as usize).min(brick_dim as usize) {
                for z in mat_index.z..(mat_index.z + size.z as usize).min(brick_dim as usize) {
                    let mat_index = flat_projection(x, y, z, brick_dim as usize);
                    if overwrite_if_empty {
                        brick[mat_index] = *data;
                    } else {
                        if NodeContent::pix_color_is_some(data) {
                            brick[mat_index] =
                                NodeContent::pix_overwrite_color(brick[mat_index], data);
                        }
                        if NodeContent::pix_data_is_some(data) {
                            brick[mat_index] =
                                NodeContent::pix_overwrite_data(brick[mat_index], data);
                        }
                    }
                }
            }
        }
    }

    //####################################################################################
    //   █████████  █████ ██████   ██████ ███████████  █████       █████ ███████████ █████ █████
    //  ███░░░░░███░░███ ░░██████ ██████ ░░███░░░░░███░░███       ░░███ ░░███░░░░░░█░░███ ░░███
    // ░███    ░░░  ░███  ░███░█████░███  ░███    ░███ ░███        ░███  ░███   █ ░  ░░███ ███
    // ░░█████████  ░███  ░███░░███ ░███  ░██████████  ░███        ░███  ░███████     ░░█████
    //  ░░░░░░░░███ ░███  ░███ ░░░  ░███  ░███░░░░░░   ░███        ░███  ░███░░░█      ░░███
    //  ███    ░███ ░███  ░███      ░███  ░███         ░███      █ ░███  ░███  ░        ░███
    // ░░█████████  █████ █████     █████ █████        ███████████ █████ █████          █████
    //  ░░░░░░░░░  ░░░░░ ░░░░░     ░░░░░ ░░░░░        ░░░░░░░░░░░ ░░░░░ ░░░░░          ░░░░░
    //####################################################################################
    /// Updates the given node recursively to collapse nodes with uniform children into a leaf
    /// Returns with true if the given node was simplified
    pub(crate) fn simplify(&mut self, node_key: usize, recursive: bool) -> bool {
        if self.nodes.key_is_valid(node_key) {
            #[cfg(debug_assertions)]
            {
                if matches!(self.nodes.get(node_key).content, NodeContent::Internal)
                    && !matches!(self.nodes.get(node_key).children, NodeChildren::NoChildren)
                {
                    for sectant in 0..BOX_NODE_CHILDREN_COUNT as u8 {
                        if self.node_empty_at(node_key, sectant) {
                            debug_assert_eq!(
                                0,
                                self.nodes.get(node_key).occupied_bits & (0x01 << sectant),
                                "Expected node[{:?}] ocbits({:#10X}) to represent child at sectant[{:?}]: \n{:?}",
                                node_key, self.nodes.get(node_key).occupied_bits, sectant,
                                self.nodes.get(self.nodes.get(node_key).child(sectant))
                            )
                        }
                    }
                }
            }

            match &mut self.nodes.get_mut(node_key).content {
                NodeContent::Nothing => true,
                NodeContent::UniformLeaf(brick) => match brick {
                    BrickData::Empty => true,
                    BrickData::Solid(voxel) => {
                        if NodeContent::pix_points_to_empty(
                            voxel,
                            &self.voxel_color_palette,
                            &self.voxel_data_palette,
                        ) {
                            debug_assert_eq!(
                                0, self.nodes.get(node_key).occupied_bits,
                                "Solid empty voxel should have its occupied bits set to 0, instead of {:#10X}",
                                self.nodes.get(node_key).occupied_bits
                            );
                            self.nodes.get_mut(node_key).content = NodeContent::Nothing;
                            self.nodes.get_mut(node_key).children = NodeChildren::NoChildren;
                            true
                        } else {
                            debug_assert_eq!(
                                u64::MAX, self.nodes.get(node_key).occupied_bits,
                                "Solid full voxel should have its occupied bits set to u64::MAX, instead of {:#10X}",
                                self.nodes.get(node_key).occupied_bits
                            );
                            false
                        }
                    }
                    BrickData::Parted(_brick) => {
                        if brick.simplify(&self.voxel_color_palette, &self.voxel_data_palette) {
                            debug_assert!(
                                self.nodes.get(node_key).occupied_bits == u64::MAX
                                || self.nodes.get(node_key).occupied_bits == 0,
                                "Expected brick occuped bits of node[{node_key}] to be either full or empty, becasue it could be simplified",
                            );
                            true
                        } else {
                            false
                        }
                    }
                },
                NodeContent::Leaf(bricks) => {
                    // Try to simplify bricks
                    let mut simplified = false;
                    let mut is_leaf_uniform_solid = true;
                    let mut uniform_solid_value = None;

                    for brick in bricks.iter_mut().take(BOX_NODE_CHILDREN_COUNT) {
                        simplified |=
                            brick.simplify(&self.voxel_color_palette, &self.voxel_data_palette);

                        if is_leaf_uniform_solid {
                            if let BrickData::Solid(voxel) = brick {
                                if let Some(ref uniform_solid_value) = uniform_solid_value {
                                    if *uniform_solid_value != voxel {
                                        is_leaf_uniform_solid = false;
                                    }
                                } else {
                                    uniform_solid_value = Some(voxel);
                                }
                            } else {
                                is_leaf_uniform_solid = false;
                            }
                        }
                    }

                    // Try to unite bricks into a solid brick
                    let mut unified_brick = BrickData::Empty;
                    if is_leaf_uniform_solid {
                        debug_assert_ne!(uniform_solid_value, None);
                        self.nodes.get_mut(node_key).content = NodeContent::UniformLeaf(
                            BrickData::Solid(*uniform_solid_value.unwrap()),
                        );
                        debug_assert_eq!(
                            self.nodes.get(node_key).occupied_bits,
                            u64::MAX,
                            "Expected Leaf with uniform solid value to have u64::MAX value"
                        );
                        return true;
                    }

                    // Do not try to unite bricks into a uniform brick
                    // since contents are not solid, it is not unifyable
                    // into a 1x1x1 brick ( that's equivalent to a solid brick )
                    if self.brick_dim == 1 {
                        return false;
                    }

                    // Try to unite bricks into a Uniform parted brick
                    let mut unified_brick_data =
                        vec![empty_marker::<PaletteIndexValues>(); self.brick_dim.pow(3) as usize];
                    let mut is_leaf_uniform = true;
                    const BRICK_CELL_SIZE: usize = BOX_NODE_DIMENSION;
                    let superbrick_size = self.brick_dim as f32 * BOX_NODE_DIMENSION as f32;
                    'brick_process: for x in 0..self.brick_dim {
                        for y in 0..self.brick_dim {
                            for z in 0..self.brick_dim {
                                let cell_start =
                                    V3c::new(x as f32, y as f32, z as f32) * BRICK_CELL_SIZE as f32;
                                let ref_sectant =
                                    offset_sectant(&cell_start, superbrick_size) as usize;
                                let pos_in_child =
                                    cell_start - SECTANT_OFFSET_LUT[ref_sectant] * superbrick_size;
                                let ref_voxel = match &bricks[ref_sectant] {
                                    BrickData::Empty => empty_marker(),
                                    BrickData::Solid(voxel) => *voxel,
                                    BrickData::Parted(brick) => {
                                        brick[flat_projection(
                                            pos_in_child.x as usize,
                                            pos_in_child.y as usize,
                                            pos_in_child.z as usize,
                                            self.brick_dim as usize,
                                        )]
                                    }
                                };

                                for cx in 0..BRICK_CELL_SIZE {
                                    for cy in 0..BRICK_CELL_SIZE {
                                        for cz in 0..BRICK_CELL_SIZE {
                                            if !is_leaf_uniform {
                                                break 'brick_process;
                                            }
                                            let pos = cell_start
                                                + V3c::new(cx as f32, cy as f32, cz as f32);
                                            let sectant =
                                                offset_sectant(&pos, superbrick_size) as usize;
                                            let pos_in_child =
                                                pos - SECTANT_OFFSET_LUT[sectant] * superbrick_size;

                                            is_leaf_uniform &= match &bricks[sectant] {
                                                BrickData::Empty => {
                                                    ref_voxel
                                                        == empty_marker::<PaletteIndexValues>()
                                                }
                                                BrickData::Solid(voxel) => ref_voxel == *voxel,
                                                BrickData::Parted(brick) => {
                                                    ref_voxel
                                                        == brick[flat_projection(
                                                            pos_in_child.x as usize,
                                                            pos_in_child.y as usize,
                                                            pos_in_child.z as usize,
                                                            self.brick_dim as usize,
                                                        )]
                                                }
                                            };
                                        }
                                    }
                                }
                                // All voxel are the same in this cell! set value in unified brick
                                unified_brick_data[flat_projection(
                                    x as usize,
                                    y as usize,
                                    z as usize,
                                    self.brick_dim as usize,
                                )] = ref_voxel;
                            }
                        }
                    }

                    // bricks can be represented as a uniform parted brick matrix!
                    if is_leaf_uniform {
                        unified_brick = BrickData::Parted(unified_brick_data);
                        simplified = true;
                    }

                    if !matches!(unified_brick, BrickData::Empty) {
                        self.nodes.get_mut(node_key).content =
                            NodeContent::UniformLeaf(unified_brick);
                    }

                    simplified
                }
                NodeContent::Internal => {
                    if 0 == self.nodes.get(node_key).occupied_bits
                        || matches!(self.nodes.get(node_key).children, NodeChildren::NoChildren)
                    {
                        if let NodeContent::Nothing = self.nodes.get(node_key).content {
                            return false;
                        }

                        self.nodes.get_mut(node_key).content = NodeContent::Nothing;
                        return true;
                    }

                    let child_keys = if let NodeChildren::Children(children) =
                        self.nodes.get(node_key).children
                    {
                        children
                    } else {
                        return false;
                    };

                    // Try to simplify each child of the node
                    if recursive {
                        for child_key in child_keys.iter() {
                            self.simplify(*child_key as usize, true);
                        }
                    }

                    for sectant in 1..BOX_NODE_CHILDREN_COUNT {
                        let child_key = child_keys[0] as usize;
                        if !self.nodes.key_is_valid(child_key)
                            || !matches!(
                                self.nodes.get(child_key).content,
                                NodeContent::UniformLeaf(BrickData::Solid(_))
                            )
                            || !self.compare_nodes(child_key, child_keys[sectant] as usize)
                        {
                            return false;
                        }
                    }

                    // All solid children are the same!
                    // make the current node a leaf, erase the children
                    debug_assert!(matches!(
                        self.nodes.get(child_keys[0] as usize).content,
                        NodeContent::Leaf(_) | NodeContent::UniformLeaf(_)
                    ));
                    self.nodes.swap(node_key, child_keys[0] as usize);

                    // Deallocate children
                    self.deallocate_children_of(node_key);
                    self.nodes.get_mut(node_key).children = NodeChildren::NoChildren;

                    // At this point there's no need to call simplify on the new leaf node
                    // because it's been attempted already on the data it copied from
                    true
                }
            }
        } else {
            // can't simplify invalid node
            false
        }
    }
}
