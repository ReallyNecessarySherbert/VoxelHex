mod cache;
pub(crate) mod upload_queue;

use crate::{
    boxtree::{BOX_NODE_CHILDREN_COUNT, BoxTree, VoxelData, V3c},
    object_pool::empty_marker,
    raytracing::bevy::types::BoxTreeGPUView,
};
use bevy::render::{
    render_resource::{Buffer, ShaderSize, encase::internal::WriteInto},
    renderer::RenderQueue,
};
use std::ops::Range;

pub(crate) fn boxtree_properties<T: VoxelData>(tree: &BoxTree<T>) -> u32 {
    (tree.brick_dim & 0x0000FFFF) | ((tree.mip_map_strategy.is_enabled() as u32) << 16)
}

/// Invalidates view to be rebuilt on the size needed by bricks and nodes
pub(crate) fn re_evaluate_view_size(view: &mut BoxTreeGPUView) {
    // Decide if there's enough space to host the required number of nodes
    let nodes_needed_overall = view
        .data_handler
        .upload_targets
        .node_upload_queue
        .iter()
        .skip(view.data_handler.upload_state.node_upload_progress)
        .filter(|item| {
            !view
                .data_handler
                .upload_targets
                .node_key_vs_meta_index
                .contains_right(&item.node_key)
        })
        .count()
        + view
            .data_handler
            .upload_targets
            .node_key_vs_meta_index
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
    let bricks_needed_overall = view
        .data_handler
        .upload_targets
        .brick_upload_queue
        .iter()
        .skip(view.data_handler.upload_state.brick_upload_progress)
        .filter(|item| {
            !view
                .data_handler
                .upload_targets
                .brick_ownership
                .contains_right(&item.ownership)
        })
        .count()
        + view.data_handler.upload_targets.brick_ownership.len()
        + nodes_needed_overall;
    let rebuild_bricks = bricks_needed_overall > view.data_handler.bricks_in_view;
    if rebuild_bricks {
        let new_brick_count = (bricks_needed_overall as f32 * 1.1) as usize;
        view.data_handler.bricks_in_view = new_brick_count;
        view.data_handler
            .upload_targets
            .brick_positions
            .resize(new_brick_count, V3c::default());
    }

    debug_assert!(
        rebuild_nodes || rebuild_bricks,
        "Expected view to be too small while calling size evaluation!",
    );
    view.resize = true;
}

//##############################################################################
//    █████████  ███████████  █████  █████
//   ███░░░░░███░░███░░░░░███░░███  ░░███
//  ███     ░░░  ░███    ░███ ░███   ░███
// ░███          ░██████████  ░███   ░███
// ░███    █████ ░███░░░░░░   ░███   ░███
// ░░███  ░░███  ░███         ░███   ░███
//  ░░█████████  █████        ░░████████
//   ░░░░░░░░░  ░░░░░          ░░░░░░░░

//  █████   ███   █████ ███████████   █████ ███████████ ██████████
// ░░███   ░███  ░░███ ░░███░░░░░███ ░░███ ░█░░░███░░░█░░███░░░░░█
//  ░███   ░███   ░███  ░███    ░███  ░███ ░   ░███  ░  ░███  █ ░
//  ░███   ░███   ░███  ░██████████   ░███     ░███     ░██████
//  ░░███  █████  ███   ░███░░░░░███  ░███     ░███     ░███░░█
//   ░░░█████░█████░    ░███    ░███  ░███     ░███     ░███ ░   █
//     ░░███ ░░███      █████   █████ █████    █████    ██████████
//      ░░░   ░░░      ░░░░░   ░░░░░ ░░░░░    ░░░░░    ░░░░░░░░░░
//##############################################################################

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
