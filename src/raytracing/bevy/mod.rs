mod data;
mod pipeline;
pub mod types;
mod view;

pub use crate::raytracing::bevy::types::{
    BoxTreeGPUHost, BoxTreeGPUView, BoxTreeSpyGlass, RenderBevyPlugin, VhxViewSet, Viewport,
};
use crate::{
    boxtree::{Albedo, V3c, VoxelData},
    raytracing::bevy::{
        data::upload_queue::{handle_changes, rebuild},
        pipeline::prepare_bind_groups,
        types::{VhxLabel, VhxRenderNode, VhxRenderPipeline},
        view::{handle_resolution_updates_main_world, handle_resolution_updates_render_world},
    },
    spatial::Cube,
};
use bevy::{
    app::{App, Plugin},
    ecs::prelude::IntoScheduleConfigs,
    prelude::{Commands, ExtractSchedule, FixedUpdate, Res, ResMut, Vec4},
    render::{
        Render, RenderApp, RenderSet, extract_resource::ExtractResourcePlugin,
        render_graph::RenderGraph,
    },
};
use std::sync::{RwLockReadGuard, RwLockWriteGuard, TryLockResult};

impl From<Vec4> for Albedo {
    fn from(vec: Vec4) -> Self {
        Albedo::default()
            .with_red((vec.x * 255.).min(255.) as u8)
            .with_green((vec.y * 255.).min(255.) as u8)
            .with_blue((vec.z * 255.).min(255.) as u8)
            .with_alpha((vec.w * 255.).min(255.) as u8)
    }
}

impl From<Albedo> for Vec4 {
    fn from(color: Albedo) -> Self {
        Vec4::new(
            color.r as f32 / 255.,
            color.g as f32 / 255.,
            color.b as f32 / 255.,
            color.a as f32 / 255.,
        )
    }
}

//##############################################################################
//  █████   █████ █████ ██████████ █████   ███   █████  █████████  ██████████ ███████████
// ░░███   ░░███ ░░███ ░░███░░░░░█░░███   ░███  ░░███  ███░░░░░███░░███░░░░░█░█░░░███░░░█
//  ░███    ░███  ░███  ░███  █ ░  ░███   ░███   ░███ ░███    ░░░  ░███  █ ░ ░   ░███  ░
//  ░███    ░███  ░███  ░██████    ░███   ░███   ░███ ░░█████████  ░██████       ░███
//  ░░███   ███   ░███  ░███░░█    ░░███  █████  ███   ░░░░░░░░███ ░███░░█       ░███
//   ░░░█████░    ░███  ░███ ░   █  ░░░█████░█████░    ███    ░███ ░███ ░   █    ░███
//     ░░███      █████ ██████████    ░░███ ░░███     ░░█████████  ██████████    █████
//      ░░░      ░░░░░ ░░░░░░░░░░      ░░░   ░░░       ░░░░░░░░░  ░░░░░░░░░░    ░░░░░
//##############################################################################
impl Default for VhxViewSet {
    fn default() -> Self {
        Self::new()
    }
}

impl VhxViewSet {
    pub fn new() -> Self {
        Self {
            changed: true,
            views: vec![],
        }
    }

    /// Returns the number of views
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// True if the viewset is empty
    pub fn is_empty(&self) -> bool {
        0 == self.len()
    }

    /// Provides a view for immutable access; Blocks until view is available
    pub fn view(&self, index: usize) -> Option<RwLockReadGuard<'_, BoxTreeGPUView>> {
        if index < self.views.len() {
            Some(
                self.views[index]
                    .read()
                    .expect("Expected to be able to lock data view for read access"),
            )
        } else {
            None
        }
    }

    /// Tries to provide a view for immutable access; Fails if view is not available
    pub fn try_view(
        &self,
        index: usize,
    ) -> Option<TryLockResult<RwLockReadGuard<'_, BoxTreeGPUView>>> {
        if index < self.views.len() {
            Some(self.views[index].try_read())
        } else {
            None
        }
    }

    /// Provides a view for mutable access; Blocks until view is available
    pub fn view_mut(&mut self, index: usize) -> Option<RwLockWriteGuard<'_, BoxTreeGPUView>> {
        if index < self.views.len() {
            Some(
                self.views[index]
                    .write()
                    .expect("Expected to be able to lock data view for write access"),
            )
        } else {
            None
        }
    }

    /// Tries to provide a view for mutable access; Fails if view is not available
    pub fn try_view_mut(
        &mut self,
        index: usize,
    ) -> Option<TryLockResult<RwLockWriteGuard<'_, BoxTreeGPUView>>> {
        if index < self.views.len() {
            Some(self.views[index].try_write())
        } else {
            None
        }
    }

    /// Empties the viewset erasing all contained views
    pub fn clear(&mut self) {
        self.views.clear();
        self.changed = true;
    }
}

//##############################################################################
//  ███████████  █████       █████  █████   █████████  █████ ██████   █████
// ░░███░░░░░███░░███       ░░███  ░░███   ███░░░░░███░░███ ░░██████ ░░███
//  ░███    ░███ ░███        ░███   ░███  ███     ░░░  ░███  ░███░███ ░███
//  ░██████████  ░███        ░███   ░███ ░███          ░███  ░███░░███░███
//  ░███░░░░░░   ░███        ░███   ░███ ░███    █████ ░███  ░███ ░░██████
//  ░███         ░███      █ ░███   ░███ ░░███  ░░███  ░███  ░███  ░░█████
//  █████        ███████████ ░░████████   ░░█████████  █████ █████  ░░█████
// ░░░░░        ░░░░░░░░░░░   ░░░░░░░░     ░░░░░░░░░  ░░░░░ ░░░░░    ░░░░░
//##############################################################################
impl<T> Default for RenderBevyPlugin<T>
where
    T: Default + Clone + Eq + VoxelData + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> RenderBevyPlugin<T>
where
    T: Default + Clone + Eq + VoxelData + Send + Sync + 'static,
{
    pub fn new() -> Self {
        RenderBevyPlugin {
            dummy: std::marker::PhantomData,
        }
    }
}

/// Handles data sync between Bevy main(CPU) world and rendering world
/// Logic here should be as lightweight as possible!
pub(crate) fn sync_from_main_world(
    mut commands: Commands,
    mut world: ResMut<bevy::render::MainWorld>,
    render_world_viewset: Option<Res<VhxViewSet>>,
) {
    let Some(mut main_world_viewset) = world.get_resource_mut::<VhxViewSet>() else {
        return; // Nothing to do without a viewset..
    };

    if render_world_viewset.is_none() || main_world_viewset.changed {
        commands.insert_resource(main_world_viewset.clone());
        main_world_viewset.changed = false;
        return;
    }

    if main_world_viewset.is_empty() {
        return; // Nothing else to do without views..
    }

    let Some(render_world_viewset) = render_world_viewset else {
        // This shouldn't happen ?! In case main world already has an available viewset
        // where the view images are updated, there should already be a viewset in the render world
        commands.insert_resource(main_world_viewset.clone());
        return;
    };

    if render_world_viewset.view(0).unwrap().new_images_ready
        && !main_world_viewset.view(0).unwrap().new_images_ready
    {
        main_world_viewset.view_mut(0).unwrap().new_images_ready = true;
    }
}

fn handle_viewport_position_updates<T: VoxelData>(
    mut tree_gpu_host: Option<Res<BoxTreeGPUHost<T>>>,
    mut viewset: Option<ResMut<VhxViewSet>>,
) {
    if let (Some(tree_host), Some(viewset)) = (tree_gpu_host.as_mut(), viewset.as_mut()) {
        if viewset.is_empty() {
            return; // Nothing to do without views..
        }
        let Some(mut view) = viewset.view_mut(0) else {
            return;
        };

        // There have been movement lately
        if view.spyglass.viewport.origin_delta != V3c::unit(0.) {
            // Check if the new origin fits into the brick slot
            debug_assert!(
                view.brick_slot.contains(
                    &(view.spyglass.viewport.origin - view.spyglass.viewport.origin_delta)
                ),
                "Expected old viewport position to be inside old brick slot"
            );

            if !view.brick_slot.contains(&view.spyglass.viewport.origin) {
                view.data_handler.upload_range = Cube {
                    min_position: view.spyglass.viewport.origin
                        - V3c::unit(view.spyglass.viewport.frustum.z / 2.),
                    size: view.spyglass.viewport.frustum.z,
                };
                rebuild::<T>(
                    &tree_host.tree,
                    &view.spyglass.viewport.origin.clone(),
                    view.spyglass.viewport.frustum.z,
                    &mut view.data_handler.upload_targets,
                );

                view.data_handler.upload_state.brick_upload_progress = 0;
                view.data_handler.upload_state.node_upload_progress = 0;
                view.brick_slot =
                    Cube::brick_slot_for(&view.spyglass.viewport.origin, tree_host.tree.brick_dim);
            }

            view.spyglass.viewport.origin_delta = V3c::unit(0.);
            // update viewport matrices
            let resolution = view.resolution;
            view.spyglass.viewport.update_matrices(resolution);
        }
    }
}

impl<T: VoxelData> Plugin for RenderBevyPlugin<T> {
    fn build(&self, app: &mut App) {
        app.add_plugins((ExtractResourcePlugin::<BoxTreeGPUHost<T>>::default(),));
        app.add_systems(
            FixedUpdate,
            (
                handle_resolution_updates_main_world,
                handle_viewport_position_updates::<T>,
                handle_changes::<T>,
            ),
        );
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(ExtractSchedule, sync_from_main_world);
        render_app.add_systems(
            Render,
            (
                handle_changes::<T>.in_set(RenderSet::PrepareAssets),
                prepare_bind_groups.in_set(RenderSet::PrepareBindGroups),
                handle_resolution_updates_render_world,
            ),
        );
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(VhxLabel, VhxRenderNode { ready: false });
        render_graph.add_node_edge(VhxLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<VhxRenderPipeline>();
    }
}
