mod loader;
mod ui;

use crate::ui::input::CameraPosition;
use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    log::LogPlugin,
    prelude::*,
    render::view::RenderLayers,
    window::WindowPlugin,
};
use bevy_egui::{egui, EguiContext, EguiPlugin, EguiPrimaryContextPass};
use egui_plot::{Line, Plot, PlotPoints};
use bevy_lunex::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_pkv::PkvStore;
use voxelhex::raytracing::VhxViewSet;

#[derive(Resource)]
struct FpsGraphState {
    capturing: bool,
    capture_count: usize,
}

impl Default for FpsGraphState {
    fn default() -> Self {
        Self {
            capturing: true,
            capture_count: 1000,
        }
    }
}

fn main() {
    let preferences = init_preferences_cache();
    let ui_state = ui::UiState::new(&preferences);

    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(FpsHistory::default())
        .init_resource::<FpsGraphState>()
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
                        title: "Whisp - Press g to hide/show UI".to_string(),
                        ..default()
                    }),
                    ..default()
                })
                .set(LogPlugin {
                    level: bevy::log::Level::INFO,
                    filter: "wgpu=error,naga=warn,bevy_render::camera::camera=error".to_string(),
                    custom_layer: |_| None,
                }),
            voxelhex::raytracing::RenderBevyPlugin::<u32>::new(),
            FrameTimeDiagnosticsPlugin::default(),
            PanOrbitCameraPlugin,
            UiLunexPlugins,
            EguiPlugin::default(),
        ))
        .add_systems(Startup, (ui::layout::setup, setup))
        .add_systems(
            Startup,
            (ui::behavior::setup, ui::input::setup_mouse_action).after(crate::ui::layout::setup),
        )
        .add_systems(
            Startup,
            loader::load_last_loaded_model.after(ui::behavior::setup),
        )
        .add_systems(
            Update,
            (
                ui::behavior::update,
                ui::input::mouse_action_cleanup,
                ui::input::handle_settings_update,
                ui::input::handle_camera_update,
                ui::input::handle_world_interaction_block_by_ui,
                record_fps_system,
            ),
        )
        .add_systems(EguiPrimaryContextPass, fps_graph_system)
        .add_systems(
            FixedUpdate,
            (
                ui::input::init_camera.run_if(run_once),
                ui::behavior::handle_model_load_animation,
                ui::behavior::update_performance_stats,
                ui::behavior::update_output_resolution_and_view_dist,
                ui::behavior::handle_ui_hidden,
                loader::observe_file_drop,
                loader::handle_model_load_finished,
            ),
        )
        .insert_resource(ui_state)
        .insert_resource(preferences)
        .insert_resource(VhxViewSet::new())
        .add_observer(ui::behavior::resolution_changed_observer)
        .add_observer(ui::behavior::settings_changed_observer)
        .run();
}

fn init_preferences_cache() -> PkvStore {
    let mut pkv = PkvStore::new("MinistryOfVoxelAffairs", "Whisp");

    if pkv.get::<CameraPosition>("camera_position").is_err() {
        pkv.set("CameraPosition", &CameraPosition::baked_model_pose())
            .expect("Failed to store default value: camera_position");
    }
    if pkv.get::<String>("camera_locked").is_err() {
        pkv.set("camera_locked", &"false")
            .expect("Failed to store default value: camera_locked");
    }
    if pkv.get::<String>("output_resolution_width").is_err() {
        pkv.set("output_resolution_width", &"1920")
            .expect("Failed to store default value: output_resolution_width");
    }
    if pkv.get::<String>("output_resolution_height").is_err() {
        pkv.set("output_resolution_height", &"1080")
            .expect("Failed to store default value: output_resolution_height");
    }
    if pkv.get::<String>("fov").is_err() {
        pkv.set("fov", &"50")
            .expect("Failed to store default value: fov");
    }
    if pkv.get::<String>("viewport_resolution_width").is_err() {
        pkv.set("viewport_resolution_width", &"100")
            .expect("Failed to store default value: viewport_resolution_width");
    }
    if pkv.get::<String>("viewport_resolution_height").is_err() {
        pkv.set("viewport_resolution_height", &"100")
            .expect("Failed to store default value: viewport_resolution_height");
    }
    if pkv.get::<String>("view_distance").is_err() {
        pkv.set("view_distance", &"512")
            .expect("Failed to store default value: view_distance");
    }
    if pkv.get::<String>("ui_hidden").is_err() {
        pkv.set("ui_hidden", &"false")
            .expect("Failed to store default value: ui_hidden");
    }
    if pkv.get::<String>("shortcuts_hidden").is_err() {
        pkv.set("shortcuts_hidden", &"false")
            .expect("Failed to store default value: shortcuts_hidden");
    }
    if pkv.get::<String>("output_resolution_linked").is_err() {
        pkv.set("output_resolution_linked", &"false")
            .expect("Failed to store default value: output_resolution_linked");
    }
    if pkv.get::<String>("viewport_resolution_linked").is_err() {
        pkv.set("viewport_resolution_linked", &"false")
            .expect("Failed to store default value: viewport_resolution_linked");
    }
    pkv
}

fn setup(mut commands: Commands) {
    commands.spawn((
        bevy::prelude::Camera {
            is_active: true,
            ..default()
        },
        PanOrbitCamera {
            focus: Vec3::new(0., 100., 0.),
            ..default()
        },
    ));

    commands.spawn((
        Camera {
            order: 1,
            ..default()
        },
        Camera2d::default(),
        UiSourceCamera::<0>,
        RenderLayers::from_layers(&[0, 1]),
    ));

    commands.spawn((
        Camera {
            order: 2,
            ..default()
        },
        Camera2d::default(),
        EguiContext::default(),
    ));
}

#[derive(Resource, Default)]
struct FpsHistory {
    values: Vec<f64>,
}

fn record_fps_system(
    diagnostics: Res<DiagnosticsStore>,
    mut history: ResMut<FpsHistory>,
    state: Res<FpsGraphState>,
) {
    if !state.capturing {
        return;
    }

    if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
        if let Some(value) = fps.smoothed() {
            if history.values.len() >= state.capture_count {
                history.values.remove(0);
            }
            history.values.push(value);
        }
    }
}

fn fps_graph_system(
    mut contexts: Query<(&mut EguiContext, &Camera)>,
    history: Res<FpsHistory>,
    mut state: ResMut<FpsGraphState>,
) {
    for (mut context, camera) in contexts.iter_mut() {
        if camera.order == 2 {
            let ctx = context.get_mut();
            egui::Window::new("Performance Graph").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut state.capturing, "Capture Data");
                    ui.add(
                        egui::Slider::new(&mut state.capture_count, 1..=100000)
                            .text("History Length"),
                    );
                });

                let (avg_fps, avg_ms) = if !history.values.is_empty() {
                    let sum: f64 = history.values.iter().sum();
                    let avg = sum / history.values.len() as f64;
                    (avg, 1000.0 / avg)
                } else {
                    (0.0, 0.0)
                };
                ui.label(format!(
                    "Average: {:.1} FPS ({:.2} ms)",
                    avg_fps, avg_ms
                ));
                ui.separator();

                let points: PlotPoints = history
                    .values
                    .iter()
                    .enumerate()
                    .map(|(i, &fps)| [i as f64, 1000.0/fps])
                    .collect();

                let line = Line::new("ms", points);

                Plot::new("ms_plot")
                    .view_aspect(2.0)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .allow_boxed_zoom(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(line);
                    });
            });
            return;
        }
    }
}


