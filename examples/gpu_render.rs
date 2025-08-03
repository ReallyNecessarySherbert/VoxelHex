#[cfg(feature = "bevy_wgpu")]
use bevy::{prelude::*, window::WindowPlugin};

#[cfg(feature = "bevy_wgpu")]
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

#[cfg(feature = "bevy_wgpu")]
use voxelhex::{
    boxtree::{Albedo, BoxTree, BoxTreeEntry, V3c, V3cf32},
    raytracing::{BoxTreeGPUHost, Ray, VhxViewSet, Viewport},
};

#[cfg(feature = "bevy_wgpu")]
use image::{ImageBuffer, Rgb};

#[cfg(feature = "bevy_wgpu")]
const DISPLAY_RESOLUTION: [u32; 2] = [1024, 768];

#[cfg(feature = "bevy_wgpu")]
const BRICK_DIMENSION: u32 = 32;

#[cfg(feature = "bevy_wgpu")]
const TREE_SIZE: u32 = 128;

#[cfg(feature = "bevy_wgpu")]
fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    // uncomment for unthrottled FPS
                    present_mode: bevy::window::PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            }),
            voxelhex::raytracing::RenderBevyPlugin::<u32>::new(),
            PanOrbitCameraPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, set_viewport_for_camera)
        .add_systems(Update, handle_zoom)
        .run();
}

#[cfg(feature = "bevy_wgpu")]
fn setup(mut commands: Commands, images: ResMut<Assets<Image>>) {
    // fill boxtree with data
    let mut tree: BoxTree = voxelhex::boxtree::BoxTree::new(TREE_SIZE, BRICK_DIMENSION)
        .ok()
        .unwrap();

    for x in 0..TREE_SIZE {
        for y in 0..TREE_SIZE {
            for z in 0..TREE_SIZE {
                if ((x < (TREE_SIZE / 4) || y < (TREE_SIZE / 4) || z < (TREE_SIZE / 4))
                    && (0 == x % 2 && 0 == y % 4 && 0 == z % 2))
                    || ((TREE_SIZE / 2) <= x && (TREE_SIZE / 2) <= y && (TREE_SIZE / 2) <= z)
                {
                    let r = if 0 == x % (TREE_SIZE / 4) {
                        (x as f32 / TREE_SIZE as f32 * 255.) as u32
                    } else {
                        128
                    };
                    let g = if 0 == y % (TREE_SIZE / 4) {
                        (y as f32 / TREE_SIZE as f32 * 255.) as u32
                    } else {
                        128
                    };
                    let b = if 0 == z % (TREE_SIZE / 4) {
                        (z as f32 / TREE_SIZE as f32 * 255.) as u32
                    } else {
                        128
                    };
                    tree.insert(
                        &V3c::new(x, y, z),
                        &Albedo::default()
                            .with_red(r as u8)
                            .with_green(g as u8)
                            .with_blue(b as u8)
                            .with_alpha(255),
                    )
                    .ok()
                    .unwrap();
                    assert_eq!(
                        tree.get(&V3c::new(x, y, z)),
                        BoxTreeEntry::Visual(
                            &Albedo::default()
                                .with_red(r as u8)
                                .with_green(g as u8)
                                .with_blue(b as u8)
                                .with_alpha(255)
                        )
                    );
                }
            }
        }
    }

    let mut host = BoxTreeGPUHost::new(tree);
    let mut views = VhxViewSet::new();
    let view_index = host.create_new_view(
        &mut views,
        Viewport::new(
            V3c {
                x: 0.,
                y: 100.,
                z: 0.,
            },
            V3c {
                x: 0.,
                y: 0.,
                z: -10.,
            },
            V3c::new(10., 10., 1024.),
            50.,
        ),
        DISPLAY_RESOLUTION,
        images,
    );
    commands.insert_resource(host);
    let mut display = Sprite::from_image(views.view(view_index).unwrap().output_texture().clone());
    display.custom_size = Some(Vec2::new(
        DISPLAY_RESOLUTION[0] as f32,
        DISPLAY_RESOLUTION[1] as f32,
    ));
    commands.spawn(display);
    commands.insert_resource(views);
    commands.spawn((
        Camera {
            is_active: false,
            ..default()
        },
        PanOrbitCamera {
            focus: Vec3::new(0., 100., -10.),
            ..default()
        },
    ));
    commands.spawn(Camera2d::default());
    println!("Takes a while to create the compute pipeline, please hang on! Thanks");
    #[cfg(debug_assertions)]
    println!(
        "WARNING! Example ran in debug mode, might take a while to load, and will be pretty slow.."
    );
}

#[cfg(feature = "bevy_wgpu")]
fn direction_from_cam(cam: &PanOrbitCamera) -> Option<V3cf32> {
    if let Some(radius) = cam.radius {
        Some(
            V3c::new(
                radius / 2. + cam.yaw.unwrap().sin() * radius,
                radius + cam.pitch.unwrap().sin() * radius * 2.,
                radius / 2. + cam.yaw.unwrap().cos() * radius,
            )
            .normalized(),
        )
    } else {
        None
    }
}

#[cfg(feature = "bevy_wgpu")]
fn set_viewport_for_camera(
    camera_query: Query<&mut PanOrbitCamera>,
    mut view_set: ResMut<VhxViewSet>,
) {
    let cam = camera_query
        .single()
        .expect("Expecet PanOrbitCamera to be available");
    if let Some(_) = cam.radius {
        let Some(mut tree_view) = view_set.view_mut(0) else {
            return; // Nothing to do without views!
        };
        tree_view
            .spyglass
            .viewport_mut()
            .set_viewport_origin(V3c::new(cam.focus.x, cam.focus.y, cam.focus.z));
        tree_view.spyglass.viewport_mut().direction = direction_from_cam(cam).unwrap();
    }
}

#[cfg(feature = "bevy_wgpu")]
fn handle_zoom(
    keys: Res<ButtonInput<KeyCode>>,
    tree: ResMut<BoxTreeGPUHost>,
    mut view_set: ResMut<VhxViewSet>,
    mut camera_query: Query<&mut PanOrbitCamera>,
) {
    let Some(mut view) = view_set.view_mut(0) else {
        return; // Nothing to do without views!
    };

    // Render the current view with CPU
    if keys.pressed(KeyCode::Tab) {
        // define light
        let diffuse_light_normal = V3c::new(0., -1., 1.).normalized();
        let mut img = ImageBuffer::new(DISPLAY_RESOLUTION[0], DISPLAY_RESOLUTION[1]);

        // cast each ray for a hit
        view.spyglass
            .viewport_mut()
            .update_matrices(DISPLAY_RESOLUTION);
        for x in 0..DISPLAY_RESOLUTION[0] {
            for y in 0..DISPLAY_RESOLUTION[1] {
                let actual_y_in_image = DISPLAY_RESOLUTION[1] - y - 1;

                // Calculate NDC (Normalized Device Coordinates) from pixel coordinates
                let ndc_x = (x as f32 + 0.5) / DISPLAY_RESOLUTION[0] as f32 * 2.0 - 1.0;
                let ndc_y = (y as f32 + 0.5) / DISPLAY_RESOLUTION[1] as f32 * 2.0 - 1.0;
                let ndc_near = Vec4::new(ndc_x, ndc_y, -1.0, 1.0); // near plane in NDC
                let ndc_far = Vec4::new(ndc_x, ndc_y, 1.0, 1.0); // far plane in NDC

                // Transform NDC coordinates to world space
                let world_near = view.spyglass.viewport().inverse_view_projection_matrix * ndc_near;
                let world_far = view.spyglass.viewport().inverse_view_projection_matrix * ndc_far;
                let world_near_pos = world_near.truncate() / world_near.w;
                let world_far_pos = world_far.truncate() / world_far.w;
                let ray_direction = (world_far_pos - world_near_pos).normalize();

                let ray = Ray {
                    origin: view.spyglass.viewport().origin(),
                    direction: V3cf32::from(ray_direction),
                };

                use std::io::Write;
                std::io::stdout().flush().ok().unwrap();

                if let Some(hit) = tree
                    .tree
                    .read()
                    .expect("Expected to be able to read Tree from GPU host")
                    .get_by_ray(&ray)
                {
                    let (data, _, normal) = hit;
                    //Because both vector should be normalized, the dot product should be 1*1*cos(angle)
                    //That means it is in range -1, +1, which should be accounted for
                    let diffuse_light_strength =
                        1. - (normal.dot(&diffuse_light_normal) / 2. + 0.5);
                    img.put_pixel(
                        x,
                        actual_y_in_image,
                        Rgb([
                            (data.albedo().unwrap().r as f32 * diffuse_light_strength) as u8,
                            (data.albedo().unwrap().g as f32 * diffuse_light_strength) as u8,
                            (data.albedo().unwrap().b as f32 * diffuse_light_strength) as u8,
                        ]),
                    );
                } else {
                    img.put_pixel(x, actual_y_in_image, Rgb([128, 128, 128]));
                }
            }
        }

        img.save("example_junk_cpu_render.png").ok().unwrap();
    }

    if keys.pressed(KeyCode::Home) {
        view.spyglass.viewport_mut().fov *= 1. + 0.09;
    }
    if keys.pressed(KeyCode::End) {
        view.spyglass.viewport_mut().fov *= 1. - 0.09;
    }

    let mut cam = camera_query
        .single_mut()
        .expect("Expecet PanOrbitCamera to be available");
    if keys.pressed(KeyCode::ShiftLeft) {
        cam.target_focus.y += 1.;
    }
    if keys.pressed(KeyCode::ControlLeft) {
        cam.target_focus.y -= 1.;
    }

    if keys.pressed(KeyCode::NumpadAdd) {
        view.spyglass.viewport_mut().frustum.z *= 1.01;
    }
    if keys.pressed(KeyCode::NumpadSubtract) {
        view.spyglass.viewport_mut().frustum.z *= 0.99;
    }
    if keys.pressed(KeyCode::F3) {
        println!("{:?}", view.spyglass.viewport());
    }

    if let Some(_) = cam.radius {
        let dir = direction_from_cam(&cam).unwrap();
        let dir = Vec3::new(dir.x, dir.y, dir.z);
        let right = dir.cross(Vec3::new(0., 1., 0.));
        if keys.pressed(KeyCode::KeyW) {
            cam.target_focus += dir;
        }
        if keys.pressed(KeyCode::KeyS) {
            cam.target_focus -= dir;
        }
        if keys.pressed(KeyCode::KeyA) {
            cam.target_focus -= right;
        }
        if keys.pressed(KeyCode::KeyD) {
            cam.target_focus += right;
        }
    }
}

#[cfg(not(feature = "bevy_wgpu"))]
fn main() {
    println!("You probably forgot to enable the bevy_wgpu feature!");
    //nothing to do when the feature is not enabled
}
