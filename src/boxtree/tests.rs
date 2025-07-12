mod iterate_tests {
    use crate::{
        boxtree::{
            iterate::execute_for_relevant_sectants, Albedo, BoxTree, BOX_NODE_CHILDREN_COUNT,
            BOX_NODE_DIMENSION,
        },
        make_tree,
        spatial::{math::vector::V3c, raytracing::step_sectant, Cube},
    };

    #[test]
    fn test_sibling_jump_to_internal_sibling() {
        const BRICK_DIM: u32 = 4;
        let mut tree: BoxTree = make_tree!(1024, BRICK_DIM);
        let start_position = V3c::new(507, 331, 0);
        let sibling_position = start_position + V3c::new(BRICK_DIM, 0, 0);
        let step_direction = V3c::new(1., 0., 0.);

        tree.insert(
            &start_position,
            &Albedo::default().with_red(100).with_alpha(255),
        )
        .expect("Expected to be able to update Boxtree");

        tree.insert(
            &sibling_position,
            &Albedo::default().with_red(100).with_alpha(255),
        )
        .expect("Expected to be able to update Boxtree");

        let start_node = tree
            .get_node_internal(
                BoxTree::<u32>::ROOT_NODE_KEY as usize,
                &mut Cube::root_bounds(1024.),
                &(start_position.into()),
            )
            .expect("Expected start node to exist");
        let node_stack_for_start_node =
            tree.get_access_stack_for(start_node, start_position.into());
        let start_sectant = node_stack_for_start_node
            .last()
            .expect("Expected start node access stack to be non-empty")
            .1;

        let sibling_node = tree
            .get_node_internal(
                BoxTree::<u32>::ROOT_NODE_KEY as usize,
                &mut Cube::root_bounds(1024.),
                &(sibling_position.into()),
            )
            .expect("Expected sibling node to exist");
        let node_stack_for_sibling_node =
            tree.get_access_stack_for(sibling_node, sibling_position.into());
        let sibling_sectant = node_stack_for_sibling_node
            .last()
            .expect("Expected sibling node access stack to be non-empty")
            .1;

        assert_eq!(
            start_node, sibling_node,
            "Start and sibling nodes should not differ"
        );
        assert_ne!(
            start_sectant, sibling_sectant,
            "Start and sibling node target sectants should differ"
        );

        // get target sectant of starting node
        let (queried_sibling_node, queried_sibling_sectant) = tree
            .get_sibling_by_position(start_node, step_direction, &(start_position.into()))
            .expect(
                &format!(
                    "Expected to be able to query sibling node in the direction {:?}",
                    step_direction
                )
                .to_string(),
            );
        assert_eq!(queried_sibling_node, sibling_node);
        assert_eq!(
            queried_sibling_sectant,
            step_sectant(start_sectant, step_direction)
        );

        // Check result with the other interface too
        let (queried_sibling_node, queried_sibling_sectant) = tree
            .get_sibling_by_stack(step_direction, &node_stack_for_start_node)
            .expect(
                &format!(
                    "Expected to be able to query sibling node in the direction {:?}",
                    step_direction
                )
                .to_string(),
            );

        assert_eq!(queried_sibling_node, sibling_node);
        assert_eq!(
            queried_sibling_sectant,
            step_sectant(start_sectant, step_direction)
        );
    }

    #[test]
    fn test_sibling_jump_to_hit_in_parent() {
        const BRICK_DIM: u32 = 4;
        let mut tree: BoxTree = make_tree!(1024, BRICK_DIM);
        let start_position = V3c::new(495, 331, 0);
        let sibling_position = V3c::new(496, 331, 0);
        let step_direction = V3c::new(1., 0., 0.);

        tree.insert(
            &start_position,
            &Albedo::default().with_red(100).with_alpha(255),
        )
        .expect("Expected to be able to update Boxtree");

        tree.insert(
            &sibling_position,
            &Albedo::default().with_red(100).with_alpha(255),
        )
        .expect("Expected to be able to update Boxtree");

        let start_node = tree
            .get_node_internal(
                BoxTree::<u32>::ROOT_NODE_KEY as usize,
                &mut Cube::root_bounds(1024.),
                &(start_position.into()),
            )
            .expect("Expected start node to exist");
        let node_stack_for_start_node =
            tree.get_access_stack_for(start_node, start_position.into());
        let start_sectant = node_stack_for_start_node
            .last()
            .expect("Expected start node access stack to be non-empty")
            .1;

        let sibling_node = tree
            .get_node_internal(
                BoxTree::<u32>::ROOT_NODE_KEY as usize,
                &mut Cube::root_bounds(1024.),
                &(sibling_position.into()),
            )
            .expect("Expected sibling node to exist");
        let node_stack_for_sibling_node =
            tree.get_access_stack_for(sibling_node, sibling_position.into());
        let sibling_sectant = node_stack_for_sibling_node
            .last()
            .expect("Expected sibling node access stack to be non-empty")
            .1;

        assert_ne!(
            start_node, sibling_node,
            "Start and sibling nodes should differ"
        );
        assert_ne!(
            start_sectant, sibling_sectant,
            "Start and sibling node target sectants should differ"
        );

        // get target sectant of starting node
        let (queried_sibling_node, queried_sibling_sectant) = tree
            .get_sibling_by_position(start_node, step_direction, &(start_position.into()))
            .expect(
                &format!(
                    "Expected to be able to query sibling node in the direction {:?}",
                    step_direction
                )
                .to_string(),
            );
        assert_eq!(queried_sibling_node, sibling_node);
        assert_eq!(
            queried_sibling_sectant,
            step_sectant(start_sectant, step_direction) - BOX_NODE_CHILDREN_COUNT as u8
        );

        // Check result with the other interface too
        let (queried_sibling_node, queried_sibling_sectant) = tree
            .get_sibling_by_stack(step_direction, &node_stack_for_start_node)
            .expect(
                &format!(
                    "Expected to be able to query sibling node in the direction {:?}",
                    step_direction
                )
                .to_string(),
            );

        assert_eq!(queried_sibling_node, sibling_node);
        assert_eq!(
            queried_sibling_sectant,
            step_sectant(start_sectant, step_direction) - BOX_NODE_CHILDREN_COUNT as u8
        );
    }

    #[test]
    fn test_sibling_jump_to_hit_in_root() {
        const BRICK_DIM: u32 = 4;
        let mut tree: BoxTree = make_tree!(1024, BRICK_DIM);
        let start_position = V3c::new(511, 331, 0);
        let sibling_position = V3c::new(512, 331, 0);
        let step_direction = V3c::new(1., 0., 0.);

        tree.insert(
            &start_position,
            &Albedo::default().with_red(100).with_alpha(255),
        )
        .expect("Expected to be able to update Boxtree");

        tree.insert(
            &sibling_position,
            &Albedo::default().with_red(100).with_alpha(255),
        )
        .expect("Expected to be able to update Boxtree");

        let start_node = tree
            .get_node_internal(
                BoxTree::<u32>::ROOT_NODE_KEY as usize,
                &mut Cube::root_bounds(1024.),
                &(start_position.into()),
            )
            .expect("Expected start node to exist");
        let node_stack_for_start_node =
            tree.get_access_stack_for(start_node, start_position.into());
        let start_sectant = node_stack_for_start_node
            .last()
            .expect("Expected start node access stack to be non-empty")
            .1;

        let sibling_node = tree
            .get_node_internal(
                BoxTree::<u32>::ROOT_NODE_KEY as usize,
                &mut Cube::root_bounds(1024.),
                &(sibling_position.into()),
            )
            .expect("Expected sibling node to exist");

        let _ = tree.get_access_stack_for(sibling_node, sibling_position.into());
        println!(
            "start vs sibling: {start_node}, {:?} <> {sibling_node}, {:?}",
            start_position, sibling_position
        );

        assert_ne!(
            start_node, sibling_node,
            "Start and sibling nodes should differ"
        );

        // get target sectant of starting node
        let (queried_sibling_node, queried_sibling_sectant) = tree
            .get_sibling_by_position(start_node, step_direction, &(start_position.into()))
            .expect(
                &format!(
                    "Expected to be able to query sibling node in the direction {:?}",
                    step_direction
                )
                .to_string(),
            );
        assert_eq!(queried_sibling_node, sibling_node);
        assert_eq!(
            queried_sibling_sectant,
            step_sectant(start_sectant, step_direction) - BOX_NODE_CHILDREN_COUNT as u8
        );

        // Check result with the other interface too
        let (queried_sibling_node, queried_sibling_sectant) = tree
            .get_sibling_by_stack(step_direction, &node_stack_for_start_node)
            .expect(
                &format!(
                    "Expected to be able to query sibling node in the direction {:?}",
                    step_direction
                )
                .to_string(),
            );

        assert_eq!(queried_sibling_node, sibling_node);
        assert_eq!(
            queried_sibling_sectant,
            step_sectant(start_sectant, step_direction) - BOX_NODE_CHILDREN_COUNT as u8
        );
    }

    #[test]
    fn test_sibling_jump_to_higher_level_leaf() {
        const BRICK_DIM: u32 = 4;
        let mut tree: BoxTree = make_tree!(1024, BRICK_DIM);
        let start_position = V3c::new(511, 0, 0);
        let sibling_position = V3c::new(512, 0, 0);
        let step_direction = V3c::new(1., 0., 0.);

        tree.insert(
            &start_position,
            &Albedo::default().with_red(100).with_alpha(255),
        )
        .expect("Expected to be able to update Boxtree");

        tree.insert_at_lod(
            &sibling_position,
            256,
            &Albedo::default().with_red(100).with_alpha(255),
        )
        .expect("Expected to be able to update Boxtree");

        let start_node = tree
            .get_node_internal(
                BoxTree::<u32>::ROOT_NODE_KEY as usize,
                &mut Cube::root_bounds(1024.),
                &(start_position.into()),
            )
            .expect("Expected start node to exist");
        let node_stack_for_start_node =
            tree.get_access_stack_for(start_node, start_position.into());
        let start_sectant = node_stack_for_start_node
            .last()
            .expect("Expected start node access stack to be non-empty")
            .1;

        let sibling_node = tree
            .get_node_internal(
                BoxTree::<u32>::ROOT_NODE_KEY as usize,
                &mut Cube::root_bounds(1024.),
                &(sibling_position.into()),
            )
            .expect("Expected sibling node to exist");
        let node_stack_for_sibling_node =
            tree.get_access_stack_for(sibling_node, sibling_position.into());
        let sibling_sectant = node_stack_for_sibling_node
            .last()
            .expect("Expected sibling node access stack to be non-empty")
            .1;

        assert_ne!(
            start_node, sibling_node,
            "Start and sibling nodes should differ"
        );
        assert_ne!(
            start_sectant, sibling_sectant,
            "Start and sibling node target sectants should differ"
        );

        // get target sectant of starting node
        let (queried_sibling_node, queried_sibling_sectant) = tree
            .get_sibling_by_position(start_node, step_direction, &(start_position.into()))
            .expect(
                &format!(
                    "Expected to be able to query sibling node in the direction {:?}",
                    step_direction
                )
                .to_string(),
            );
        assert_eq!(queried_sibling_node, sibling_node);
        assert_eq!(queried_sibling_sectant, BOX_NODE_CHILDREN_COUNT as u8);

        // Check result with the other interface too
        let (queried_sibling_node, queried_sibling_sectant) = tree
            .get_sibling_by_stack(step_direction, &node_stack_for_start_node)
            .expect(
                &format!(
                    "Expected to be able to query sibling node in the direction {:?}",
                    step_direction
                )
                .to_string(),
            );

        assert_eq!(queried_sibling_node, sibling_node);
        assert_eq!(queried_sibling_sectant, BOX_NODE_CHILDREN_COUNT as u8);
    }

    #[test]
    #[ignore = "FIXME: Undefined behavior, fix in #34"]
    fn test_sibling_jump_from_higher_level_leaf() {
        const BRICK_DIM: u32 = 4;
        let mut tree: BoxTree = make_tree!(1024, BRICK_DIM);
        let start_position = V3c::new(256, 0, 0);
        let sibling_position = V3c::new(512, 0, 0);
        let step_direction = V3c::new(1., 0., 0.);

        tree.insert_at_lod(
            &start_position,
            256,
            &Albedo::default().with_red(100).with_alpha(255),
        )
        .expect("Expected to be able to update Boxtree");

        tree.insert(
            &sibling_position,
            &Albedo::default().with_red(100).with_alpha(255),
        )
        .expect("Expected to be able to update Boxtree");

        let start_node = tree
            .get_node_internal(
                BoxTree::<u32>::ROOT_NODE_KEY as usize,
                &mut Cube::root_bounds(1024.),
                &(start_position.into()),
            )
            .expect("Expected start node to exist");
        let node_stack_for_start_node =
            tree.get_access_stack_for(start_node, start_position.into());

        assert!(tree
            .get_sibling_by_position(start_node, step_direction, &(start_position.into()),)
            .is_none());

        // Check result with the other interface too
        assert!(tree
            .get_sibling_by_stack(step_direction, &node_stack_for_start_node)
            .is_none());
    }

    #[test]
    fn test_sibling_jump_out_of_bounds() {
        const BRICK_DIM: u32 = 4;
        let mut tree: BoxTree = make_tree!(1024, BRICK_DIM);
        let start_position = V3c::new(1023, 331, 0);
        let step_direction = V3c::new(1., 0., 0.);

        tree.insert(
            &start_position,
            &Albedo::default().with_red(100).with_alpha(255),
        )
        .expect("Expected to be able to update Boxtree");

        let start_node = tree
            .get_node_internal(
                BoxTree::<u32>::ROOT_NODE_KEY as usize,
                &mut Cube::root_bounds(1024.),
                &(start_position.into()),
            )
            .expect("Expected start node to exist");
        let node_stack_for_start_node =
            tree.get_access_stack_for(start_node, start_position.into());

        assert!(tree
            .get_sibling_by_position(start_node, step_direction, &(start_position.into()),)
            .is_none());

        // Check result with the other interface too
        assert!(tree
            .get_sibling_by_stack(step_direction, &node_stack_for_start_node)
            .is_none());
    }

    #[test]
    fn test_sectant_execution_aligned_single_within() {
        let confines = Cube::root_bounds(400.);
        let update_size = 20;
        execute_for_relevant_sectants(
            &confines,
            &V3c::unit(0),
            update_size,
            |position_in_target, update_size_in_target, target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                assert_eq!(target_child_sectant, 0);
                assert_eq!(target_bounds.min_position, V3c::unit(0.));
                assert_eq!(update_size_in_target.x, update_size);
                assert_eq!(update_size_in_target.y, update_size);
                assert_eq!(update_size_in_target.z, update_size);
                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
            },
        );

        let execute_position = V3c::new(100, 0, 0);
        execute_for_relevant_sectants(
            &confines,
            &execute_position,
            update_size,
            |position_in_target, update_size_in_target, target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                assert_eq!(target_child_sectant, 1);
                assert_eq!(target_bounds.min_position, execute_position.into());
                assert_eq!(update_size_in_target.x, update_size);
                assert_eq!(update_size_in_target.y, update_size);
                assert_eq!(update_size_in_target.z, update_size);

                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
            },
        );
    }

    #[test]
    fn test_sectant_execution_aligned_single_bounds_smaller_position() {
        let confines = Cube {
            min_position: V3c::unit(400.),
            size: 400.,
        };
        let update_size = 20;
        execute_for_relevant_sectants(
            &confines,
            &V3c::unit(0),
            update_size,
            |position_in_target, update_size_in_target, target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                assert_eq!(target_child_sectant, 0);
                assert_eq!(target_bounds.min_position, V3c::unit(400.));
                assert_eq!(update_size_in_target.x, update_size);
                assert_eq!(update_size_in_target.y, update_size);
                assert_eq!(update_size_in_target.z, update_size);
                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
            },
        );

        let execute_position = V3c::new(100, 500, 0);
        execute_for_relevant_sectants(
            &confines,
            &execute_position,
            update_size,
            |position_in_target, update_size_in_target, _target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                assert_eq!(
                    target_bounds.min_position,
                    confines.min_position + V3c::new(0., 100., 0.)
                );
                assert_eq!(update_size_in_target.x, update_size);
                assert_eq!(update_size_in_target.y, update_size);
                assert_eq!(update_size_in_target.z, update_size);

                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
            },
        );
    }

    #[test]
    fn test_sectant_execution_single_target_with_smaller_position_aligned() {
        let confines = Cube {
            min_position: V3c::unit(400.),
            size: 400.,
        };
        let update_size = 450;
        let execute_position = V3c::unit(0);
        execute_for_relevant_sectants(
            &confines,
            &execute_position,
            update_size,
            |position_in_target, update_size_in_target, target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                assert_eq!(target_child_sectant, 0);
                assert_eq!(target_bounds.min_position, confines.min_position);
                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
                assert_eq!(
                    update_size_in_target.x,
                    update_size - confines.min_position.x as u32
                );
                assert_eq!(
                    update_size_in_target.y,
                    update_size - confines.min_position.y as u32
                );
                assert_eq!(
                    update_size_in_target.z,
                    update_size - confines.min_position.z as u32
                );
            },
        );
    }

    #[test]
    fn test_sectant_execution_single_target_with_smaller_position_unaligned() {
        let confines = Cube {
            min_position: V3c::unit(400.),
            size: 400.,
        };
        let update_size = 450;
        let y_offset_for_unalignment = 100;
        let execute_position = V3c::new(0, y_offset_for_unalignment, 0);
        execute_for_relevant_sectants(
            &confines,
            &execute_position,
            update_size,
            |position_in_target, update_size_in_target, target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                assert!(target_child_sectant == 0 || target_child_sectant == 4);
                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
                assert_eq!(
                    update_size_in_target.x,
                    update_size - confines.min_position.x as u32
                );
                assert!(
                    (update_size_in_target.y as f32 == target_bounds.size)
                        || update_size_in_target.y as f32
                            == ((update_size as f32 - confines.min_position.y
                                + y_offset_for_unalignment as f32)
                                % target_bounds.size)
                );
                assert_eq!(
                    update_size_in_target.z,
                    update_size - confines.min_position.z as u32
                );
            },
        );
    }

    #[test]
    fn test_sectant_execution_single_target_with_larger_position() {
        let confines = Cube {
            min_position: V3c::unit(400.),
            size: 400.,
        };
        let update_size = 100;
        let execute_position = V3c::new(0, 1000, 0);
        execute_for_relevant_sectants(
            &confines,
            &execute_position,
            update_size,
            |_position_in_target,
             _update_size_in_target,
             _target_child_sectant,
             &_target_bounds| {
                assert!(false);
            },
        );
    }

    #[test]
    fn test_sectant_execution_single_target_out_of_bounds() {
        let confines = Cube::root_bounds(400.);
        let update_size = 500;
        let execute_position = V3c::new(300, 300, 300);
        execute_for_relevant_sectants(
            &confines,
            &execute_position,
            update_size,
            |position_in_target, update_size_in_target, target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                assert_eq!(target_child_sectant, 63);
                assert_eq!(target_bounds.min_position, execute_position.into());
                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
                assert_eq!(update_size_in_target.x as f32, target_bounds.size);
                assert_eq!(update_size_in_target.y as f32, target_bounds.size);
                assert_eq!(update_size_in_target.z as f32, target_bounds.size);
            },
        );
    }

    #[test]
    fn test_sectant_execution_aligned_target_within() {
        let confines = Cube::root_bounds(400.);
        let update_size = 400;
        let execute_position = V3c::new(100, 0, 0);
        let mut visited_sectants: Vec<u8> = vec![];
        execute_for_relevant_sectants(
            &confines,
            &execute_position,
            update_size,
            |position_in_target, update_size_in_target, target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                assert!(!visited_sectants.contains(&target_child_sectant));
                visited_sectants.push(target_child_sectant);
                if 1 == target_child_sectant {
                    assert_eq!(target_bounds.min_position, execute_position.into());
                }
                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
                assert_eq!(update_size_in_target.x as f32, target_bounds.size);
                assert_eq!(update_size_in_target.y as f32, target_bounds.size);
                assert_eq!(update_size_in_target.z as f32, target_bounds.size);
            },
        );
        assert_eq!(
            visited_sectants.len(),
            (BOX_NODE_DIMENSION - 1) * BOX_NODE_DIMENSION * BOX_NODE_DIMENSION
        );
    }

    #[test]
    fn test_sectant_execution_aligned_target_out_of_bounds_smaller_position_larger_size() {
        let confines = Cube {
            min_position: V3c::unit(400.),
            size: 400.,
        };
        let update_size = 1000;
        let execute_position = V3c::new(500, 0, 0);
        let mut visited_sectants: Vec<u8> = vec![];
        execute_for_relevant_sectants(
            &confines,
            &execute_position,
            update_size,
            |position_in_target, update_size_in_target, target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                visited_sectants.push(target_child_sectant);
                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
                assert_eq!(update_size_in_target.x as f32, target_bounds.size);
                assert_eq!(update_size_in_target.y as f32, target_bounds.size);
                assert_eq!(update_size_in_target.z as f32, target_bounds.size);
            },
        );
        assert_eq!(
            visited_sectants.len(),
            (BOX_NODE_DIMENSION - 1) * BOX_NODE_DIMENSION * BOX_NODE_DIMENSION,
            "visited sectant mismatch! \n visited sectants: {:?}",
            visited_sectants
        );
    }

    #[test]
    fn test_sectant_execution_aligned_target_out_of_bounds() {
        let confines = Cube::root_bounds(400.);
        let update_size = 500;
        let execute_position = V3c::new(100, 0, 0);
        let mut visited_sectants: Vec<u8> = vec![];
        execute_for_relevant_sectants(
            &confines,
            &execute_position,
            update_size,
            |position_in_target, update_size_in_target, target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                visited_sectants.push(target_child_sectant);
                if 1 == target_child_sectant {
                    assert_eq!(target_bounds.min_position, execute_position.into());
                }
                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
                assert_eq!(update_size_in_target.x as f32, target_bounds.size);
                assert_eq!(update_size_in_target.y as f32, target_bounds.size);
                assert_eq!(update_size_in_target.z as f32, target_bounds.size);
            },
        );
        assert_eq!(
            visited_sectants.len(),
            (BOX_NODE_DIMENSION - 1) * BOX_NODE_DIMENSION * BOX_NODE_DIMENSION
        );
    }

    #[test]
    fn test_sectant_execution_unaligned_target_within() {
        let confines = Cube::root_bounds(400.);
        let update_size = 210;
        let execute_position = V3c::new(100, 0, 0);
        let mut visited_sectants: Vec<u8> = vec![];
        execute_for_relevant_sectants(
            &confines,
            &execute_position,
            update_size,
            |position_in_target, update_size_in_target, target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                visited_sectants.push(target_child_sectant);
                if 1 == target_child_sectant {
                    assert_eq!(target_bounds.min_position, execute_position.into());
                }
                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
                assert!(
                    update_size_in_target.x as f32 == target_bounds.size
                        || update_size_in_target.x as f32
                            == (update_size as f32 % target_bounds.size)
                );
                assert!(
                    update_size_in_target.y as f32 == target_bounds.size
                        || update_size_in_target.y as f32
                            == (update_size as f32 % target_bounds.size)
                );
                assert!(
                    update_size_in_target.z as f32 == target_bounds.size
                        || update_size_in_target.z as f32
                            == (update_size as f32 % target_bounds.size)
                );
            },
        );
        assert_eq!(visited_sectants.len(), (BOX_NODE_DIMENSION - 1).pow(3));
    }

    #[test]
    fn test_sectant_execution_unaligned_target_out_of_bounds() {
        let confines = Cube::root_bounds(400.);
        let update_size = 510;
        let execute_position = V3c::new(100, 0, 0);
        let mut visited_sectants: Vec<u8> = vec![];
        execute_for_relevant_sectants(
            &confines,
            &execute_position,
            update_size,
            |position_in_target, update_size_in_target, target_child_sectant, &target_bounds| {
                assert!(confines.contains(&V3c::from(
                    position_in_target + update_size_in_target - V3c::unit(1)
                )));
                visited_sectants.push(target_child_sectant);
                if 1 == target_child_sectant {
                    assert_eq!(target_bounds.min_position, execute_position.into());
                }
                assert_eq!(
                    target_bounds.size,
                    confines.size / BOX_NODE_DIMENSION as f32
                );
                assert_eq!(update_size_in_target.x as f32, target_bounds.size);
                assert_eq!(update_size_in_target.y as f32, target_bounds.size);
                assert_eq!(update_size_in_target.z as f32, target_bounds.size);
            },
        );
        assert_eq!(
            visited_sectants.len(),
            (BOX_NODE_DIMENSION - 1) * BOX_NODE_DIMENSION * BOX_NODE_DIMENSION
        );
    }
}

mod mipmap_tests {
    use crate::boxtree::{Albedo, BoxTree, MIPResamplingMethods, V3c, BOX_NODE_CHILDREN_COUNT};

    #[test]
    fn test_mixed_mip_lvl1() {
        let red: Albedo = 0xFF0000FF.into();
        let green: Albedo = 0x00FF00FF.into();
        let mix: Albedo = (
            // Gamma corrected values follow mip = ((a^2 + b^2) / 2).sqrt()
            (((255_f32.powf(2.) / 2.).sqrt() as u32) << 16)
                | (((255_f32.powf(2.) / 2.).sqrt() as u32) << 24)
                | 0x000000FF
        )
        .into();

        let mut tree: BoxTree = BoxTree::new(4, 1).ok().unwrap();
        tree.auto_simplify = false;
        tree.albedo_mip_map_resampling_strategy()
            .switch_albedo_mip_maps(true)
            .set_method_at(1, MIPResamplingMethods::BoxFilter);
        tree.insert(&V3c::new(0, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 0, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 1, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 1, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(1, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(1, 0, 1), &green)
            .expect("boxtree insert");

        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 0))
            .albedo()
            .is_some());
        assert_eq!(
            mix,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 0))
                .albedo()
                .unwrap()
        );
    }

    #[test]
    fn test_mixed_mip_lvl1_where_dim_is_32() {
        let red: Albedo = 0xFF0000FF.into();
        let green: Albedo = 0x00FF00FF.into();
        let mix: Albedo = (
            // Gamma corrected values follow mip = ((a^2 + b^2) / 2).sqrt()
            (((255_f32.powf(2.) / 2.).sqrt() as u32) << 16)
                | (((255_f32.powf(2.) / 2.).sqrt() as u32) << 24)
                | 0x000000FF
        )
        .into();

        let mut tree: BoxTree = BoxTree::new(128, 32).ok().unwrap();
        tree.auto_simplify = false;
        tree.albedo_mip_map_resampling_strategy()
            .switch_albedo_mip_maps(true)
            .set_method_at(1, MIPResamplingMethods::BoxFilter);
        tree.insert(&V3c::new(126, 126, 126), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(126, 126, 127), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(126, 127, 126), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(126, 127, 127), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(127, 126, 126), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(127, 126, 127), &green)
            .expect("boxtree insert");

        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(31, 31, 31))
            .albedo()
            .is_some());
        assert_eq!(
            mix,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(31, 31, 31))
                .albedo()
                .unwrap()
        );
    }

    #[test]
    fn test_simple_solid_mip_lvl2_where_dim_is_2() {
        let red: Albedo = 0xFF0000FF.into();

        let mut tree: BoxTree = BoxTree::new(8, 2).ok().unwrap();
        tree.auto_simplify = false;
        tree.albedo_mip_map_resampling_strategy()
            .switch_albedo_mip_maps(true)
            .set_method_at(1, MIPResamplingMethods::BoxFilter);
        tree.insert(&V3c::new(0, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 0, 1), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 1, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 1, 1), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(1, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(1, 0, 1), &red)
            .expect("boxtree insert");

        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 0))
            .albedo()
            .is_some());
        assert_eq!(
            red,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 0))
                .albedo()
                .unwrap()
        );
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 1))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 1, 0))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 1, 1))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 0, 0))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 0, 1))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 1, 0))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 1, 1))
            .albedo()
            .is_none());
    }

    #[test]
    fn test_mixed_mip_lvl2_where_dim_is_2() {
        let red: Albedo = 0xFF0000FF.into();
        let green: Albedo = 0x00FF00FF.into();
        let mix: Albedo = (
            // Gamma corrected values follow mip = ((a^2 + b^2) / 2).sqrt()
            (((255_f32.powf(2.) / 2.).sqrt() as u32) << 16)
                | (((255_f32.powf(2.) / 2.).sqrt() as u32) << 24)
                | 0x000000FF
        )
        .into();

        let mut tree: BoxTree = BoxTree::new(8, 2).ok().unwrap();
        tree.auto_simplify = false;
        tree.albedo_mip_map_resampling_strategy()
            .switch_albedo_mip_maps(true)
            .set_method_at(1, MIPResamplingMethods::BoxFilter);
        tree.insert(&V3c::new(0, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 0, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 1, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 1, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(1, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(1, 0, 1), &green)
            .expect("boxtree insert");

        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 0))
            .albedo()
            .is_some());
        assert_eq!(
            mix,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 0))
                .albedo()
                .unwrap()
        );
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 1))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 1, 0))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 1, 1))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 0, 0))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 0, 1))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 1, 0))
            .albedo()
            .is_none());
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 1, 1))
            .albedo()
            .is_none());
    }

    #[test]
    fn test_mixed_mip_lvl2_where_dim_is_4() {
        let red: Albedo = 0xFF0000FF.into();
        let green: Albedo = 0x00FF00FF.into();
        let blue: Albedo = 0x0000FFFF.into();

        let mut tree: BoxTree = BoxTree::new(64, 4).ok().unwrap();
        tree.auto_simplify = false;
        tree.albedo_mip_map_resampling_strategy()
            .switch_albedo_mip_maps(true)
            .set_method_at(1, MIPResamplingMethods::BoxFilter)
            .set_method_at(2, MIPResamplingMethods::BoxFilter);
        tree.insert(&V3c::new(0, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 0, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 1, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 1, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(1, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(1, 0, 1), &green)
            .expect("boxtree insert");

        tree.insert(&V3c::new(16, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(16, 0, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(16, 1, 0), &blue)
            .expect("boxtree insert");
        tree.insert(&V3c::new(16, 1, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(17, 1, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(17, 0, 1), &blue)
            .expect("boxtree insert");

        // For child position 0,0,0
        let rg_mix: Albedo = (
            // Gamma corrected values follow mip = ((a^2 + b^2) / 2).sqrt()
            (((255_f32.powf(2.) / 2.).sqrt() as u32) << 16)
                | (((255_f32.powf(2.) / 2.).sqrt() as u32) << 24)
                | 0x000000FF
        )
        .into();
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(0, &V3c::new(0, 0, 0))
            .albedo()
            .is_some());
        assert_eq!(
            rg_mix,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(0, &V3c::new(0, 0, 0))
                .albedo()
                .unwrap()
        );

        // For child position 16,0,0
        let rgb_mix: Albedo = (
            // Gamma corrected values follow mip = ((a^2 + b^2) / 2).sqrt()
            (((255_f32.powf(2.) / 3.).sqrt() as u32) << 8)
                | (((255_f32.powf(2.) / 3.).sqrt() as u32) << 16)
                | (((255_f32.powf(2.) / 3.).sqrt() as u32) << 24)
                | 0x000000FF
        )
        .into();
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(1, &V3c::new(0, 0, 0))
            .albedo()
            .is_some());
        assert_eq!(
            rgb_mix,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(1, &V3c::new(0, 0, 0))
                .albedo()
                .unwrap()
        );

        // root mip position 0,0,0
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 0))
            .albedo()
            .is_some());
        assert_eq!(
            rg_mix,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 0))
                .albedo()
                .unwrap()
        );

        // root mip position 16,0,0
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 0, 0))
            .albedo()
            .is_some());
        assert_eq!(
            rgb_mix,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 0, 0))
                .albedo()
                .unwrap()
        );
    }

    #[test]
    fn test_mixed_mip_regeneration_lvl2_where_dim_is_4() {
        let red: Albedo = 0xFF0000FF.into();
        let green: Albedo = 0x00FF00FF.into();
        let blue: Albedo = 0x0000FFFF.into();

        let mut tree: BoxTree = BoxTree::new(64, 4).ok().unwrap();
        tree.auto_simplify = false;
        tree.insert(&V3c::new(0, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 0, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 1, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(0, 1, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(1, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(1, 0, 1), &green)
            .expect("boxtree insert");

        tree.insert(&V3c::new(16, 0, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(16, 0, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(16, 1, 0), &blue)
            .expect("boxtree insert");
        tree.insert(&V3c::new(16, 1, 1), &green)
            .expect("boxtree insert");
        tree.insert(&V3c::new(17, 1, 0), &red)
            .expect("boxtree insert");
        tree.insert(&V3c::new(17, 0, 1), &blue)
            .expect("boxtree insert");

        // Switch MIP maps on, calculate the correct values
        tree.albedo_mip_map_resampling_strategy()
            .switch_albedo_mip_maps(true)
            .set_method_at(1, MIPResamplingMethods::BoxFilter)
            .set_method_at(2, MIPResamplingMethods::BoxFilter)
            .recalculate_mips();

        // For child position 0,0,0
        let rg_mix: Albedo = (
            // Gamma corrected values follow mip = ((a^2 + b^2) / 2).sqrt()
            (((255_f32.powf(2.) / 2.).sqrt() as u32) << 16)
                | (((255_f32.powf(2.) / 2.).sqrt() as u32) << 24)
                | 0x000000FF
        )
        .into();
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(0, &V3c::new(0, 0, 0))
            .albedo()
            .is_some());
        assert_eq!(
            rg_mix,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(0, &V3c::new(0, 0, 0))
                .albedo()
                .unwrap()
        );

        // For child position 8,0,0
        let rgb_mix: Albedo = (
            // Gamma corrected values follow mip = ((a^2 + b^2) / 2).sqrt()
            (((255_f32.powf(2.) / 3.).sqrt() as u32) << 8)
                | (((255_f32.powf(2.) / 3.).sqrt() as u32) << 16)
                | (((255_f32.powf(2.) / 3.).sqrt() as u32) << 24)
                | 0x000000FF
        )
        .into();
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(1, &V3c::new(0, 0, 0))
            .albedo()
            .is_some());
        assert_eq!(
            rgb_mix,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(1, &V3c::new(0, 0, 0))
                .albedo()
                .unwrap()
        );

        // root mip position 0,0,0
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 0))
            .albedo()
            .is_some());
        assert_eq!(
            rg_mix,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(0, 0, 0))
                .albedo()
                .unwrap()
        );

        // root mip position 16,0,0
        assert!(tree
            .albedo_mip_map_resampling_strategy()
            .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 0, 0))
            .albedo()
            .is_some());
        assert_eq!(
            rgb_mix,
            *tree
                .albedo_mip_map_resampling_strategy()
                .sample_root_mip(BOX_NODE_CHILDREN_COUNT as u8, &V3c::new(1, 0, 0))
                .albedo()
                .unwrap()
        );
    }
}
