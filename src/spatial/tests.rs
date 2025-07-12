#[cfg(test)]
mod vector_tests {
    use crate::spatial::V3c;

    #[test]
    fn test_cross_product() {
        let a = V3c::new(3., 0., 2.);
        let b = V3c::new(-1., 4., 2.);
        let cross = a.cross(b);
        assert!(cross.x == -8.);
        assert!(cross.y == -8.);
        assert!(cross.z == 12.);
    }
}

#[cfg(test)]
mod sectant_tests {
    use crate::spatial::{math::offset_sectant, V3c};

    #[cfg(feature = "raytracing")]
    use crate::{boxtree::BOX_NODE_CHILDREN_COUNT, spatial::raytracing::step_sectant};

    #[test]
    fn test_hash_region() {
        assert_eq!(offset_sectant(&V3c::new(0.0, 0.0, 0.0), 12.0), 0);
        assert_eq!(offset_sectant(&V3c::new(3.0, 0.0, 0.0), 12.0), 1);
        assert_eq!(offset_sectant(&V3c::new(0.0, 3.0, 0.0), 12.0), 4);
        assert_eq!(offset_sectant(&V3c::new(0.0, 0.0, 3.0), 12.0), 16);
        assert_eq!(offset_sectant(&V3c::new(10.0, 10.0, 10.0), 12.0), 63);
    }

    #[test]
    #[cfg(feature = "raytracing")]
    fn test_step_sectant_internal() {
        let bounds_size = 40.;
        let start_sectant = offset_sectant(&V3c::new(0., 10., 0.), bounds_size);
        assert_eq!(4, start_sectant);
        assert_eq!(5, step_sectant(start_sectant, V3c::new(1., 0., 0.)));
        assert_eq!(0, step_sectant(start_sectant, V3c::new(0., -1., 0.)));
        assert_eq!(8, step_sectant(start_sectant, V3c::new(0., 1., 0.)));
        assert_eq!(9, step_sectant(start_sectant, V3c::new(1., 1., 0.)));
        assert_eq!(25, step_sectant(start_sectant, V3c::new(1., 1., 1.)));
    }

    #[test]
    #[cfg(feature = "raytracing")]
    fn test_step_sectant_wrap_around() {
        let bounds_size = 40.;
        let start_sectant = offset_sectant(&V3c::new(0., 0., 0.), bounds_size);
        assert_eq!(0, start_sectant);
        assert_eq!(
            3,
            step_sectant(start_sectant, V3c::new(-1., 0., 0.)) as usize - BOX_NODE_CHILDREN_COUNT
        );
        assert_eq!(
            12,
            step_sectant(start_sectant, V3c::new(0., -1., 0.)) as usize - BOX_NODE_CHILDREN_COUNT
        );
        assert_eq!(
            48,
            step_sectant(start_sectant, V3c::new(0., 0., -1.)) as usize - BOX_NODE_CHILDREN_COUNT
        );
        assert_eq!(
            63,
            step_sectant(start_sectant, V3c::new(-1., -1., -1.)) as usize - BOX_NODE_CHILDREN_COUNT
        );
    }
}

#[cfg(test)]
mod bitmask_tests {

    use crate::boxtree::V3c;
    use crate::spatial::math::{flat_projection, offset_sectant};
    use std::collections::HashSet;

    #[test]
    fn test_flat_projection() {
        const DIMENSION: usize = 10;
        assert!(0 == flat_projection(0, 0, 0, DIMENSION));
        assert!(DIMENSION == flat_projection(10, 0, 0, DIMENSION));
        assert!(DIMENSION == flat_projection(0, 1, 0, DIMENSION));
        assert!(DIMENSION * DIMENSION == flat_projection(0, 0, 1, DIMENSION));
        assert!(DIMENSION * DIMENSION * 4 == flat_projection(0, 0, 4, DIMENSION));
        assert!((DIMENSION * DIMENSION * 4) + 3 == flat_projection(3, 0, 4, DIMENSION));
        assert!(
            (DIMENSION * DIMENSION * 4) + (DIMENSION * 2) + 3
                == flat_projection(3, 2, 4, DIMENSION)
        );

        let mut number_coverage = HashSet::new();
        for x in 0..DIMENSION {
            for y in 0..DIMENSION {
                for z in 0..DIMENSION {
                    let address = flat_projection(x, y, z, DIMENSION);
                    assert!(!number_coverage.contains(&address));
                    number_coverage.insert(address);
                }
            }
        }
    }

    #[test]
    fn test_bitmap_flat_projection_exact_size_match() {
        assert_eq!(0, offset_sectant(&V3c::new(0., 0., 0.), 4.));
        assert_eq!(32, offset_sectant(&V3c::new(0., 0., 2.), 4.));
        assert_eq!(63, offset_sectant(&V3c::new(3., 3., 3.), 4.));
    }

    #[test]
    fn test_bitmap_flat_projection_greater_dimension() {
        assert_eq!(0, offset_sectant(&V3c::new(0., 0., 0.), 10.));
        assert_eq!(32, offset_sectant(&V3c::new(0., 0., 5.), 10.));
        assert_eq!(42, offset_sectant(&V3c::new(5., 5., 5.), 10.));
        assert_eq!(63, offset_sectant(&V3c::new(9., 9., 9.), 10.));
    }

    #[test]
    fn test_bitmap_flat_projection_smaller_dimension() {
        assert_eq!(0, offset_sectant(&V3c::new(0., 0., 0.), 2.));
        assert_eq!(2, offset_sectant(&V3c::new(1., 0., 0.), 2.));
        assert_eq!(8, offset_sectant(&V3c::new(0., 1., 0.), 2.));
        assert_eq!(10, offset_sectant(&V3c::new(1., 1., 0.), 2.));
        assert_eq!(32, offset_sectant(&V3c::new(0., 0., 1.), 2.));
        assert_eq!(34, offset_sectant(&V3c::new(1., 0., 1.), 2.));
        assert_eq!(40, offset_sectant(&V3c::new(0., 1., 1.), 2.));
        assert_eq!(42, offset_sectant(&V3c::new(1., 1., 1.), 2.));
    }
}
