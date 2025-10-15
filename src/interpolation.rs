//! Support routines for interpolation

use itertools::izip;
use rlst::RlstScalar;

pub fn evaluate_1d_with_stride<T: RlstScalar>(
    eval_points: &[T::Real],
    interp_nodes: &[T::Real],
    interp_values: &[T],
    interp_weights: &[T::Real],
    result: &mut [T],
    stride: usize,
) {
    let npoints = eval_points.len();
    let nnodes = interp_nodes.len();

    assert_eq!(result.len(), npoints);
    assert_eq!(nnodes, interp_values.len());
    assert_eq!(nnodes, interp_weights.len());

    let mut nominator = vec![T::zero(); npoints];
    let mut denominator = vec![T::zero(); npoints];
    let mut exact = vec![-1; npoints];

    for (&interp_node, &interp_weight, (interp_value_index, &interp_value)) in izip!(
        interp_nodes,
        interp_weights,
        interp_values.iter().enumerate().step_by(stride)
    ) {
        for (nom, denom, e, &eval_point) in izip!(
            nominator.iter_mut(),
            denominator.iter_mut(),
            exact.iter_mut(),
            eval_points.iter()
        ) {
            let diff = T::from_real(eval_point - interp_node);
            if diff == T::zero() {
                *e = interp_value_index as i32;
            }
            let inv_diff = T::from_real(interp_weight) / diff;
            *nom += inv_diff * interp_value;
            *denom += inv_diff;
        }
        for (&nom, &denom, res) in izip!(nominator.iter(), denominator.iter(), result.iter_mut()) {
            *res = nom / denom;
        }
    }

    for (index, &val_index) in exact.iter().enumerate().filter(|(_, v)| **v != -1) {
        result[index] = interp_values[val_index as usize];
    }
}
