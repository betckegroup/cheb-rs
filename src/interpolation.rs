//! Support routines for interpolation

use itertools::izip;
use rlst::{
    Array, ArrayIteratorByValue, ArrayIteratorMut, DynArray, RlstScalar, Shape,
    UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut, UnsafeRandomAccessByValue,
    UnsafeRandomAccessMut,
};

use crate::{
    cheb_points::{self, barycentric_weights, get_cheb_points},
    types::Kind,
};

pub fn evaluate_1d<
    T: RlstScalar,
    ArrayImplEvalPoints,
    ArrayImplInterpNodes,
    ArrayImplInterpValues,
    ArrayImplInterpWeights,
    ArrayImplResult,
>(
    eval_points: &Array<ArrayImplEvalPoints, 1>,
    interp_nodes: &Array<ArrayImplInterpNodes, 1>,
    interp_values: &Array<ArrayImplInterpValues, 1>,
    interp_weights: &Array<ArrayImplInterpWeights, 1>,
    result: &mut Array<ArrayImplResult, 1>,
) where
    ArrayImplEvalPoints: UnsafeRandom1DAccessByValue<Item = T::Real> + Shape<1>,
    ArrayImplInterpNodes: UnsafeRandom1DAccessByValue<Item = T::Real> + Shape<1>,
    ArrayImplInterpValues:
        UnsafeRandomAccessByValue<1, Item = T> + UnsafeRandom1DAccessByValue<Item = T> + Shape<1>,
    ArrayImplInterpWeights: UnsafeRandom1DAccessByValue<Item = T::Real> + Shape<1>,
    ArrayImplResult:
        UnsafeRandomAccessMut<1, Item = T> + UnsafeRandom1DAccessMut<Item = T> + Shape<1>,
{
    let npoints = eval_points.len();
    let nnodes = interp_nodes.len();

    assert_eq!(result.len(), npoints);
    assert_eq!(nnodes, interp_values.len());
    assert_eq!(nnodes, interp_weights.len());

    let mut nominator = vec![T::zero(); npoints];
    let mut denominator = vec![T::zero(); npoints];
    let mut exact: Vec<i32> = vec![-1; npoints];

    for (interp_node, interp_weight, (interp_value_index, interp_value)) in izip!(
        interp_nodes.iter_value(),
        interp_weights.iter_value(),
        interp_values.iter_value().enumerate()
    ) {
        for (nom, denom, e, eval_point) in izip!(
            nominator.iter_mut(),
            denominator.iter_mut(),
            exact.iter_mut(),
            eval_points.iter_value()
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
        *result.get_mut([index]).unwrap() = interp_values.get_value([val_index as usize]).unwrap();
    }
}

pub fn cheb1d<T: RlstScalar, ArrayImplPoints, ArrayImplValues>(
    eval_points: &Array<ArrayImplPoints, 1>,
    interp_values: &Array<ArrayImplValues, 1>,
    kind: Kind,
) -> DynArray<T, 1>
where
    ArrayImplPoints: UnsafeRandom1DAccessByValue<Item = T::Real> + Shape<1>,
    ArrayImplValues:
        UnsafeRandomAccessByValue<1, Item = T> + UnsafeRandom1DAccessByValue<Item = T> + Shape<1>,
{
    let n = interp_values.len();
    let n_eval = eval_points.len();

    let (nodes, weights) = {
        (
            get_cheb_points::<T>(kind, n),
            barycentric_weights::<T>(kind, n),
        )
    };

    let mut output = DynArray::<T, 1>::from_shape([n_eval]);
    evaluate_1d(eval_points, &nodes, interp_values, &weights, &mut output);

    output
}

pub fn cheb2d<T: RlstScalar, ArrayImplPoints, ArrayImplValues>(
    eval_points: &Array<ArrayImplPoints, 2>,
    interp_values: &Array<ArrayImplValues, 2>,
    kind: Kind,
) -> DynArray<T, 1>
where
    ArrayImplPoints: UnsafeRandomAccessByValue<2, Item = T::Real>
        + UnsafeRandom1DAccessByValue<Item = T::Real>
        + Shape<2>,
    ArrayImplValues:
        UnsafeRandomAccessByValue<2, Item = T> + UnsafeRandom1DAccessByValue<Item = T> + Shape<2>,
{
    let n_shape = interp_values.shape();
    let n_eval = eval_points.shape()[0];

    let mut res = DynArray::<T, 1>::from_shape([n_eval]);

    let nodes_x = get_cheb_points::<T>(kind, n_shape[0]);
    let nodes_y = get_cheb_points::<T>(kind, n_shape[1]);

    let weights_x = barycentric_weights::<T>(kind, n_shape[0]);
    let weights_y = barycentric_weights::<T>(kind, n_shape[1]);

    let mut one_elem_res = DynArray::<T, 1>::from_shape([1]);
    let mut one_elem_point = DynArray::<T::Real, 1>::from_shape([1]);

    let mut one_d_results = DynArray::<T, 1>::from_shape([n_shape[0]]);

    for (point, out_value) in izip!(eval_points.col_iter(), res.iter_mut()) {
        *one_elem_point.get_mut([0]).unwrap() = point.get_value([1]).unwrap();

        for (interp_row, res) in izip!(interp_values.row_iter(), one_d_results.iter_mut()) {
            evaluate_1d(
                &one_elem_point,
                &nodes_y,
                &interp_row,
                &weights_y,
                &mut one_elem_res,
            );
            *res = one_elem_res.get_value([0]).unwrap();
        }

        *one_elem_point.get_mut([0]).unwrap() = point.get_value([0]).unwrap();

        evaluate_1d(
            &one_elem_point,
            &nodes_x,
            &one_d_results,
            &weights_x,
            &mut one_elem_res,
        );

        *out_value = *one_elem_res.data().first().unwrap();
    }

    res
}

#[cfg(test)]
mod test {
    use itertools::izip;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use rlst::DynArray;

    use crate::{cheb_points::get_cheb_points, interpolation::cheb2d, types::Kind};

    use super::cheb1d;

    #[test]
    fn test_cheb1d_first_kind() {
        let n = 10;
        let m = 1000;

        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let cheb_points = get_cheb_points::<f64>(Kind::First, n);
        let interp_values: DynArray<f64, 1> = cheb_points
            .iter_value()
            .map(|point| point.cos())
            .collect::<Vec<_>>()
            .into();

        let mut eval_points = DynArray::<f64, 1>::from_shape([m]);

        eval_points
            .iter_mut()
            .for_each(|point| *point = 2.0 * rng.random::<f64>() - 1.0);

        let eval_values = cheb1d(&eval_points, &interp_values, Kind::First);

        let max_error = izip!(eval_points.iter_value(), eval_values.iter_value())
            .map(|(point, value)| {
                let c = point.cos();
                (value - c).abs() / c.abs()
            })
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        assert!(max_error < 1E-9);
    }

    #[test]
    fn test_cheb1d_second_kind() {
        let n = 10;
        let m = 1000;

        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let cheb_points = get_cheb_points::<f64>(Kind::Second, n);
        let interp_values: DynArray<f64, 1> = cheb_points
            .iter_value()
            .map(|point| point.cos())
            .collect::<Vec<_>>()
            .into();

        let mut eval_points = DynArray::<f64, 1>::from_shape([m]);

        eval_points
            .iter_mut()
            .for_each(|point| *point = 2.0 * rng.random::<f64>() - 1.0);

        let eval_values = cheb1d(&eval_points, &interp_values, Kind::Second);

        let max_error = izip!(eval_points.iter_value(), eval_values.iter_value())
            .map(|(point, value)| {
                let c = point.cos();
                (value - c).abs() / c.abs()
            })
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        assert!(max_error < 1E-8);
    }

    #[test]
    fn test_cheb2d_second_kind() {
        let n_x = 10;
        let n_y = 15;

        let m = 1000;

        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let cheb_points_x = get_cheb_points::<f64>(Kind::Second, n_x);
        let cheb_points_y = get_cheb_points::<f64>(Kind::Second, n_y);

        let mut interp_values = DynArray::<f64, 2>::from_shape([n_x, n_y]);

        for (index_x, point_x) in cheb_points_x.iter_value().enumerate() {
            for (index_y, point_y) in cheb_points_y.iter_value().enumerate() {
                *interp_values.get_mut([index_x, index_y]).unwrap() =
                    point_x.cos() * point_y.sinh();
            }
        }

        let mut eval_points = DynArray::<f64, 2>::from_shape([2, m]);

        eval_points
            .iter_mut()
            .for_each(|point| *point = 2.0 * rng.random::<f64>() - 1.0);

        let eval_values = cheb2d(&eval_points, &interp_values, Kind::Second);

        let mut max_error = 0.0;

        for (point, actual) in izip!(eval_points.col_iter(), eval_values.iter_value()) {
            let exact = point[[0]].cos() * point[[1]].sinh();
            max_error = f64::max(max_error, (exact - actual).abs() / exact.abs());
        }

        println!("Error: {max_error}");
    }
}
