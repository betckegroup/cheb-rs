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

#[cfg(test)]
mod test {
    use itertools::izip;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use rlst::DynArray;

    use crate::{cheb_points::get_cheb_points, types::Kind};

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
}
