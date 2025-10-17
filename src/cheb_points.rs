//! Chebychev interpolation in 1 dimension

use core::f64;

use itertools::izip;
use rlst::{DynArray, RlstScalar};

use crate::types::Kind;

use num::{One, traits::FloatConst};

/// Return the n Chebychev points of type `kind` in [-1, 1].
///
/// - The Chebychev points are returned in ascending order.
/// - Chebychev points of the first kind include the end points. Chebychev points of the second kind don't.
/// - For Chebychev points of the second kind require `n > 1`.
pub fn get_cheb_points<T: RlstScalar>(kind: Kind, n: usize) -> DynArray<T::Real, 1> {
    let mut output = DynArray::<T::Real, 1>::from_shape([n]);
    let indices = (0..n).rev();

    let pi = num::cast::<_, T::Real>(f64::PI()).unwrap();

    match kind {
        Kind::First => {
            assert!(n > 0);
            let pi_div_two_n = pi / num::cast::<_, T::Real>(2 * n).unwrap();
            for (value, index) in izip!(output.iter_mut(), indices) {
                *value = <T::Real as RlstScalar>::cos(
                    num::cast::<_, T::Real>(2 * index + 1).unwrap() * pi_div_two_n,
                );
            }
        }
        Kind::Second => {
            assert!(n > 1);
            let pi_div_nm1 = pi / num::cast::<_, T::Real>(n - 1).unwrap();
            for (value, index) in izip!(output.iter_mut(), indices) {
                *value = <T::Real as RlstScalar>::cos(
                    num::cast::<_, T::Real>(index).unwrap() * pi_div_nm1,
                );
            }
        }
    }

    output
}

/// Return the barycentric weights required for barycentric interpolation
pub fn barycentric_weights<T: RlstScalar>(kind: Kind, n: usize) -> DynArray<T::Real, 1> {
    let mut output = DynArray::<T::Real, 1>::from_shape([n]);
    match kind {
        Kind::First => {
            assert!(n > 0);
            let indices = (0..n).rev();
            let mut pm_one = <T::Real as One>::one();
            let pi = num::cast::<_, T::Real>(f64::PI()).unwrap();
            let pi_div_two_n = pi / num::cast::<_, T::Real>(2 * n).unwrap();
            for (value, index) in izip!(output.iter_mut(), indices) {
                *value = pm_one
                    * <T::Real as RlstScalar>::sin(
                        num::cast::<_, T::Real>(2 * index + 1).unwrap() * pi_div_two_n,
                    );
                pm_one = -pm_one;
            }
        }
        Kind::Second => {
            assert!(n > 1);
            let mut pm_one = <T::Real as One>::one();
            for value in output.iter_mut() {
                *value = pm_one;
                pm_one = -pm_one;
            }
            *output.data_mut().first_mut().unwrap() *= num::cast(0.5).unwrap();
            *output.data_mut().last_mut().unwrap() *= num::cast(0.5).unwrap();
        }
    }
    output
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use super::*;
    use crate::types::Kind;

    #[test]
    fn test_cheb_points_first_kind() {
        let n = 5;

        let points = get_cheb_points::<f64>(Kind::First, n);

        for (index, &value) in points.data().iter().rev().enumerate() {
            assert_relative_eq!(
                value,
                f64::cos((2 * index + 1) as f64 * (f64::PI()) / (2.0 * n as f64)),
                epsilon = 1E-13
            );
        }
    }

    #[test]
    fn test_cheb_points_second_kind() {
        let n = 5;

        let values = get_cheb_points::<f64>(Kind::Second, n);

        for (index, &value) in values.data().iter().rev().enumerate() {
            assert_relative_eq!(
                value,
                f64::cos((index) as f64 * (f64::PI()) / (n - 1) as f64),
                epsilon = 1E-13
            );
        }
    }
}
