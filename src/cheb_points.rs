//! Chebychev interpolation in 1 dimension

use core::f64;

use itertools::izip;
use rlst::RlstScalar;

use crate::types::Kind;

use num::traits::FloatConst;

/// Return the n Chebychev points of type `kind` in [-1, 1].
///
/// - The Chebychev points are returned in ascending order.
/// - Chebychev points of the first kind include the end points. Chebychev points of the second kind don't.
/// - For Chebychev points of the second kind require `n > 1`.
pub fn cheb_points<T: RlstScalar>(kind: Kind, output: &mut [T]) {
    let n = output.len();
    let indices = (0..n).rev();

    let pi = num::cast::<_, T>(f64::PI()).unwrap();

    match kind {
        Kind::First => {
            assert!(n > 0);
            let pi_div_two_n = pi / num::cast::<_, T>(2 * n).unwrap();
            for (value, index) in izip!(output.iter_mut(), indices) {
                *value = T::cos(num::cast::<_, T>(2 * index + 1).unwrap() * pi_div_two_n);
            }
        }
        Kind::Second => {
            assert!(n > 1);
            let pi_div_nm1 = pi / num::cast::<_, T>(n - 1).unwrap();
            for (value, index) in izip!(output.iter_mut(), indices) {
                *value = T::cos(num::cast::<_, T>(index).unwrap() * pi_div_nm1);
            }
        }
    }
}

/// Return the barycentric weights required for barycentric interpolation
pub fn barycentric_weights<T: RlstScalar>(kind: Kind, output: &mut [T]) {
    let n = output.len();
    match kind {
        Kind::First => {
            assert!(n > 0);
            let indices = (0..n).rev();
            let mut pm_one = T::one();
            let pi = num::cast::<_, T>(f64::PI()).unwrap();
            let pi_div_two_n = pi / num::cast::<_, T>(2 * n).unwrap();
            for (value, index) in izip!(output.iter_mut(), indices) {
                *value = pm_one * T::sin(num::cast::<_, T>(2 * index + 1).unwrap() * pi_div_two_n);
                pm_one = -pm_one;
            }
        }
        Kind::Second => {
            assert!(n > 1);
            let mut pm_one = T::one();
            for value in output.iter_mut() {
                *value = pm_one;
                pm_one = -pm_one;
            }
            *output.first_mut().unwrap() *= num::cast(0.5).unwrap();
            *output.last_mut().unwrap() *= num::cast(0.5).unwrap();
        }
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use super::*;
    use crate::types::Kind;

    #[test]
    fn test_cheb_points_first_kind() {
        let n = 5;

        let mut values = vec![0 as f64; n];

        cheb_points::<f64>(Kind::First, &mut values);

        for (index, &value) in values.iter().rev().enumerate() {
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

        let mut values = vec![0 as f64; n];

        cheb_points::<f64>(Kind::Second, &mut values);

        for (index, &value) in values.iter().rev().enumerate() {
            assert_relative_eq!(
                value,
                f64::cos((index) as f64 * (f64::PI()) / (n - 1) as f64),
                epsilon = 1E-13
            );
        }
    }
}
