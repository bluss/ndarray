// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::{Add, Div, Mul};
use num_traits::{self, Zero, Float, FromPrimitive};

use crate::imp_prelude::*;
use crate::numeric_util;

use crate::{FoldWhile, Zip};

/// # Numerical Methods for Arrays
impl<A, S, D> ArrayBase<S, D>
    where S: Data<Elem=A>,
          D: Dimension,
{
    /// Return the sum of all elements in the array.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert_eq!(a.sum(), 10.);
    /// ```
    pub fn sum(&self) -> A
        where A: Clone + Add<Output=A> + num_traits::Zero,
    {
        if let Some(slc) = self.as_slice_memory_order() {
            return numeric_util::pairwise_sum(&slc);
        }
        if self.ndim() > 1 {
            let ax = self.dim.min_stride_axis(&self.strides);
            if self.len_of(ax) >= numeric_util::UNROLL_SIZE && self.stride_of(ax) == 1 {
                let partial_sums: Vec<_> =
                    self.lanes(ax).into_iter().map(|lane| lane.sum()).collect();
                return numeric_util::pure_pairwise_sum(&partial_sums);
            }
        }
        numeric_util::iterator_pairwise_sum(self.iter())
    }

    /// Return the sum of all elements in the array.
    ///
    /// *This method has been renamed to `.sum()` and will be deprecated in the
    /// next version.*
    // #[deprecated(note="renamed to `sum`", since="0.13")]
    pub fn scalar_sum(&self) -> A
        where A: Clone + Add<Output=A> + num_traits::Zero,
    {
        self.sum()
    }

    /// Return the product of all elements in the array.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert_eq!(a.product(), 24.);
    /// ```
    pub fn product(&self) -> A
        where A: Clone + Mul<Output=A> + num_traits::One,
    {
        if let Some(slc) = self.as_slice_memory_order() {
            return numeric_util::unrolled_fold(slc, A::one, A::mul);
        }
        let mut sum = A::one();
        for row in self.inner_rows() {
            if let Some(slc) = row.as_slice() {
                sum = sum * numeric_util::unrolled_fold(slc, A::one, A::mul);
            } else {
                sum = sum * row.iter().fold(A::one(), |acc, elt| acc * elt.clone());
            }
        }
        sum
    }

    /// Return sum along `axis`.
    ///
    /// ```
    /// use ndarray::{aview0, aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2., 3.],
    ///                [4., 5., 6.]]);
    /// assert!(
    ///     a.sum_axis(Axis(0)) == aview1(&[5., 7., 9.]) &&
    ///     a.sum_axis(Axis(1)) == aview1(&[6., 15.]) &&
    ///
    ///     a.sum_axis(Axis(0)).sum_axis(Axis(0)) == aview0(&21.)
    /// );
    /// ```
    ///
    /// **Panics** if `axis` is out of bounds.
    pub fn sum_axis(&self, axis: Axis) -> Array<A, D::Smaller>
        where A: Clone + Zero + Add<Output=A>,
              D: RemoveAxis,
    {
        let n = self.len_of(axis);
        if self.stride_of(axis) == 1 {
            // contiguous along the axis we are summing
            let mut res = Array::zeros(self.raw_dim().remove_axis(axis));
            Zip::from(&mut res)
                .and(self.lanes(axis))
                .apply(|sum, lane| *sum = lane.sum());
            res
        } else if n <= numeric_util::NAIVE_SUM_THRESHOLD {
            self.fold_axis(axis, A::zero(), |acc, x| acc.clone() + x.clone())
        } else {
            let (v1, v2) = self.view().split_at(axis, n / 2);
            v1.sum_axis(axis) + v2.sum_axis(axis)
        }
    }

    /// Return mean along `axis`.
    ///
    /// **Panics** if `axis` is out of bounds, if the length of the axis is
    /// zero and division by zero panics for type `A`, or if `A::from_usize()`
    /// fails for the axis length.
    ///
    /// ```
    /// use ndarray::{aview0, aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2., 3.],
    ///                [4., 5., 6.]]);
    /// assert!(
    ///     a.mean_axis(Axis(0)) == aview1(&[2.5, 3.5, 4.5]) &&
    ///     a.mean_axis(Axis(1)) == aview1(&[2., 5.]) &&
    ///
    ///     a.mean_axis(Axis(0)).mean_axis(Axis(0)) == aview0(&3.5)
    /// );
    /// ```
    pub fn mean_axis(&self, axis: Axis) -> Array<A, D::Smaller>
        where A: Clone + Zero + FromPrimitive + Add<Output=A> + Div<Output=A>,
              D: RemoveAxis,
    {
        let n = A::from_usize(self.len_of(axis)).expect("Converting axis length to `A` must not fail.");
        let sum = self.sum_axis(axis);
        sum / &aview0(&n)
    }

    /// Return variance along `axis`.
    ///
    /// The variance is computed using the [Welford one-pass
    /// algorithm](https://www.jstor.org/stable/1266577).
    ///
    /// The parameter `ddof` specifies the "delta degrees of freedom". For
    /// example, to calculate the population variance, use `ddof = 0`, or to
    /// calculate the sample variance, use `ddof = 1`.
    ///
    /// The variance is defined as:
    ///
    /// ```text
    ///               1       n
    /// variance = ――――――――   ∑ (xᵢ - x̅)²
    ///            n - ddof  i=1
    /// ```
    ///
    /// where
    ///
    /// ```text
    ///     1   n
    /// x̅ = ―   ∑ xᵢ
    ///     n  i=1
    /// ```
    ///
    /// and `n` is the length of the axis.
    ///
    /// **Panics** if `ddof` is less than zero or greater than `n`, if `axis`
    /// is out of bounds, or if `A::from_usize()` fails for any any of the
    /// numbers in the range `0..=n`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.],
    ///                [5., 6.]]);
    /// let var = a.var_axis(Axis(0), 1.);
    /// assert_eq!(var, aview1(&[4., 4.]));
    /// ```
    pub fn var_axis(&self, axis: Axis, ddof: A) -> Array<A, D::Smaller>
    where
        A: Float + FromPrimitive,
        D: RemoveAxis,
    {
        let zero = A::from_usize(0).expect("Converting 0 to `A` must not fail.");
        let n = A::from_usize(self.len_of(axis)).expect("Converting length to `A` must not fail.");
        assert!(
            !(ddof < zero || ddof > n),
            "`ddof` must not be less than zero or greater than the length of \
             the axis",
        );
        let dof = n - ddof;
        let mut mean = Array::<A, _>::zeros(self.dim.remove_axis(axis));
        let mut sum_sq = Array::<A, _>::zeros(self.dim.remove_axis(axis));
        for (i, subview) in self.axis_iter(axis).enumerate() {
            let count = A::from_usize(i + 1).expect("Converting index to `A` must not fail.");
            azip!(mut mean, mut sum_sq, x (subview) in {
                let delta = x - *mean;
                *mean = *mean + delta / count;
                *sum_sq = (x - *mean).mul_add(delta, *sum_sq);
            });
        }
        sum_sq.mapv_into(|s| s / dof)
    }

    /// Return standard deviation along `axis`.
    ///
    /// The standard deviation is computed from the variance using
    /// the [Welford one-pass algorithm](https://www.jstor.org/stable/1266577).
    ///
    /// The parameter `ddof` specifies the "delta degrees of freedom". For
    /// example, to calculate the population standard deviation, use `ddof = 0`,
    /// or to calculate the sample standard deviation, use `ddof = 1`.
    ///
    /// The standard deviation is defined as:
    ///
    /// ```text
    ///                    1       n
    /// stddev = sqrt ( ――――――――   ∑ (xᵢ - x̅)² )
    ///                 n - ddof  i=1
    /// ```
    ///
    /// where
    ///
    /// ```text
    ///     1   n
    /// x̅ = ―   ∑ xᵢ
    ///     n  i=1
    /// ```
    ///
    /// and `n` is the length of the axis.
    ///
    /// **Panics** if `ddof` is less than zero or greater than `n`, if `axis`
    /// is out of bounds, or if `A::from_usize()` fails for any any of the
    /// numbers in the range `0..=n`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.],
    ///                [5., 6.]]);
    /// let stddev = a.std_axis(Axis(0), 1.);
    /// assert_eq!(stddev, aview1(&[2., 2.]));
    /// ```
    pub fn std_axis(&self, axis: Axis, ddof: A) -> Array<A, D::Smaller>
    where
        A: Float + FromPrimitive,
        D: RemoveAxis,
    {
        self.var_axis(axis, ddof).mapv_into(|x| x.sqrt())
    }

    /// Return `true` if the arrays' elementwise differences are all within
    /// the given absolute tolerance, `false` otherwise.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting to the same shape isn’t possible.
    pub fn all_close<S2, E>(&self, rhs: &ArrayBase<S2, E>, tol: A) -> bool
        where A: Float,
              S2: Data<Elem=A>,
              E: Dimension,
    {
        !Zip::from(self)
            .and(rhs.broadcast_unwrap(self.raw_dim()))
            .fold_while((), |_, x, y| {
                if (*x - *y).abs() <= tol {
                    FoldWhile::Continue(())
                } else {
                    FoldWhile::Done(())
                }
            }).is_done()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::numeric_util::{NAIVE_SUM_THRESHOLD, UNROLL_SIZE};
    use self::{Array, s};
    use quickcheck::{QuickCheck, StdGen, TestResult};

    #[test]
    fn test_sum_value_does_not_depend_on_axis() {
        // `size` controls the length of the array of data
        // We set it to be randomly drawn between 0 and
        // a number larger than NAIVE_SUM_THRESHOLD * UNROLL_SIZE
        let rng = StdGen::new(
            rand::thread_rng(),
            5* (NAIVE_SUM_THRESHOLD * UNROLL_SIZE).pow(3)
        );
        let mut quickcheck = QuickCheck::new().gen(rng).tests(100);
        quickcheck.quickcheck(
           _sum_value_does_not_depend_on_axis
            as fn(
               Vec<f64>
            ) -> TestResult,
        );
    }

    fn _sum_value_does_not_depend_on_axis(xs: Vec<f64>) -> TestResult {
        // We want three axis of equal length - we drop some elements
        // to get the right number
        let axis_length = (xs.len() as f64).cbrt().floor() as usize;
        let xs = &xs[..axis_length.pow(3)];

        // We want to check that summing with respect to an axis
        // is independent from the specific underlying implementation of
        // pairwise sum, which is itself conditional on the arrangement
        // in memory of the array elements.
        // We will thus swap axes and compute the sum, in turn, with respect to
        // axes 0, 1 and 2, while making sure that mathematically the same
        // number should be spit out (because we are properly transposing before summing).
        if axis_length > 0 {
            let (a, b, c) = equivalent_arrays(xs.to_vec(), axis_length);

            let sum1 = a.sum_axis(Axis(0));
            let sum2 = b.sum_axis(Axis(1));
            let sum3 = c.sum_axis(Axis(2));

            let tol = 1e-10;
            let first = (sum2.clone() - sum1.clone()).iter().all(|x| x.abs() < tol);
            let second = (sum3.clone() - sum1.clone()).iter().all(|x| x.abs() < tol);
            let third = (sum3.clone() - sum2.clone()).iter().all(|x| x.abs() < tol);

            if first && second && third {
                TestResult::passed()
            } else {
                TestResult::failed()
            }
        } else {
            TestResult::passed()
        }
    }

    #[test]
    fn test_sum_value_does_not_depend_on_axis_with_discontinuous_array() {
        // `size` controls the length of the array of data
        // We set it to be randomly drawn between 0 and
        // a number larger than NAIVE_SUM_THRESHOLD * UNROLL_SIZE
        let rng = StdGen::new(
            rand::thread_rng(),
            5* (NAIVE_SUM_THRESHOLD * UNROLL_SIZE).pow(3)
        );
        let mut quickcheck = QuickCheck::new().gen(rng).tests(100);
        quickcheck.quickcheck(
            _sum_value_does_not_depend_on_axis_w_discontinuous_array
                as fn(
                Vec<f64>
            ) -> TestResult,
        );
    }

    fn _sum_value_does_not_depend_on_axis_w_discontinuous_array(xs: Vec<f64>) -> TestResult {
        // We want three axis of equal length - we drop some elements
        // to get the right number
        let axis_length = (xs.len() as f64).cbrt().floor() as usize;
        let xs = &xs[..axis_length.pow(3)];

        // We want to check that summing with respect to an axis
        // is independent from the specific underlying implementation of
        // pairwise sum, which is itself conditional on the arrangement
        // in memory of the array elements.
        // We will thus swap axes and compute the sum, in turn, with respect to
        // axes 0, 1 and 2, while making sure that mathematically the same
        // number should be spit out (because we are properly transposing before summing).
        if axis_length > 0 {
            let (a, b, c) = equivalent_arrays(xs.to_vec(), axis_length);

            let sum1 = a.slice(s![..;2, .., ..]).sum_axis(Axis(0));
            let sum2 = b.slice(s![.., ..;2, ..]).sum_axis(Axis(1));
            let sum3 = c.slice(s![.., .., ..;2]).sum_axis(Axis(2));

            let tol = 1e-10;
            let first = (sum2.clone() - sum1.clone()).iter().all(|x| x.abs() < tol);
            let second = (sum3.clone() - sum1.clone()).iter().all(|x| x.abs() < tol);
            let third = (sum3.clone() - sum2.clone()).iter().all(|x| x.abs() < tol);

            if first && second && third {
                TestResult::passed()
            } else {
                TestResult::failed()
            }
        } else {
            TestResult::passed()
        }
    }

    // Given a vector with axis_length^3 elements, it returns three arrays,
    // built using the vector elements, such that (mathematically):
    // a.sum_axis(Axis(0) == b.sum_axis(Axis(1)) == c.sum_axis(Axis(2))
    fn equivalent_arrays(xs: Vec<f64>, axis_length: usize) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        assert!(xs.len() == axis_length.pow(3));

        let a = Array::from_vec(xs)
            .into_shape((axis_length, axis_length, axis_length))
            .unwrap();
        assert!(a.is_standard_layout());

        let mut b = Array::zeros(a.raw_dim());
        assert!(b.is_standard_layout());
        for i in 0..axis_length {
            for j in 0..axis_length {
                for k in 0..axis_length {
                    b[(i, j, k)] = a[(j, i, k)].clone();
                }
            }
        }

        let mut c = Array::zeros(a.raw_dim());
        assert!(c.is_standard_layout());
        for i in 0..axis_length {
            for j in 0..axis_length {
                for k in 0..axis_length {
                    c[(i, j, k)] = a[(k, i, j)].clone();
                }
            }
        }
        return (a, b, c)
    }

}