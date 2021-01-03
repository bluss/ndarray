// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use num_traits::{self, Float, FromPrimitive, Zero};
use std::ops::{Add, Div, Mul};

use crate::imp_prelude::*;
use crate::itertools::enumerate;
use crate::numeric_util;

/// # Numerical Methods for Arrays
impl<A, S, D> ArrayBase<S, D>
where
    S: Data<Elem = A>,
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
    where
        A: Clone + Add<Output = A> + num_traits::Zero,
    {
        if let Some(slc) = self.as_slice_memory_order() {
            return numeric_util::unrolled_fold(slc, A::zero, A::add);
        }
        let mut sum = A::zero();
        for row in self.inner_rows() {
            if let Some(slc) = row.as_slice() {
                sum = sum + numeric_util::unrolled_fold(slc, A::zero, A::add);
            } else {
                sum = sum + row.iter().fold(A::zero(), |acc, elt| acc + elt.clone());
            }
        }
        sum
    }

    /// Return the sum of all elements in the array.
    ///
    /// *This method has been renamed to `.sum()`*
    #[deprecated(note="renamed to `sum`", since="0.15.0")]
    pub fn scalar_sum(&self) -> A
    where
        A: Clone + Add<Output = A> + num_traits::Zero,
    {
        self.sum()
    }

    /// Returns the [arithmetic mean] x̅ of all elements in the array:
    ///
    /// ```text
    ///     1   n
    /// x̅ = ―   ∑ xᵢ
    ///     n  i=1
    /// ```
    ///
    /// If the array is empty, `None` is returned.
    ///
    /// **Panics** if `A::from_usize()` fails to convert the number of elements in the array.
    ///
    /// [arithmetic mean]: https://en.wikipedia.org/wiki/Arithmetic_mean
    pub fn mean(&self) -> Option<A>
    where
        A: Clone + FromPrimitive + Add<Output = A> + Div<Output = A> + Zero,
    {
        let n_elements = self.len();
        if n_elements == 0 {
            None
        } else {
            let n_elements = A::from_usize(n_elements)
                .expect("Converting number of elements to `A` must not fail.");
            Some(self.sum() / n_elements)
        }
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
    where
        A: Clone + Mul<Output = A> + num_traits::One,
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

    /// Return a reference to a maximum of all values.
    /// Return None if a comparison fails or if self is empty.
    /// 
    /// # Example
    /// ```
    /// use ndarray::{arr2, Array2};
    /// use std::f64;
    /// 
    /// let a = arr2(&[[1., 2.], [3., 4.]]);
    /// assert_eq!(a.max(), Some(&4.));
    /// 
    /// let b = arr2(&[[1., f64::NAN], [3., 4.]]);
    /// assert_eq!(b.max(), None);
    /// 
    /// let c = arr2(&[[f64::NAN]]);
    /// assert_eq!(c.max(), None);
    /// 
    /// let d: Array2<f64> = arr2(&[[]]);
    /// assert_eq!(d.max(), None);
    /// ```
    pub fn max(&self) -> Option<&A>
    where A: PartialOrd
    {
        if let Some(first) = self.first() {
            let max = self.fold(first, |acc, x| if acc == acc && !(x < acc) {x} else {acc});
            if max == max {
                Some(max)
            } else {
                None
            }
        } else {
            None
        }

    }  

    /// Return a reference to a maximum of all values, ignoring
    /// incomparable elements. Returns None if `self` is empty,
    /// or contains only incomparable elements.
    /// 
    /// # Example
    /// ```
    /// use ndarray::{arr2, Array2};
    /// use std::f64;
    /// 
    /// let a = arr2(&[[1., 2.], [3., 4.]]);
    /// assert_eq!(a.nanmax(), Some(&4.));
    /// 
    /// let b = arr2(&[[1., f64::NAN], [3., f64::NAN]]);
    /// assert_eq!(b.nanmax(), Some(&3.));
    /// 
    /// let c: Array2<f64> = arr2(&[[]]);
    /// assert_eq!(c.nanmax(), None);
    /// 
    /// let d = arr2(&[[f64::NAN, f64::NAN],[f64::NAN, f64::NAN]]);
    /// assert_eq!(d.nanmax(), None);
    /// ```
    pub fn nanmax(&self) -> Option<&A> 
    where A: PartialOrd,
    {
        if let Some(first) = self.first() {
            let max = self.fold(first, |acc, x| if acc == acc && !(x > acc) {acc} else {x});
            if max == max {
                Some(max)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Return a reference to a minimum of all values.
    /// Return None if a comparison fails or if self is empty.
    /// 
    /// # Example
    /// ```
    /// use ndarray::{arr2, Array2};
    /// use std::f64;
    /// 
    /// let a = arr2(&[[1., 2.], [3., 4.]]);
    /// assert_eq!(a.min(), Some(&1.));
    /// 
    /// let b = arr2(&[[1., f64::NAN], [3., 4.]]);
    /// assert_eq!(b.min(), None);
    /// 
    /// let c = arr2(&[[f64::NAN]]);
    /// assert_eq!(c.min(), None);
    /// 
    /// let d: Array2<f64> = arr2(&[[]]);
    /// assert_eq!(d.min(), None);
    /// ```
    pub fn min(&self) -> Option<&A>
    where A: PartialOrd
    {
        if let Some(first) = self.first() {
            let min = self.fold(first, |acc, x| if acc == acc && !(acc < x) {x} else {acc});
            if min == min {
                Some(min)
            } else {
                None
            }
        } else {
            None
        }
    }  

    /// Return a reference to a minimum of all values, ignoring
    /// incomparable elements. Returns None if `self` is empty,
    /// or contains only incomparable elements.
    /// 
    /// # Example
    /// ```
    /// use ndarray::{arr2, Array2};
    /// use std::f64;
    /// 
    /// let a = arr2(&[[1., 2.], [3., 4.]]);
    /// assert_eq!(a.nanmin(), Some(&1.));
    /// 
    /// let b = arr2(&[[f64::NAN, 2.], [3., f64::NAN]]);
    /// assert_eq!(b.nanmin(), Some(&2.));
    /// 
    /// let c: Array2<f64> = arr2(&[[]]);
    /// assert_eq!(c.nanmin(), None);
    /// 
    /// let d = arr2(&[[f64::NAN, f64::NAN],[f64::NAN, f64::NAN]]);
    /// assert_eq!(d.nanmin(), None);
    /// ```
    pub fn nanmin(&self) -> Option<&A> 
    where A: PartialOrd,
    {
        if let Some(first) = self.first() {
            let min = self.fold(first, |acc, x| if acc == acc && !(x < acc) {acc} else {x});
            if min == min {
                Some(min)
            } else {
                None
            }
        } else {
            None
        }
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
    where
        A: Clone + Zero + Add<Output = A>,
        D: RemoveAxis,
    {
        let n = self.len_of(axis);
        let mut res = Array::zeros(self.raw_dim().remove_axis(axis));
        let stride = self.strides()[axis.index()];
        if self.ndim() == 2 && stride == 1 {
            // contiguous along the axis we are summing
            let ax = axis.index();
            for (i, elt) in enumerate(&mut res) {
                *elt = self.index_axis(Axis(1 - ax), i).sum();
            }
        } else {
            for i in 0..n {
                let view = self.index_axis(axis, i);
                res = res + &view;
            }
        }
        res
    }

    /// Return mean along `axis`.
    ///
    /// Return `None` if the length of the axis is zero.
    ///
    /// **Panics** if `axis` is out of bounds or if `A::from_usize()`
    /// fails for the axis length.
    ///
    /// ```
    /// use ndarray::{aview0, aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2., 3.],
    ///                [4., 5., 6.]]);
    /// assert!(
    ///     a.mean_axis(Axis(0)).unwrap() == aview1(&[2.5, 3.5, 4.5]) &&
    ///     a.mean_axis(Axis(1)).unwrap() == aview1(&[2., 5.]) &&
    ///
    ///     a.mean_axis(Axis(0)).unwrap().mean_axis(Axis(0)).unwrap() == aview0(&3.5)
    /// );
    /// ```
    pub fn mean_axis(&self, axis: Axis) -> Option<Array<A, D::Smaller>>
    where
        A: Clone + Zero + FromPrimitive + Add<Output = A> + Div<Output = A>,
        D: RemoveAxis,
    {
        let axis_length = self.len_of(axis);
        if axis_length == 0 {
            None
        } else {
            let axis_length =
                A::from_usize(axis_length).expect("Converting axis length to `A` must not fail.");
            let sum = self.sum_axis(axis);
            Some(sum / aview0(&axis_length))
        }
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
            azip!((mean in &mut mean, sum_sq in &mut sum_sq, &x in &subview) {
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
    ///               ⎛    1       n          ⎞
    /// stddev = sqrt ⎜ ――――――――   ∑ (xᵢ - x̅)²⎟
    ///               ⎝ n - ddof  i=1         ⎠
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
}
