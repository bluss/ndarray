// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::{Add, Div, Mul};
use libnum::{self, One, Zero, Float};
use itertools::free::enumerate;

use imp_prelude::*;
use numeric_util;

use {FoldWhile, Zip};

/// Numerical methods for arrays.
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
    /// assert_eq!(a.scalar_sum(), 10.);
    /// ```
    pub fn scalar_sum(&self) -> A
        where A: Clone + Add<Output=A> + libnum::Zero,
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

    /// Return the product of all elements in the array.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert_eq!(a.scalar_prod(), 24.);
    /// ```
    pub fn scalar_prod(&self) -> A
        where A: Clone + Mul<Output=A> + libnum::One,
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
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.sum_axis(Axis(0)) == aview1(&[4., 6.]) &&
    ///     a.sum_axis(Axis(1)) == aview1(&[3., 7.]) &&
    ///
    ///     a.sum_axis(Axis(0)).sum_axis(Axis(0)) == aview0(&10.)
    /// );
    /// ```
    ///
    /// **Panics** if `axis` is out of bounds.
    pub fn sum_axis(&self, axis: Axis) -> Array<A, D::Smaller>
        where A: Clone + Zero + Add<Output=A>,
              D: RemoveAxis,
    {
        let n = self.len_of(axis);
        let mut res = Array::zeros(self.raw_dim().remove_axis(axis));
        let stride = self.strides()[axis.index()];
        if self.ndim() == 2 && stride == 1 {
            // contiguous along the axis we are summing
            let ax = axis.index();
            for (i, elt) in enumerate(&mut res) {
                *elt = self.subview(Axis(1 - ax), i).scalar_sum();
            }
        } else {
            for i in 0..n {
                let view = self.subview(axis, i);
                res = res + &view;
            }
        }
        res
    }

    /// Return mean along `axis`.
    ///
    /// **Panics** if `axis` is out of bounds or if the length of the axis is
    /// zero and division by zero panics for type `A`.
    ///
    /// ```
    /// use ndarray::{aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.mean_axis(Axis(0)) == aview1(&[2.0, 3.0]) &&
    ///     a.mean_axis(Axis(1)) == aview1(&[1.5, 3.5])
    /// );
    /// ```
    pub fn mean_axis(&self, axis: Axis) -> Array<A, D::Smaller>
        where A: Clone + Zero + One + Add<Output=A> + Div<Output=A>,
              D: RemoveAxis,
    {
        let n = self.len_of(axis);
        let sum = self.sum_axis(axis);
        let mut cnt = A::zero();
        for _ in 0..n {
            cnt = cnt + A::one();
        }
        sum / &aview0(&cnt)
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
    /// **Panics** if `ddof` is greater than or equal to the length of the
    /// axis, if `axis` is out of bounds, or if the length of the axis is zero.
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
        A: Float,
        D: RemoveAxis,
    {
        let mut count = A::zero();
        let mut mean = Array::<A, _>::zeros(self.dim.remove_axis(axis));
        let mut sum_sq = Array::<A, _>::zeros(self.dim.remove_axis(axis));
        for subview in self.axis_iter(axis) {
            count = count + A::one();
            azip!(mut mean, mut sum_sq, x (subview) in {
                let delta = x - *mean;
                *mean = *mean + delta / count;
                *sum_sq = (x - *mean).mul_add(delta, *sum_sq);
            });
        }
        if ddof >= count {
            panic!("`ddof` needs to be strictly smaller than the length \
                    of the axis you are computing the variance for!")
        } else {
            let dof = count - ddof;
            sum_sq.mapv_into(|s| s / dof)
        }
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
    /// **Panics** if `ddof` is greater than or equal to the length of the
    /// axis, if `axis` is out of bounds, or if the length of the axis is zero.
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
        A: Float,
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

