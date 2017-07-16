// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::slice;

use imp_prelude::*;
use dimension::{self, stride_offset};
use error::ShapeError;

use StrideShape;

use {
    ElementsBase,
    ElementsBaseMut,
    Iter,
    IterMut,
    Baseiter,
};

use iter;
use iterators;

/// # Methods Specific to Array Views
///
/// Methods for read-only array views `ArrayView<'a, A, D>`
///
/// Note that array views implement traits like [`From`][f] and `IntoIterator` too.
///
/// [f]: #method.from
impl<'a, A, D> ArrayBase<ViewRepr<&'a A>, D>
    where D: Dimension,
{
    /// Create a read-only array view borrowing its data from a slice.
    ///
    /// Checks whether `shape` are compatible with the slice's
    /// length, returning an `Err` if not compatible.
    ///
    /// ```
    /// use ndarray::ArrayView;
    /// use ndarray::arr3;
    /// use ndarray::ShapeBuilder;
    ///
    /// let s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    /// let a = ArrayView::from_shape((2, 3, 2).strides((1, 4, 2)),
    ///                               &s).unwrap();
    ///
    /// assert!(
    ///     a == arr3(&[[[0, 2],
    ///                  [4, 6],
    ///                  [8, 10]],
    ///                 [[1, 3],
    ///                  [5, 7],
    ///                  [9, 11]]])
    /// );
    /// assert!(a.strides() == &[1, 4, 2]);
    /// ```
    pub fn from_shape<Sh>(shape: Sh, xs: &'a [A])
        -> Result<Self, ShapeError>
        where Sh: Into<StrideShape<D>>,
    {
        let shape = shape.into();
        let dim = shape.dim;
        let strides = shape.strides;
        dimension::can_index_slice(xs, &dim, &strides).map(|_| {
            unsafe {
                Self::new_(xs.as_ptr(), dim, strides)
            }
        })
    }

    /// Create an `ArrayView<A, D>` from shape information and a
    /// raw pointer to the elements.
    ///
    /// Unsafe because caller is responsible for ensuring that the pointer is
    /// valid, not mutably aliased and coherent with the dimension and stride information.
    pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *const A) -> Self
        where Sh: Into<StrideShape<D>>
    {
        let shape = shape.into();
        let dim = shape.dim;
        let strides = shape.strides;
        ArrayView::new_(ptr, dim, strides)
    }

    /// Split the array view along `axis` and return one view strictly before the
    /// split and one view after the split.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    ///
    /// Below, an illustration of `.split_at(Axis(2), 2)` on
    /// an array with shape 3 × 5 × 5.
    ///
    /// <img src="https://bluss.github.io/ndarray/images/split_at.svg" width="300px" height="271px">
    pub fn split_at(self, axis: Axis, index: Ix)
        -> (Self, Self)
    {
        // NOTE: Keep this in sync with the ArrayViewMut version
        assert!(index <= self.shape().axis(axis));
        let left_ptr = self.ptr;
        let right_ptr = if index == self.shape().axis(axis) {
            self.ptr
        } else {
            let offset = stride_offset(index, self.strides.axis(axis));
            unsafe {
                self.ptr.offset(offset)
            }
        };

        let mut dim_left = self.dim.clone();
        dim_left.set_axis(axis, index);
        let left = unsafe {
            Self::new_(left_ptr, dim_left, self.strides.clone())
        };

        let mut dim_right = self.dim;
        let right_len  = dim_right.axis(axis) - index;
        dim_right.set_axis(axis, right_len);
        let right = unsafe {
            Self::new_(right_ptr, dim_right, self.strides)
        };

        (left, right)
    }

    /// Return the array’s data as a slice, if it is contiguous and in standard order.
    /// Return `None` otherwise.
    pub fn into_slice(&self) -> Option<&'a [A]> {
        if self.is_standard_layout() {
            unsafe {
                Some(slice::from_raw_parts(self.ptr, self.len()))
            }
        } else {
            None
        }
    }

}

/// Methods for read-write array views `ArrayViewMut<'a, A, D>`
///
/// Note that array views implement traits like [`From`][f] and `IntoIterator` too.
///
/// [f]: #method.from
impl<'a, A, D> ArrayBase<ViewRepr<&'a mut A>, D>
    where D: Dimension,
{
    /// Create a read-write array view borrowing its data from a slice.
    ///
    /// Checks whether `dim` and `strides` are compatible with the slice's
    /// length, returning an `Err` if not compatible.
    ///
    /// ```
    /// use ndarray::ArrayViewMut;
    /// use ndarray::arr3;
    /// use ndarray::ShapeBuilder;
    ///
    /// let mut s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    /// let mut a = ArrayViewMut::from_shape((2, 3, 2).strides((1, 4, 2)),
    ///                                      &mut s).unwrap();
    ///
    /// a[[0, 0, 0]] = 1;
    /// assert!(
    ///     a == arr3(&[[[1, 2],
    ///                  [4, 6],
    ///                  [8, 10]],
    ///                 [[1, 3],
    ///                  [5, 7],
    ///                  [9, 11]]])
    /// );
    /// assert!(a.strides() == &[1, 4, 2]);
    /// ```
    pub fn from_shape<Sh>(shape: Sh, xs: &'a mut [A])
        -> Result<Self, ShapeError>
        where Sh: Into<StrideShape<D>>,
    {
        let shape = shape.into();
        let dim = shape.dim;
        let strides = shape.strides;
        dimension::can_index_slice(xs, &dim, &strides).map(|_| {
            unsafe {
                Self::new_(xs.as_mut_ptr(), dim, strides)
            }
        })
    }

    /// Create an `ArrayViewMut<A, D>` from shape information and a
    /// raw pointer to the elements.
    ///
    /// Unsafe because caller is responsible for ensuring that the pointer is
    /// valid, not aliased and coherent with the dimension and stride information.
    pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *mut A) -> Self
        where Sh: Into<StrideShape<D>>
    {
        let shape = shape.into();
        let dim = shape.dim;
        let strides = shape.strides;
        ArrayViewMut::new_(ptr, dim, strides)
    }

    /// Split the array view along `axis` and return one mutable view strictly
    /// before the split and one mutable view after the split.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    pub fn split_at(self, axis: Axis, index: Ix)
        -> (Self, Self)
    {
        // NOTE: Keep this in sync with the ArrayView version
        assert!(index <= self.shape().axis(axis));
        let left_ptr = self.ptr;
        let right_ptr = if index == self.shape().axis(axis) {
            self.ptr
        } else {
            let offset = stride_offset(index, self.strides.axis(axis));
            unsafe {
                self.ptr.offset(offset)
            }
        };

        let mut dim_left = self.dim.clone();
        dim_left.set_axis(axis, index);
        let left = unsafe {
            Self::new_(left_ptr, dim_left, self.strides.clone())
        };

        let mut dim_right = self.dim;
        let right_len  = dim_right.axis(axis) - index;
        dim_right.set_axis(axis, right_len);
        let right = unsafe {
            Self::new_(right_ptr, dim_right, self.strides)
        };

        (left, right)
    }

    /// Return the array’s data as a slice, if it is contiguous and in standard order.
    /// Return `None` otherwise.
    pub fn into_slice(self) -> Option<&'a mut [A]> {
        self.into_slice_().ok()
    }

}

/// Private array view methods
impl<'a, A, D> ArrayBase<ViewRepr<&'a A>, D>
    where D: Dimension,
{
    /// Create a new `ArrayView`
    ///
    /// Unsafe because: `ptr` must be valid for the given dimension and strides.
    #[inline(always)]
    pub(crate) unsafe fn new_(ptr: *const A, dim: D, strides: D) -> Self {
        ArrayView {
            data: ViewRepr::new(),
            ptr: ptr as *mut A,
            dim: dim,
            strides: strides,
        }
    }

    #[inline]
    pub(crate) fn into_base_iter(self) -> Baseiter<'a, A, D> {
        unsafe {
            Baseiter::new(self.ptr, self.dim, self.strides)
        }
    }

    #[inline]
    pub(crate) fn into_elements_base(self) -> ElementsBase<'a, A, D> {
        ElementsBase { inner: self.into_base_iter() }
    }

    pub(crate) fn into_iter_(self) -> Iter<'a, A, D> {
        Iter::new(self)
    }

    /// Return an outer iterator for this view.
    #[doc(hidden)] // not official
    #[deprecated(note="This method will be replaced.")]
    pub fn into_outer_iter(self) -> iter::AxisIter<'a, A, D::Smaller>
        where D: RemoveAxis,
    {
        iterators::new_outer_iter(self)
    }

}

impl<'a, A, D> ArrayBase<ViewRepr<&'a mut A>, D>
    where D: Dimension,
{
    /// Create a new `ArrayView`
    ///
    /// Unsafe because: `ptr` must be valid for the given dimension and strides.
    #[inline(always)]
    pub(crate) unsafe fn new_(ptr: *mut A, dim: D, strides: D) -> Self {
        ArrayViewMut {
            data: ViewRepr::new(),
            ptr: ptr,
            dim: dim,
            strides: strides,
        }
    }

    // Convert into a read-only view
    pub(crate) fn into_view(self) -> ArrayView<'a, A, D> {
        unsafe {
            ArrayView::new_(self.ptr, self.dim, self.strides)
        }
    }

    #[inline]
    pub(crate) fn into_base_iter(self) -> Baseiter<'a, A, D> {
        unsafe {
            Baseiter::new(self.ptr, self.dim, self.strides)
        }
    }

    #[inline]
    pub(crate) fn into_elements_base(self) -> ElementsBaseMut<'a, A, D> {
        ElementsBaseMut { inner: self.into_base_iter() }
    }

    pub(crate) fn into_slice_(self) -> Result<&'a mut [A], Self> {
        if self.is_standard_layout() {
            unsafe {
                Ok(slice::from_raw_parts_mut(self.ptr, self.len()))
            }
        } else {
            Err(self)
        }
    }

    pub(crate) fn into_iter_(self) -> IterMut<'a, A, D> {
        IterMut::new(self)
    }

    /// Return an outer iterator for this view.
    #[doc(hidden)] // not official
    #[deprecated(note="This method will be replaced.")]
    pub fn into_outer_iter(self) -> iter::AxisIterMut<'a, A, D::Smaller>
        where D: RemoveAxis,
    {
        iterators::new_outer_iter_mut(self)
    }
}

