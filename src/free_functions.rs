// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem::{forget, size_of};
use alloc::slice;
use alloc::vec;
use alloc::vec::Vec;

use crate::imp_prelude::*;
use crate::{dimension, ArcArray1, ArcArray2};

/// Create an [**`Array`**](type.Array.html) with one, two or
/// three dimensions.
///
/// ```
/// use ndarray::array;
/// let a1 = array![1, 2, 3, 4];
///
/// let a2 = array![[1, 2],
///                 [3, 4]];
///
/// let a3 = array![[[1, 2], [3, 4]],
///                 [[5, 6], [7, 8]]];
///
/// assert_eq!(a1.shape(), &[4]);
/// assert_eq!(a2.shape(), &[2, 2]);
/// assert_eq!(a3.shape(), &[2, 2, 2]);
/// ```
///
/// This macro uses `vec![]`, and has the same ownership semantics;
/// elements are moved into the resulting `Array`.
///
/// Use `array![...].into_shared()` to create an `ArcArray`.
#[macro_export]
macro_rules! array {
    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::Array3::from(vec![$([$([$($x,)*],)*],)*])
    }};
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::Array2::from(vec![$([$($x,)*],)*])
    }};
    ($($x:expr),* $(,)*) => {{
        $crate::Array::from(vec![$($x,)*])
    }};
}

/// Create a zero-dimensional array with the element `x`.
pub fn arr0<A>(x: A) -> Array0<A> {
    unsafe { ArrayBase::from_shape_vec_unchecked((), vec![x]) }
}

/// Create a one-dimensional array with elements from `xs`.
pub fn arr1<A: Clone>(xs: &[A]) -> Array1<A> {
    ArrayBase::from(xs.to_vec())
}

/// Create a one-dimensional array with elements from `xs`.
pub fn rcarr1<A: Clone>(xs: &[A]) -> ArcArray1<A> {
    arr1(xs).into_shared()
}

/// Create a zero-dimensional array view borrowing `x`.
pub fn aview0<A>(x: &A) -> ArrayView0<'_, A> {
    unsafe { ArrayView::from_shape_ptr(Ix0(), x) }
}

/// Create a one-dimensional array view with elements borrowing `xs`.
///
/// ```
/// use ndarray::aview1;
///
/// let data = [1.0; 1024];
///
/// // Create a 2D array view from borrowed data
/// let a2d = aview1(&data).into_shape((32, 32)).unwrap();
///
/// assert_eq!(a2d.sum(), 1024.0);
/// ```
pub fn aview1<A>(xs: &[A]) -> ArrayView1<'_, A> {
    ArrayView::from(xs)
}

/// Create a two-dimensional array view with elements borrowing `xs`.
///
/// **Panics** if the product of non-zero axis lengths overflows `isize`. (This
/// can only occur when `V` is zero-sized.)
pub fn aview2<A, V: FixedInitializer<Elem = A>>(xs: &[V]) -> ArrayView2<'_, A> {
    let cols = V::len();
    let rows = xs.len();
    let dim = Ix2(rows, cols);
    if size_of::<V>() == 0 {
        dimension::size_of_shape_checked(&dim)
            .expect("Product of non-zero axis lengths must not overflow isize.");
    }
    // `rows` is guaranteed to fit in `isize` because we've checked the ZST
    // case and slices never contain > `isize::MAX` bytes. `cols` is guaranteed
    // to fit in `isize` because `FixedInitializer` is not implemented for any
    // array lengths > `isize::MAX`. `cols * rows` is guaranteed to fit in
    // `isize` because we've checked the ZST case and slices never contain >
    // `isize::MAX` bytes.
    unsafe {
        let data = slice::from_raw_parts(xs.as_ptr() as *const A, cols * rows);
        ArrayView::from_shape_ptr(dim, data.as_ptr())
    }
}

/// Create a one-dimensional read-write array view with elements borrowing `xs`.
///
/// ```
/// use ndarray::{aview_mut1, s};
/// // Create an array view over some data, then slice it and modify it.
/// let mut data = [0; 1024];
/// {
///     let mut a = aview_mut1(&mut data).into_shape((32, 32)).unwrap();
///     a.slice_mut(s![.., ..;3]).fill(5);
/// }
/// assert_eq!(&data[..10], [5, 0, 0, 5, 0, 0, 5, 0, 0, 5]);
/// ```
pub fn aview_mut1<A>(xs: &mut [A]) -> ArrayViewMut1<'_, A> {
    ArrayViewMut::from(xs)
}

/// Create a two-dimensional read-write array view with elements borrowing `xs`.
///
/// **Panics** if the product of non-zero axis lengths overflows `isize`. (This
/// can only occur when `V` is zero-sized.)
///
/// # Example
///
/// ```
/// use ndarray::aview_mut2;
///
/// // The inner (nested) array must be of length 1 to 16, but the outer
/// // can be of any length.
/// let mut data = [[0.; 2]; 128];
/// {
///     // Make a 128 x 2 mut array view then turn it into 2 x 128
///     let mut a = aview_mut2(&mut data).reversed_axes();
///     // Make the first row ones and second row minus ones.
///     a.row_mut(0).fill(1.);
///     a.row_mut(1).fill(-1.);
/// }
/// // look at the start of the result
/// assert_eq!(&data[..3], [[1., -1.], [1., -1.], [1., -1.]]);
/// ```
pub fn aview_mut2<A, V: FixedInitializer<Elem = A>>(xs: &mut [V]) -> ArrayViewMut2<'_, A> {
    let cols = V::len();
    let rows = xs.len();
    let dim = Ix2(rows, cols);
    if size_of::<V>() == 0 {
        dimension::size_of_shape_checked(&dim)
            .expect("Product of non-zero axis lengths must not overflow isize.");
    }
    // `rows` is guaranteed to fit in `isize` because we've checked the ZST
    // case and slices never contain > `isize::MAX` bytes. `cols` is guaranteed
    // to fit in `isize` because `FixedInitializer` is not implemented for any
    // array lengths > `isize::MAX`. `cols * rows` is guaranteed to fit in
    // `isize` because we've checked the ZST case and slices never contain >
    // `isize::MAX` bytes.
    unsafe {
        let data = slice::from_raw_parts_mut(xs.as_mut_ptr() as *mut A, cols * rows);
        ArrayViewMut::from_shape_ptr(dim, data.as_mut_ptr())
    }
}

/// Fixed-size array used for array initialization
pub unsafe trait FixedInitializer {
    type Elem;
    fn as_init_slice(&self) -> &[Self::Elem];
    fn len() -> usize;
}

macro_rules! impl_arr_init {
    (__impl $n: expr) => (
        unsafe impl<T> FixedInitializer for [T;  $n] {
            type Elem = T;
            fn as_init_slice(&self) -> &[T] { self }
            fn len() -> usize { $n }
        }
    );
    () => ();
    ($n: expr, $($m:expr,)*) => (
        impl_arr_init!(__impl $n);
        impl_arr_init!($($m,)*);
    )

}

// For implementors: If you ever implement `FixedInitializer` for array lengths
// > `isize::MAX` (e.g. once Rust adds const generics), you must update
// `aview2` and `aview_mut2` to perform the necessary checks. In particular,
// the assumption that `cols` can never exceed `isize::MAX` would be incorrect.
// (Consider e.g. `let xs: &[[i32; ::std::usize::MAX]] = &[]`.)
impl_arr_init!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,);

/// Create a two-dimensional array with elements from `xs`.
///
/// ```
/// use ndarray::arr2;
///
/// let a = arr2(&[[1, 2, 3],
///                [4, 5, 6]]);
/// assert!(
///     a.shape() == [2, 3]
/// );
/// ```
pub fn arr2<A: Clone, V: FixedInitializer<Elem = A>>(xs: &[V]) -> Array2<A>
where
    V: Clone,
{
    Array2::from(xs.to_vec())
}

impl<A, V> From<Vec<V>> for Array2<A>
where
    V: FixedInitializer<Elem = A>,
{
    /// Converts the `Vec` of arrays to an owned 2-D array.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    fn from(mut xs: Vec<V>) -> Self {
        let dim = Ix2(xs.len(), V::len());
        let ptr = xs.as_mut_ptr();
        let cap = xs.capacity();
        let expand_len = dimension::size_of_shape_checked(&dim)
            .expect("Product of non-zero axis lengths must not overflow isize.");
        forget(xs);
        unsafe {
            let v = if size_of::<A>() == 0 {
                Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
            } else if V::len() == 0 {
                Vec::new()
            } else {
                // Guaranteed not to overflow in this case since A is non-ZST
                // and Vec never allocates more than isize bytes.
                let expand_cap = cap * V::len();
                Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
            };
            ArrayBase::from_shape_vec_unchecked(dim, v)
        }
    }
}

impl<A, V, U> From<Vec<V>> for Array3<A>
where
    V: FixedInitializer<Elem = U>,
    U: FixedInitializer<Elem = A>,
{
    /// Converts the `Vec` of arrays to an owned 3-D array.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    fn from(mut xs: Vec<V>) -> Self {
        let dim = Ix3(xs.len(), V::len(), U::len());
        let ptr = xs.as_mut_ptr();
        let cap = xs.capacity();
        let expand_len = dimension::size_of_shape_checked(&dim)
            .expect("Product of non-zero axis lengths must not overflow isize.");
        forget(xs);
        unsafe {
            let v = if size_of::<A>() == 0 {
                Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
            } else if V::len() == 0 || U::len() == 0 {
                Vec::new()
            } else {
                // Guaranteed not to overflow in this case since A is non-ZST
                // and Vec never allocates more than isize bytes.
                let expand_cap = cap * V::len() * U::len();
                Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
            };
            ArrayBase::from_shape_vec_unchecked(dim, v)
        }
    }
}

/// Create a two-dimensional array with elements from `xs`.
///
pub fn rcarr2<A: Clone, V: Clone + FixedInitializer<Elem = A>>(xs: &[V]) -> ArcArray2<A> {
    arr2(xs).into_shared()
}

/// Create a three-dimensional array with elements from `xs`.
///
/// **Panics** if the slices are not all of the same length.
///
/// ```
/// use ndarray::arr3;
///
/// let a = arr3(&[[[1, 2],
///                 [3, 4]],
///                [[5, 6],
///                 [7, 8]],
///                [[9, 0],
///                 [1, 2]]]);
/// assert!(
///     a.shape() == [3, 2, 2]
/// );
/// ```
pub fn arr3<A: Clone, V: FixedInitializer<Elem = U>, U: FixedInitializer<Elem = A>>(
    xs: &[V],
) -> Array3<A>
where
    V: Clone,
    U: Clone,
{
    Array3::from(xs.to_vec())
}

/// Create a three-dimensional array with elements from `xs`.
pub fn rcarr3<A: Clone, V: FixedInitializer<Elem = U>, U: FixedInitializer<Elem = A>>(
    xs: &[V],
) -> ArcArray<A, Ix3>
where
    V: Clone,
    U: Clone,
{
    arr3(xs).into_shared()
}
