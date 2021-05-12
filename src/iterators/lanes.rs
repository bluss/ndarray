use std::marker::PhantomData;

use crate::imp_prelude::*;
use crate::{Layout, NdProducer};
use crate::iterators::Baseiter;
use crate::iterators::base::NoOptimization;

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone ]
    Lanes {
        base,
        inner_len,
        inner_stride,
    }
    Lanes<'a, A, D> {
        type Item = ArrayView<'a, A, Ix1>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            ArrayView::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix))
        }
    }
}

/// See [`.lanes()`](../struct.ArrayBase.html#method.lanes)
/// for more information.
pub struct Lanes<'a, A, D> {
    base: ArrayView<'a, A, D>,
    inner_len: Ix,
    inner_stride: Ixs,
}

impl<'a, A, D: Dimension> Lanes<'a, A, D> {
    pub(crate) fn new<Di>(v: ArrayView<'a, A, Di>, axis: Axis) -> Self
    where
        Di: Dimension<Smaller = D>,
    {
        let ndim = v.ndim();
        let len;
        let stride;
        let iter_v;
        if ndim == 0 {
            len = 1;
            stride = 1;
            iter_v = v.try_remove_axis(Axis(0))
        } else {
            let i = axis.index();
            len = v.dim[i];
            stride = v.strides[i] as isize;
            iter_v = v.try_remove_axis(axis)
        }
        Lanes {
            inner_len: len,
            inner_stride: stride,
            base: iter_v,
        }
    }
}

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone =>]
    LanesMut {
        base,
        inner_len,
        inner_stride,
    }
    LanesMut<'a, A, D> {
        type Item = ArrayViewMut<'a, A, Ix1>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            ArrayViewMut::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix))
        }
    }
}

impl<'a, A, D> IntoIterator for Lanes<'a, A, D>
where
    D: Dimension,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = LanesIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        LanesIter {
            iter: self.base.into_base_iter::<NoOptimization>(),
            inner_len: self.inner_len,
            inner_stride: self.inner_stride,
            life: PhantomData,
        }
    }
}

/// See [`.lanes_mut()`](../struct.ArrayBase.html#method.lanes_mut)
/// for more information.
pub struct LanesMut<'a, A, D> {
    base: ArrayViewMut<'a, A, D>,
    inner_len: Ix,
    inner_stride: Ixs,
}

impl<'a, A, D: Dimension> LanesMut<'a, A, D> {
    pub(crate) fn new<Di>(v: ArrayViewMut<'a, A, Di>, axis: Axis) -> Self
    where
        Di: Dimension<Smaller = D>,
    {
        let ndim = v.ndim();
        let len;
        let stride;
        let iter_v;
        if ndim == 0 {
            len = 1;
            stride = 1;
            iter_v = v.try_remove_axis(Axis(0))
        } else {
            let i = axis.index();
            len = v.dim[i];
            stride = v.strides[i] as isize;
            iter_v = v.try_remove_axis(axis)
        }
        LanesMut {
            inner_len: len,
            inner_stride: stride,
            base: iter_v,
        }
    }
}

impl<'a, A, D> IntoIterator for LanesMut<'a, A, D>
where
    D: Dimension,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = LanesIterMut<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        LanesIterMut {
            iter: self.base.into_base_iter::<NoOptimization>(),
            inner_len: self.inner_len,
            inner_stride: self.inner_stride,
            life: PhantomData,
        }
    }
}

/// An iterator that traverses over all axes but one, and yields a view for
/// each lane along that axis.
///
/// See [`.lanes()`](../struct.ArrayBase.html#method.lanes) for more information.
pub struct LanesIter<'a, A, D> {
    inner_len: Ix,
    inner_stride: Ixs,
    iter: Baseiter<A, D>,
    life: PhantomData<&'a A>,
}

clone_bounds!(
    ['a, A, D: Clone]
    LanesIter['a, A, D] {
        @copy {
            inner_len,
            inner_stride,
            life,
        }
        iter,
    }
);

impl<'a, A, D> Iterator for LanesIter<'a, A, D>
where
    D: Dimension,
{
    type Item = ArrayView<'a, A, Ix1>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| unsafe {
            ArrayView::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D> ExactSizeIterator for LanesIter<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

// NOTE: LanesIterMut is a mutable iterator and must not expose aliasing
// pointers. Due to this we use an empty slice for the raw data (it's unused
// anyway).
/// An iterator that traverses over all dimensions but the innermost,
/// and yields each inner row (mutable).
///
/// See [`.lanes_mut()`](../struct.ArrayBase.html#method.lanes_mut)
/// for more information.
pub struct LanesIterMut<'a, A, D> {
    inner_len: Ix,
    inner_stride: Ixs,
    iter: Baseiter<A, D>,
    life: PhantomData<&'a mut A>,
}

impl<'a, A, D> Iterator for LanesIterMut<'a, A, D>
where
    D: Dimension,
{
    type Item = ArrayViewMut<'a, A, Ix1>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| unsafe {
            ArrayViewMut::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D> ExactSizeIterator for LanesIterMut<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

