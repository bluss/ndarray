// Copyright 2014-2021 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iter::FromIterator;
use std::marker::PhantomData;
use std::slice::{Iter as SliceIter, IterMut as SliceIterMut};

use crate::imp_prelude::*;
use crate::dimension;

/// No traversal optmizations that would change element order or axis dimensions are permitted.
///
/// This option is suitable for example for the indexed iterator.
pub(crate) enum NoOptimization { }

/// Preserve element iteration order, but modify dimensions if profitable; for example we can
/// change from shape [10, 1] to [1, 10], because that axis has len == 1, without consequence here.
///
/// This option is suitable for example for the default .iter() iterator.
pub(crate) enum PreserveOrder { }

/// Allow use of arbitrary element iteration order
///
/// This option is suitable for example for an arbitrary order iterator.
pub(crate) enum ArbitraryOrder { }

pub(crate) trait OrderOption {
    const ALLOW_REMOVE_REDUNDANT_AXES: bool = false;
    const ALLOW_ARBITRARY_ORDER: bool = false;
}

impl OrderOption for NoOptimization { }

impl OrderOption for PreserveOrder {
    const ALLOW_REMOVE_REDUNDANT_AXES: bool = true;
}

impl OrderOption for ArbitraryOrder {
    const ALLOW_REMOVE_REDUNDANT_AXES: bool = true;
    const ALLOW_ARBITRARY_ORDER: bool = true;
}

/// Base for iterators over all axes.
///
/// Iterator element type is `*mut A`.
///
/// `F` is for layout/iteration order flags
pub(crate) struct Baseiter<A, D> {
    ptr: *mut A,
    dim: D,
    strides: D,
    index: Option<D>,
}

impl<A, D: Dimension> Baseiter<A, D> {
    /// Creating a Baseiter is unsafe because shape and stride parameters need
    /// to be correct to avoid performing an unsafe pointer offset while
    /// iterating.
    #[inline]
    pub unsafe fn new(ptr: *mut A, dim: D, strides: D) -> Baseiter<A, D> {
        Self::new_with_order::<NoOptimization>(ptr, dim, strides)
    }

    /// Return the iter dimension
    pub(crate) fn raw_dim(&self) -> D { self.dim.clone() }

    /// Return the iter strides
    pub(crate) fn raw_strides(&self) -> D { self.strides.clone() }

    /// Creating a Baseiter is unsafe because shape and stride parameters need
    /// to be correct to avoid performing an unsafe pointer offset while
    /// iterating.
    #[inline]
    pub unsafe fn new_with_order<Flags: OrderOption>(mut ptr: *mut A, mut dim: D, mut strides: D)
        -> Baseiter<A, D>
    {
        debug_assert_eq!(dim.ndim(), strides.ndim());
        if Flags::ALLOW_ARBITRARY_ORDER {
            // iterate in memory order; merge axes if possible
            // make all axes positive and put the pointer back to the first element in memory
            let offset = dimension::offset_from_ptr_to_memory(&dim, &strides);
            ptr = ptr.offset(offset);
            for i in 0..strides.ndim() {
                let s = strides.get_stride(Axis(i));
                if s < 0 {
                    strides.set_stride(Axis(i), -s);
                }
            }
            dimension::sort_axes_to_standard(&mut dim, &mut strides);
        }
        if Flags::ALLOW_REMOVE_REDUNDANT_AXES {
            // preserve element order but shift dimensions
            dimension::merge_axes_from_the_back(&mut dim, &mut strides);
            dimension::squeeze(&mut dim, &mut strides);
        }
        Baseiter {
            ptr,
            index: dim.first_index(),
            dim,
            strides,
        }
    }
}

impl<A, D: Dimension> Iterator for Baseiter<A, D> {
    type Item = *mut A;

    #[inline]
    fn next(&mut self) -> Option<*mut A> {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        let offset = D::stride_offset(&index, &self.strides);
        self.index = self.dim.next_for(index);
        unsafe { Some(self.ptr.offset(offset)) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn fold<Acc, G>(mut self, init: Acc, mut g: G) -> Acc
    where
        G: FnMut(Acc, *mut A) -> Acc,
    {
        let ndim = self.dim.ndim();
        debug_assert_ne!(ndim, 0);
        let mut accum = init;
        while let Some(mut index) = self.index {
            let stride = self.strides.last_elem() as isize;
            let elem_index = index.last_elem();
            let len = self.dim.last_elem();
            let offset = D::stride_offset(&index, &self.strides);
            unsafe {
                let row_ptr = self.ptr.offset(offset);
                let mut i = 0;
                let i_end = len - elem_index;
                while i < i_end {
                    accum = g(accum, row_ptr.offset(i as isize * stride));
                    i += 1;
                }
            }
            index.set_last_elem(len - 1);
            self.index = self.dim.next_for(index);
        }
        accum
    }
}

impl<'a, A, D: Dimension> ExactSizeIterator for Baseiter<A, D> {
    fn len(&self) -> usize {
        match self.index {
            None => 0,
            Some(ref ix) => {
                let gone = self
                    .dim
                    .default_strides()
                    .slice()
                    .iter()
                    .zip(ix.slice().iter())
                    .fold(0, |s, (&a, &b)| s + a as usize * b as usize);
                self.dim.size() - gone
            }
        }
    }
}

impl<A> DoubleEndedIterator for Baseiter<A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<*mut A> {
        let index = match self.index {
            None => return None,
            Some(ix) => ix,
        };
        self.dim[0] -= 1;
        let offset = <_>::stride_offset(&self.dim, &self.strides);
        if index == self.dim {
            self.index = None;
        }

        unsafe { Some(self.ptr.offset(offset)) }
    }

    fn nth_back(&mut self, n: usize) -> Option<*mut A> {
        let index = self.index?;
        let len = self.dim[0] - index[0];
        if n < len {
            self.dim[0] -= n + 1;
            let offset = <_>::stride_offset(&self.dim, &self.strides);
            if index == self.dim {
                self.index = None;
            }
            unsafe { Some(self.ptr.offset(offset)) }
        } else {
            self.index = None;
            None
        }
    }

    fn rfold<Acc, G>(mut self, init: Acc, mut g: G) -> Acc
    where
        G: FnMut(Acc, *mut A) -> Acc,
    {
        let mut accum = init;
        if let Some(index) = self.index {
            let elem_index = index[0];
            unsafe {
                // self.dim[0] is the current length
                while self.dim[0] > elem_index {
                    self.dim[0] -= 1;
                    accum = g(
                        accum,
                        self.ptr
                            .offset(Ix1::stride_offset(&self.dim, &self.strides)),
                    );
                }
            }
        }
        accum
    }
}

clone_bounds!(
    [A, D: Clone]
    Baseiter[A, D] {
        @copy {
            ptr,
        }
        dim,
        strides,
        index,
    }
);

clone_bounds!(
    ['a, A, D: Clone]
    ElementsBase['a, A, D] {
        @copy {
            life,
        }
        inner,
    }
);

impl<'a, A, D: Dimension> ElementsBase<'a, A, D> {
    pub fn new<F: OrderOption>(v: ArrayView<'a, A, D>) -> Self {
        ElementsBase {
            inner: v.into_base_iter::<F>(),
            life: PhantomData,
        }
    }
}

impl<'a, A, D: Dimension> Iterator for ElementsBase<'a, A, D> {
    type Item = &'a A;
    #[inline]
    fn next(&mut self) -> Option<&'a A> {
        self.inner.next().map(|p| unsafe { &*p })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn fold<Acc, G>(self, init: Acc, mut g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        unsafe { self.inner.fold(init, move |acc, ptr| g(acc, &*ptr)) }
    }
}

impl<'a, A> DoubleEndedIterator for ElementsBase<'a, A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a A> {
        self.inner.next_back().map(|p| unsafe { &*p })
    }

    fn rfold<Acc, G>(self, init: Acc, mut g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        unsafe { self.inner.rfold(init, move |acc, ptr| g(acc, &*ptr)) }
    }
}

impl<'a, A, D> ExactSizeIterator for ElementsBase<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

macro_rules! either {
    ($value:expr, $inner:pat => $result:expr) => {
        match $value {
            ElementsRepr::Slice($inner) => $result,
            ElementsRepr::Counted($inner) => $result,
        }
    };
}

macro_rules! either_mut {
    ($value:expr, $inner:ident => $result:expr) => {
        match $value {
            ElementsRepr::Slice(ref mut $inner) => $result,
            ElementsRepr::Counted(ref mut $inner) => $result,
        }
    };
}

clone_bounds!(
    ['a, A, D: Clone]
    Iter['a, A, D] {
        @copy {
        }
        inner,
    }
);

impl<'a, A, D> Iter<'a, A, D>
where
    D: Dimension,
{
    pub(crate) fn new(self_: ArrayView<'a, A, D>) -> Self {
        Iter {
            inner: if let Some(slc) = self_.to_slice() {
                ElementsRepr::Slice(slc.iter())
            } else {
                ElementsRepr::Counted(self_.into_elements_base_preserve_order())
            },
        }
    }
}

impl<'a, A, D> IterMut<'a, A, D>
where
    D: Dimension,
{
    pub(crate) fn new(self_: ArrayViewMut<'a, A, D>) -> Self {
        IterMut {
            inner: match self_.try_into_slice() {
                Ok(x) => ElementsRepr::Slice(x.iter_mut()),
                Err(self_) => ElementsRepr::Counted(self_.into_elements_base_preserve_order()),
            },
        }
    }
}

#[derive(Clone)]
pub enum ElementsRepr<S, C> {
    Slice(S),
    Counted(C),
}

/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a A`.
///
/// See [`.iter()`](../struct.ArrayBase.html#method.iter) for more information.
pub struct Iter<'a, A, D> {
    inner: ElementsRepr<SliceIter<'a, A>, ElementsBase<'a, A, D>>,
}

/// Counted read only iterator
pub(crate) struct ElementsBase<'a, A, D> {
    inner: Baseiter<A, D>,
    life: PhantomData<&'a A>,
}

/// An iterator over the elements of an array (mutable).
///
/// Iterator element type is `&'a mut A`.
///
/// See [`.iter_mut()`](../struct.ArrayBase.html#method.iter_mut) for more information.
pub struct IterMut<'a, A, D> {
    inner: ElementsRepr<SliceIterMut<'a, A>, ElementsBaseMut<'a, A, D>>,
}

/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a mut A`.
pub(crate) struct ElementsBaseMut<'a, A, D> {
    inner: Baseiter<A, D>,
    life: PhantomData<&'a mut A>,
}

impl<'a, A, D: Dimension> ElementsBaseMut<'a, A, D> {
    pub fn new<F: OrderOption>(v: ArrayViewMut<'a, A, D>) -> Self {
        ElementsBaseMut {
            inner: v.into_base_iter::<F>(),
            life: PhantomData,
        }
    }
}

/// An iterator over the indexes and elements of an array.
///
/// See [`.indexed_iter()`](../struct.ArrayBase.html#method.indexed_iter) for more information.
#[derive(Clone)]
pub struct IndexedIter<'a, A, D>(ElementsBase<'a, A, D>);
/// An iterator over the indexes and elements of an array (mutable).
///
/// See [`.indexed_iter_mut()`](../struct.ArrayBase.html#method.indexed_iter_mut) for more information.
pub struct IndexedIterMut<'a, A, D>(ElementsBaseMut<'a, A, D>);

impl<'a, A, D> IndexedIter<'a, A, D>
where
    D: Dimension,
{
    pub(crate) fn new(x: ElementsBase<'a, A, D>) -> Self {
        IndexedIter(x)
    }
}

impl<'a, A, D> IndexedIterMut<'a, A, D>
where
    D: Dimension,
{
    pub(crate) fn new(x: ElementsBaseMut<'a, A, D>) -> Self {
        IndexedIterMut(x)
    }
}

impl<'a, A, D: Dimension> Iterator for Iter<'a, A, D> {
    type Item = &'a A;
    #[inline]
    fn next(&mut self) -> Option<&'a A> {
        either_mut!(self.inner, iter => iter.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        either!(self.inner, ref iter => iter.size_hint())
    }

    fn fold<Acc, G>(self, init: Acc, g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        either!(self.inner, iter => iter.fold(init, g))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        either_mut!(self.inner, iter => iter.nth(n))
    }

    fn collect<B>(self) -> B
    where
        B: FromIterator<Self::Item>,
    {
        either!(self.inner, iter => iter.collect())
    }

    fn all<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        either_mut!(self.inner, iter => iter.all(f))
    }

    fn any<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        either_mut!(self.inner, iter => iter.any(f))
    }

    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        either_mut!(self.inner, iter => iter.find(predicate))
    }

    fn find_map<B, F>(&mut self, f: F) -> Option<B>
    where
        F: FnMut(Self::Item) -> Option<B>,
    {
        either_mut!(self.inner, iter => iter.find_map(f))
    }

    fn count(self) -> usize {
        either!(self.inner, iter => iter.count())
    }

    fn last(self) -> Option<Self::Item> {
        either!(self.inner, iter => iter.last())
    }

    fn position<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
    {
        either_mut!(self.inner, iter => iter.position(predicate))
    }
}

impl<'a, A> DoubleEndedIterator for Iter<'a, A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a A> {
        either_mut!(self.inner, iter => iter.next_back())
    }

    fn nth_back(&mut self, n: usize) -> Option<&'a A> {
        either_mut!(self.inner, iter => iter.nth_back(n))
    }

    fn rfold<Acc, G>(self, init: Acc, g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        either!(self.inner, iter => iter.rfold(init, g))
    }
}

impl<'a, A, D> ExactSizeIterator for Iter<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        either!(self.inner, ref iter => iter.len())
    }
}

impl<'a, A, D: Dimension> Iterator for IndexedIter<'a, A, D> {
    type Item = (D::Pattern, &'a A);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = match self.0.inner.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        match self.0.next() {
            None => None,
            Some(elem) => Some((index.into_pattern(), elem)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, A, D> ExactSizeIterator for IndexedIter<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.0.inner.len()
    }
}

impl<'a, A, D: Dimension> Iterator for IterMut<'a, A, D> {
    type Item = &'a mut A;
    #[inline]
    fn next(&mut self) -> Option<&'a mut A> {
        either_mut!(self.inner, iter => iter.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        either!(self.inner, ref iter => iter.size_hint())
    }

    fn fold<Acc, G>(self, init: Acc, g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        either!(self.inner, iter => iter.fold(init, g))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        either_mut!(self.inner, iter => iter.nth(n))
    }

    fn collect<B>(self) -> B
    where
        B: FromIterator<Self::Item>,
    {
        either!(self.inner, iter => iter.collect())
    }

    fn all<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        either_mut!(self.inner, iter => iter.all(f))
    }

    fn any<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        either_mut!(self.inner, iter => iter.any(f))
    }

    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        either_mut!(self.inner, iter => iter.find(predicate))
    }

    fn find_map<B, F>(&mut self, f: F) -> Option<B>
    where
        F: FnMut(Self::Item) -> Option<B>,
    {
        either_mut!(self.inner, iter => iter.find_map(f))
    }

    fn count(self) -> usize {
        either!(self.inner, iter => iter.count())
    }

    fn last(self) -> Option<Self::Item> {
        either!(self.inner, iter => iter.last())
    }

    fn position<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
    {
        either_mut!(self.inner, iter => iter.position(predicate))
    }
}

impl<'a, A> DoubleEndedIterator for IterMut<'a, A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A> {
        either_mut!(self.inner, iter => iter.next_back())
    }

    fn nth_back(&mut self, n: usize) -> Option<&'a mut A> {
        either_mut!(self.inner, iter => iter.nth_back(n))
    }

    fn rfold<Acc, G>(self, init: Acc, g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        either!(self.inner, iter => iter.rfold(init, g))
    }
}

impl<'a, A, D> ExactSizeIterator for IterMut<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        either!(self.inner, ref iter => iter.len())
    }
}

impl<'a, A, D: Dimension> Iterator for ElementsBaseMut<'a, A, D> {
    type Item = &'a mut A;
    #[inline]
    fn next(&mut self) -> Option<&'a mut A> {
        self.inner.next().map(|p| unsafe { &mut *p })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn fold<Acc, G>(self, init: Acc, mut g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        unsafe { self.inner.fold(init, move |acc, ptr| g(acc, &mut *ptr)) }
    }
}

impl<'a, A> DoubleEndedIterator for ElementsBaseMut<'a, A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A> {
        self.inner.next_back().map(|p| unsafe { &mut *p })
    }

    fn rfold<Acc, G>(self, init: Acc, mut g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        unsafe { self.inner.rfold(init, move |acc, ptr| g(acc, &mut *ptr)) }
    }
}

impl<'a, A, D> ExactSizeIterator for ElementsBaseMut<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, A, D: Dimension> Iterator for IndexedIterMut<'a, A, D> {
    type Item = (D::Pattern, &'a mut A);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = match self.0.inner.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        match self.0.next() {
            None => None,
            Some(elem) => Some((index.into_pattern(), elem)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, A, D> ExactSizeIterator for IndexedIterMut<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.0.inner.len()
    }
}


#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use crate::prelude::*;
    use super::Baseiter;
    use super::{ArbitraryOrder, PreserveOrder, NoOptimization};
    use itertools::assert_equal;
    use itertools::Itertools;

    // 3-d axis swaps
    fn swaps() -> impl Iterator<Item=Vec<(usize, usize)>> {
        vec![
            vec![],
            vec![(0, 1)],
            vec![(0, 2)],
            vec![(1, 2)],
            vec![(0, 1), (1, 2)],
            vec![(0, 1), (0, 2)],
        ].into_iter()
    }

    // 3-d axis inverts
    fn inverts() -> impl Iterator<Item=Vec<Axis>> {
        vec![
            vec![],
            vec![Axis(0)],
            vec![Axis(1)],
            vec![Axis(2)],
            vec![Axis(0), Axis(1)],
            vec![Axis(0), Axis(2)],
            vec![Axis(1), Axis(2)],
            vec![Axis(0), Axis(1), Axis(2)],
        ].into_iter()
    }

    #[test]
    fn test_arbitrary_order() {
        for swap in swaps() {
            for invert in inverts() {
                for &slice in &[false, true] {
                    // pattern is 0, 1; 4, 5; 8, 9; etc..
                    let mut a = Array::from_iter(0..24).into_shape((3, 4, 2)).unwrap();
                    if slice {
                        a.slice_collapse(s![.., ..;2, ..]);
                    }
                    for &(i, j) in &swap {
                        a.swap_axes(i, j);
                    }
                    for &i in &invert {
                        a.invert_axis(i);
                    }
                    unsafe {
                        // Should have in-memory order for arbitrary order
                        let iter = Baseiter::new_with_order::<ArbitraryOrder>(a.as_mut_ptr(),
                            a.dim, a.strides);
                        if !slice {
                            assert_equal(iter.map(|ptr| *ptr), 0..a.len());
                        } else {
                            assert_eq!(iter.map(|ptr| *ptr).collect_vec(),
                                (0..a.len() * 2).filter(|&x| (x / 2) % 2 == 0).collect_vec());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_logical_order() {
        for swap in swaps() {
            for invert in inverts() {
                for &slice in &[false, true] {
                    let mut a = Array::from_iter(0..24).into_shape((3, 4, 2)).unwrap();
                    for &(i, j) in &swap {
                        a.swap_axes(i, j);
                    }
                    for &i in &invert {
                        a.invert_axis(i);
                    }
                    if slice {
                        a.slice_collapse(s![.., ..;2, ..]);
                    }

                    unsafe {
                        let mut iter = Baseiter::new_with_order::<NoOptimization>(a.as_mut_ptr(),
                            a.dim, a.strides);
                        let mut index = Dim([0, 0, 0]);
                        let mut elts = 0;
                        while let Some(elt) = iter.next() {
                            assert_eq!(*elt, a[index]);
                            if let Some(index_) = a.raw_dim().next_for(index) {
                                index = index_;
                            }
                            elts += 1;
                        }
                        assert_eq!(elts, a.len());
                    }
                }
            }
        }
    }

    #[test]
    fn test_preserve_order() {
        for swap in swaps() {
            for invert in inverts() {
                for &slice in &[false, true] {
                    let mut a = Array::from_iter(0..20).into_shape((2, 10, 1)).unwrap();
                    for &(i, j) in &swap {
                        a.swap_axes(i, j);
                    }
                    for &i in &invert {
                        a.invert_axis(i);
                    }
                    if slice {
                        a.slice_collapse(s![.., ..;2, ..]);
                    }

                    unsafe {
                        let mut iter = Baseiter::new_with_order::<PreserveOrder>(
                            a.as_mut_ptr(), a.dim, a.strides);

                        // check that axes have been merged (when it's easy to check)
                        if a.shape() == &[2, 10, 1] && invert.is_empty() {
                            assert_eq!(iter.dim, Dim([1, 1, 20]));
                        }

                        let mut index = Dim([0, 0, 0]);
                        let mut elts = 0;
                        while let Some(elt) = iter.next() {
                            assert_eq!(*elt, a[index]);
                            if let Some(index_) = a.raw_dim().next_for(index) {
                                index = index_;
                            }
                            elts += 1;
                        }
                        assert_eq!(elts, a.len());
                    }
                }
            }
        }
    }
}
