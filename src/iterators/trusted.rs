
use std::ptr;
use std::slice;
use alloc::vec::Vec;

use crate::Dimension;
use super::{Iter, IterMut, IntoIter};

/// (Trait used internally) An iterator that we trust
/// to deliver exactly as many items as it said it would.
///
/// The iterator must produce exactly the number of elements it reported or
/// diverge before reaching the end.
pub(crate) unsafe trait TrustedIterator {}

use crate::indexes::IndicesIterF;
use crate::iter::IndicesIter;
#[cfg(feature = "std")]
use crate::{geomspace::Geomspace, linspace::Linspace, logspace::Logspace};
#[cfg(feature = "std")]
unsafe impl<F> TrustedIterator for Linspace<F> {}
#[cfg(feature = "std")]
unsafe impl<F> TrustedIterator for Geomspace<F> {}
#[cfg(feature = "std")]
unsafe impl<F> TrustedIterator for Logspace<F> {}
unsafe impl<'a, A, D> TrustedIterator for Iter<'a, A, D> {}
unsafe impl<'a, A, D> TrustedIterator for IterMut<'a, A, D> {}
unsafe impl<I> TrustedIterator for std::iter::Cloned<I> where I: TrustedIterator {}
unsafe impl<I, F> TrustedIterator for std::iter::Map<I, F> where I: TrustedIterator {}
unsafe impl<'a, A> TrustedIterator for slice::Iter<'a, A> {}
unsafe impl<'a, A> TrustedIterator for slice::IterMut<'a, A> {}
unsafe impl TrustedIterator for ::std::ops::Range<usize> {}
// FIXME: These indices iter are dubious -- size needs to be checked up front.
unsafe impl<D> TrustedIterator for IndicesIter<D> where D: Dimension {}
unsafe impl<D> TrustedIterator for IndicesIterF<D> where D: Dimension {}
unsafe impl<A, D> TrustedIterator for IntoIter<A, D> where D: Dimension {}

/// Like Iterator::collect, but only for trusted length iterators
pub(crate) fn to_vec<I>(iter: I) -> Vec<I::Item>
where
    I: TrustedIterator + ExactSizeIterator,
{
    to_vec_mapped(iter, |x| x)
}

/// Like Iterator::collect, but only for trusted length iterators
pub(crate) fn to_vec_mapped<I, F, B>(iter: I, mut f: F) -> Vec<B>
where
    I: TrustedIterator + ExactSizeIterator,
    F: FnMut(I::Item) -> B,
{
    // Use an `unsafe` block to do this efficiently.
    // We know that iter will produce exactly .size() elements,
    // and the loop can vectorize if it's clean (without branch to grow the vector).
    let (size, _) = iter.size_hint();
    let mut result = Vec::with_capacity(size);
    let mut out_ptr = result.as_mut_ptr();
    let mut len = 0;
    iter.fold((), |(), elt| unsafe {
        ptr::write(out_ptr, f(elt));
        len += 1;
        result.set_len(len);
        out_ptr = out_ptr.offset(1);
    });
    debug_assert_eq!(size, result.len());
    result
}

