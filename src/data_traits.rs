// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The data (inner representation) traits for ndarray

use rawpointer::PointerExt;

use std::mem::{self, size_of};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::{ArrayBase, CowRepr, Dimension, OwnedArcRepr, OwnedRepr, RawViewRepr, ViewRepr};

/// Array representation trait.
///
/// For an array that meets the invariants of the `ArrayBase` type. This trait
/// does not imply any ownership or lifetime; pointers to elements in the array
/// may not be safe to dereference.
///
/// ***Note:*** `RawData` is not an extension interface at this point.
/// Traits in Rust can serve many different roles. This trait is public because
/// it is used as a bound on public methods.
pub unsafe trait RawData: Sized {
    /// The array element type.
    type Elem;

    #[doc(hidden)]
    // This method is only used for debugging
    fn _data_slice(&self) -> Option<&[Self::Elem]>;

    private_decl! {}
}

/// Array representation trait.
///
/// For an array with writable elements.
///
/// ***Internal trait, see `RawData`.***
pub unsafe trait RawDataMut: RawData {
    /// If possible, ensures that the array has unique access to its data.
    ///
    /// If `Self` provides safe mutable access to array elements, then it
    /// **must** panic or ensure that the data is unique.
    #[doc(hidden)]
    fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension;

    /// If possible, returns whether the array has unique access to its data.
    ///
    /// If `Self` provides safe mutable access to array elements, then it
    /// **must** return `Some(_)`.
    #[doc(hidden)]
    fn try_is_unique(&mut self) -> Option<bool>;
}

/// Array representation trait.
///
/// An array representation that can be cloned.
///
/// ***Internal trait, see `RawData`.***
pub unsafe trait RawDataClone: RawData {
    #[doc(hidden)]
    /// Unsafe because, `ptr` must point inside the current storage.
    unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>);

    #[doc(hidden)]
    unsafe fn clone_from_with_ptr(
        &mut self,
        other: &Self,
        ptr: NonNull<Self::Elem>,
    ) -> NonNull<Self::Elem> {
        let (data, ptr) = other.clone_with_ptr(ptr);
        *self = data;
        ptr
    }
}

/// Array representation trait.
///
/// For an array with elements that can be accessed with safe code.
///
/// ***Internal trait, see `RawData`.***
pub unsafe trait Data: RawData {
    /// Converts the array to a uniquely owned array, cloning elements if necessary.
    #[doc(hidden)]
    #[allow(clippy::wrong_self_convention)]
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> ArrayBase<OwnedRepr<Self::Elem>, D>
    where
        Self::Elem: Clone,
        D: Dimension;

    /// Return a shared ownership (copy on write) array based on the existing one,
    /// cloning elements if necessary.
    #[doc(hidden)]
    #[allow(clippy::wrong_self_convention)]
    fn to_shared<D>(self_: &ArrayBase<Self, D>) -> ArrayBase<OwnedArcRepr<Self::Elem>, D>
    where
        Self::Elem: Clone,
        D: Dimension,
    {
        // clone to shared
        self_.to_owned().into_shared()
    }
}

/// Array representation trait.
///
/// For an array with writable elements that can be accessed with safe code.
///
/// ***Internal trait, see `Data`.***
//
// # For implementers
//
// If you implement the `DataMut` trait, you are guaranteeing that the
// `RawDataMut::try_ensure_unique` implementation always panics or ensures that
// the data is unique. You are also guaranteeing that `try_is_unique` always
// returns `Some(_)`.
pub unsafe trait DataMut: Data + RawDataMut {
    /// Ensures that the array has unique access to its data.
    #[doc(hidden)]
    #[inline]
    fn ensure_unique<D>(self_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
        Self::try_ensure_unique(self_)
    }

    /// Returns whether the array has unique access to its data.
    #[doc(hidden)]
    #[inline]
    fn is_unique(&mut self) -> bool {
        self.try_is_unique().unwrap()
    }
}

unsafe impl<A> RawData for RawViewRepr<*const A> {
    type Elem = A;
    fn _data_slice(&self) -> Option<&[A]> {
        None
    }
    private_impl! {}
}

unsafe impl<A> RawDataClone for RawViewRepr<*const A> {
    unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
        (*self, ptr)
    }
}

unsafe impl<A> RawData for RawViewRepr<*mut A> {
    type Elem = A;
    fn _data_slice(&self) -> Option<&[A]> {
        None
    }
    private_impl! {}
}

unsafe impl<A> RawDataMut for RawViewRepr<*mut A> {
    #[inline]
    fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
    }

    #[inline]
    fn try_is_unique(&mut self) -> Option<bool> {
        None
    }
}

unsafe impl<A> RawDataClone for RawViewRepr<*mut A> {
    unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
        (*self, ptr)
    }
}

unsafe impl<A> RawData for OwnedArcRepr<A> {
    type Elem = A;
    fn _data_slice(&self) -> Option<&[A]> {
        Some(self.0.as_slice())
    }
    private_impl! {}
}

// NOTE: Copy on write
unsafe impl<A> RawDataMut for OwnedArcRepr<A>
where
    A: Clone,
{
    fn try_ensure_unique<D>(self_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
        if Arc::get_mut(&mut self_.data.0).is_some() {
            return;
        }
        if self_.dim.size() <= self_.data.0.len() / 2 {
            // Create a new vec if the current view is less than half of
            // backing data.
            unsafe {
                *self_ = ArrayBase::from_shape_vec_unchecked(
                    self_.dim.clone(),
                    self_.iter().cloned().collect(),
                );
            }
            return;
        }
        let rcvec = &mut self_.data.0;
        let a_size = mem::size_of::<A>() as isize;
        let our_off = if a_size != 0 {
            (self_.ptr.as_ptr() as isize - rcvec.as_ptr() as isize) / a_size
        } else {
            0
        };
        let rvec = Arc::make_mut(rcvec);
        unsafe {
            self_.ptr = rvec.as_nonnull_mut().offset(our_off);
        }
    }

    fn try_is_unique(&mut self) -> Option<bool> {
        Some(Arc::get_mut(&mut self.0).is_some())
    }
}

unsafe impl<A> Data for OwnedArcRepr<A> {
    fn into_owned<D>(mut self_: ArrayBase<Self, D>) -> ArrayBase<OwnedRepr<Self::Elem>, D>
    where
        A: Clone,
        D: Dimension,
    {
        Self::ensure_unique(&mut self_);
        let data = Arc::try_unwrap(self_.data.0).ok().unwrap();
        // safe because data is equivalent
        unsafe {
            ArrayBase::from_data_ptr(data, self_.ptr)
                .with_strides_dim(self_.strides, self_.dim)
        }
    }

    fn to_shared<D>(self_: &ArrayBase<Self, D>) -> ArrayBase<OwnedArcRepr<Self::Elem>, D>
    where
        Self::Elem: Clone,
        D: Dimension,
    {
        // to shared using clone of OwnedArcRepr without clone of raw data.
        self_.clone()
    }
}

unsafe impl<A> DataMut for OwnedArcRepr<A> where A: Clone {}

unsafe impl<A> RawDataClone for OwnedArcRepr<A> {
    unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
        // pointer is preserved
        (self.clone(), ptr)
    }
}

unsafe impl<A> RawData for OwnedRepr<A> {
    type Elem = A;
    fn _data_slice(&self) -> Option<&[A]> {
        Some(self.as_slice())
    }
    private_impl! {}
}

unsafe impl<A> RawDataMut for OwnedRepr<A> {
    #[inline]
    fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
    }

    #[inline]
    fn try_is_unique(&mut self) -> Option<bool> {
        Some(true)
    }
}

unsafe impl<A> Data for OwnedRepr<A> {
    #[inline]
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> ArrayBase<OwnedRepr<Self::Elem>, D>
    where
        A: Clone,
        D: Dimension,
    {
        self_
    }
}

unsafe impl<A> DataMut for OwnedRepr<A> {}

unsafe impl<A> RawDataClone for OwnedRepr<A>
where
    A: Clone,
{
    unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
        let mut u = self.clone();
        let mut new_ptr = u.as_nonnull_mut();
        if size_of::<A>() != 0 {
            let our_off =
                (ptr.as_ptr() as isize - self.as_ptr() as isize) / mem::size_of::<A>() as isize;
            new_ptr = new_ptr.offset(our_off);
        }
        (u, new_ptr)
    }

    unsafe fn clone_from_with_ptr(
        &mut self,
        other: &Self,
        ptr: NonNull<Self::Elem>,
    ) -> NonNull<Self::Elem> {
        let our_off = if size_of::<A>() != 0 {
            (ptr.as_ptr() as isize - other.as_ptr() as isize) / mem::size_of::<A>() as isize
        } else {
            0
        };
        self.clone_from(&other);
        self.as_nonnull_mut().offset(our_off)
    }
}

unsafe impl<'a, A> RawData for ViewRepr<&'a A> {
    type Elem = A;
    fn _data_slice(&self) -> Option<&[A]> {
        None
    }
    private_impl! {}
}

unsafe impl<'a, A> Data for ViewRepr<&'a A> {
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> ArrayBase<OwnedRepr<Self::Elem>, D>
    where
        Self::Elem: Clone,
        D: Dimension,
    {
        self_.to_owned()
    }
}

unsafe impl<'a, A> RawDataClone for ViewRepr<&'a A> {
    unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
        (*self, ptr)
    }
}

unsafe impl<'a, A> RawData for ViewRepr<&'a mut A> {
    type Elem = A;
    fn _data_slice(&self) -> Option<&[A]> {
        None
    }
    private_impl! {}
}

unsafe impl<'a, A> RawDataMut for ViewRepr<&'a mut A> {
    #[inline]
    fn try_ensure_unique<D>(_: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
    }

    #[inline]
    fn try_is_unique(&mut self) -> Option<bool> {
        Some(true)
    }
}

unsafe impl<'a, A> Data for ViewRepr<&'a mut A> {
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> ArrayBase<OwnedRepr<Self::Elem>, D>
    where
        Self::Elem: Clone,
        D: Dimension,
    {
        self_.to_owned()
    }
}

unsafe impl<'a, A> DataMut for ViewRepr<&'a mut A> {}

/// Array representation trait.
///
/// A representation that is a unique or shared owner of its data.
///
/// ***Internal trait, see `Data`.***
// The owned storage represents the ownership and allocation of the array's elements.
// The storage may be unique or shared ownership style; it must be an aliasable owner
// (permit aliasing pointers, such as our separate array head pointer).
//
// The array storage must be initially mutable - copy on write arrays may require copying for
// unsharing storage before mutating it. The initially allocated storage must be mutable so
// that it can be mutated directly - through .raw_view_mut_unchecked() - for initialization.
pub unsafe trait DataOwned: Data {
    /// Corresponding owned data with MaybeUninit elements
    type MaybeUninit: DataOwned<Elem = MaybeUninit<Self::Elem>>
        + RawDataSubst<Self::Elem, Output=Self>;
    #[doc(hidden)]
    fn new(elements: Vec<Self::Elem>) -> Self;

    /// Converts the data representation to a shared (copy on write)
    /// representation, without any copying.
    #[doc(hidden)]
    fn into_shared(self) -> OwnedArcRepr<Self::Elem>;
}

/// Array representation trait.
///
/// A representation that is a lightweight view.
///
/// ***Internal trait, see `Data`.***
pub unsafe trait DataShared: Clone + Data + RawDataClone {}

unsafe impl<A> DataShared for OwnedArcRepr<A> {}
unsafe impl<'a, A> DataShared for ViewRepr<&'a A> {}

unsafe impl<A> DataOwned for OwnedRepr<A> {
    type MaybeUninit = OwnedRepr<MaybeUninit<A>>;

    fn new(elements: Vec<A>) -> Self {
        OwnedRepr::from(elements)
    }

    fn into_shared(self) -> OwnedArcRepr<A> {
        OwnedArcRepr(Arc::new(self))
    }
}

unsafe impl<A> DataOwned for OwnedArcRepr<A> {
    type MaybeUninit = OwnedArcRepr<MaybeUninit<A>>;

    fn new(elements: Vec<A>) -> Self {
        OwnedArcRepr(Arc::new(OwnedRepr::from(elements)))
    }

    fn into_shared(self) -> OwnedArcRepr<A> {
        self
    }
}

unsafe impl<'a, A> RawData for CowRepr<'a, A> {
    type Elem = A;
    fn _data_slice(&self) -> Option<&[A]> {
        match self {
            CowRepr::View(view) => view._data_slice(),
            CowRepr::Owned(data) => data._data_slice(),
        }
    }
    private_impl! {}
}

unsafe impl<'a, A> RawDataMut for CowRepr<'a, A>
where
    A: Clone,
{
    #[inline]
    fn try_ensure_unique<D>(array: &mut ArrayBase<Self, D>)
    where
        Self: Sized,
        D: Dimension,
    {
        match array.data {
            CowRepr::View(_) => {
                let owned = array.to_owned();
                array.data = CowRepr::Owned(owned.data);
                array.ptr = owned.ptr;
                array.dim = owned.dim;
                array.strides = owned.strides;
            }
            CowRepr::Owned(_) => {}
        }
    }

    #[inline]
    fn try_is_unique(&mut self) -> Option<bool> {
        Some(self.is_owned())
    }
}

unsafe impl<'a, A> RawDataClone for CowRepr<'a, A>
where
    A: Clone,
{
    unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
        match self {
            CowRepr::View(view) => {
                let (new_view, ptr) = view.clone_with_ptr(ptr);
                (CowRepr::View(new_view), ptr)
            }
            CowRepr::Owned(data) => {
                let (new_data, ptr) = data.clone_with_ptr(ptr);
                (CowRepr::Owned(new_data), ptr)
            }
        }
    }

    #[doc(hidden)]
    unsafe fn clone_from_with_ptr(
        &mut self,
        other: &Self,
        ptr: NonNull<Self::Elem>,
    ) -> NonNull<Self::Elem> {
        match (&mut *self, other) {
            (CowRepr::View(self_), CowRepr::View(other)) => self_.clone_from_with_ptr(other, ptr),
            (CowRepr::Owned(self_), CowRepr::Owned(other)) => self_.clone_from_with_ptr(other, ptr),
            (_, CowRepr::Owned(other)) => {
                let (cloned, ptr) = other.clone_with_ptr(ptr);
                *self = CowRepr::Owned(cloned);
                ptr
            }
            (_, CowRepr::View(other)) => {
                let (cloned, ptr) = other.clone_with_ptr(ptr);
                *self = CowRepr::View(cloned);
                ptr
            }
        }
    }
}

unsafe impl<'a, A> Data for CowRepr<'a, A> {
    #[inline]
    fn into_owned<D>(self_: ArrayBase<CowRepr<'a, A>, D>) -> ArrayBase<OwnedRepr<Self::Elem>, D>
    where
        A: Clone,
        D: Dimension,
    {
        match self_.data {
            CowRepr::View(_) => self_.to_owned(),
            CowRepr::Owned(data) => unsafe {
                // safe because the data is equivalent so ptr, dims remain valid
                ArrayBase::from_data_ptr(data, self_.ptr)
                    .with_strides_dim(self_.strides, self_.dim)
            },
        }
    }
}

unsafe impl<'a, A> DataMut for CowRepr<'a, A> where A: Clone {}

/// Array representation trait.
///
/// The RawDataSubst trait maps the element type of array storage, while
/// keeping the same kind of storage.
///
/// For example, `RawDataSubst<B>` can map the type `OwnedRepr<A>` to `OwnedRepr<B>`.
pub trait RawDataSubst<A>: RawData {
    /// The resulting array storage of the same kind but substituted element type
    type Output: RawData<Elem = A>;

    /// Unsafely translate the data representation from one element
    /// representation to another.
    ///
    /// ## Safety
    ///
    /// Caller must ensure the two types have the same representation.
    unsafe fn data_subst(self) -> Self::Output;
}

impl<A, B> RawDataSubst<B> for OwnedRepr<A> {
    type Output = OwnedRepr<B>;

    unsafe fn data_subst(self) -> Self::Output {
        self.data_subst()
    }
}

impl<A, B> RawDataSubst<B> for OwnedArcRepr<A> {
    type Output = OwnedArcRepr<B>;

    unsafe fn data_subst(self) -> Self::Output {
        OwnedArcRepr(Arc::from_raw(Arc::into_raw(self.0) as *const OwnedRepr<B>))
    }
}

impl<A, B> RawDataSubst<B> for RawViewRepr<*const A> {
    type Output = RawViewRepr<*const B>;

    unsafe fn data_subst(self) -> Self::Output {
        RawViewRepr::new()
    }
}

impl<A, B> RawDataSubst<B> for RawViewRepr<*mut A> {
    type Output = RawViewRepr<*mut B>;

    unsafe fn data_subst(self) -> Self::Output {
        RawViewRepr::new()
    }
}

impl<'a, A: 'a, B: 'a> RawDataSubst<B> for ViewRepr<&'a A> {
    type Output = ViewRepr<&'a B>;

    unsafe fn data_subst(self) -> Self::Output {
        ViewRepr::new()
    }
}

impl<'a, A: 'a, B: 'a> RawDataSubst<B> for ViewRepr<&'a mut A> {
    type Output = ViewRepr<&'a mut B>;

    unsafe fn data_subst(self) -> Self::Output {
        ViewRepr::new()
    }
}

