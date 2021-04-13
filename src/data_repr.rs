use std::mem;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use alloc::slice;
use alloc::borrow::ToOwned;
use alloc::vec::Vec;
use crate::extension::nonnull;

/// Array's representation.
///
/// *Don’t use this type directly—use the type alias
/// [`Array`](type.Array.html) for the array type!*
// Like a Vec, but with non-unique ownership semantics
//
// repr(C) to make it transmutable OwnedRepr<A> -> OwnedRepr<B> if
// transmutable A -> B.
#[derive(Debug)]
#[repr(C)]
pub struct OwnedRepr<A> {
    ptr: NonNull<A>,
    len: usize,
    capacity: usize,
}

impl<A> OwnedRepr<A> {
    pub(crate) fn from(v: Vec<A>) -> Self {
        let mut v = ManuallyDrop::new(v);
        let len = v.len();
        let capacity = v.capacity();
        let ptr = nonnull::nonnull_from_vec_data(&mut v);
        Self {
            ptr,
            len,
            capacity,
        }
    }

    pub(crate) fn into_vec(self) -> Vec<A> {
        ManuallyDrop::new(self).take_as_vec()
    }

    pub(crate) fn as_slice(&self) -> &[A] {
        unsafe {
            slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }

    pub(crate) fn len(&self) -> usize { self.len }

    pub(crate) fn as_ptr(&self) -> *const A {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_nonnull_mut(&mut self) -> NonNull<A> {
        self.ptr
    }

    /// Cast self into equivalent repr of other element type
    ///
    /// ## Safety
    ///
    /// Caller must ensure the two types have the same representation.
    /// **Panics** if sizes don't match (which is not a sufficient check).
    pub(crate) unsafe fn data_subst<B>(self) -> OwnedRepr<B> {
        // necessary but not sufficient check
        assert_eq!(mem::size_of::<A>(), mem::size_of::<B>());
        let self_ = ManuallyDrop::new(self);
        OwnedRepr {
            ptr: self_.ptr.cast::<B>(),
            len: self_.len,
            capacity: self_.capacity,
        }
    }

    fn take_as_vec(&mut self) -> Vec<A> {
        let capacity = self.capacity;
        let len = self.len;
        self.len = 0;
        self.capacity = 0;
        unsafe {
            Vec::from_raw_parts(self.ptr.as_ptr(), len, capacity)
        }
    }
}

impl<A> Clone for OwnedRepr<A>
    where A: Clone
{
    fn clone(&self) -> Self {
        Self::from(self.as_slice().to_owned())
    }

    fn clone_from(&mut self, other: &Self) {
        let mut v = self.take_as_vec();
        let other = other.as_slice();

        if v.len() > other.len() {
            v.truncate(other.len());
        }
        let (front, back) = other.split_at(v.len());
        v.clone_from_slice(front);
        v.extend_from_slice(back);
        *self = Self::from(v);
    }
}

impl<A> Drop for OwnedRepr<A> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            // correct because: If the elements don't need dropping, an
            // empty Vec is ok. Only the Vec's allocation needs dropping.
            //
            // implemented because: in some places in ndarray
            // where A: Copy (hence does not need drop) we use uninitialized elements in
            // vectors. Setting the length to 0 avoids that the vector tries to
            // drop, slice or otherwise produce values of these elements.
            // (The details of the validity letting this happen with nonzero len, are
            // under discussion as of this writing.)
            if !mem::needs_drop::<A>() {
                self.len = 0;
            }
            // drop as a Vec.
            self.take_as_vec();
        }
    }
}

unsafe impl<A> Sync for OwnedRepr<A> where A: Sync { }
unsafe impl<A> Send for OwnedRepr<A> where A: Send { }

