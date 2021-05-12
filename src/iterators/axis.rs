
use crate::imp_prelude::*;
use crate::{NdProducer, RemoveAxis};

use std::marker::PhantomData;

#[derive(Debug)]
struct AxisIterCore<A, D> {
    /// Index along the axis of the value of `.next()`, relative to the start
    /// of the axis.
    index: Ix,
    /// (Exclusive) upper bound on `index`. Initially, this is equal to the
    /// length of the axis.
    end: Ix,
    /// Stride along the axis (offset between consecutive pointers).
    stride: Ixs,
    /// Shape of the iterator's items.
    inner_dim: D,
    /// Strides of the iterator's items.
    inner_strides: D,
    /// Pointer corresponding to `index == 0`.
    ptr: *mut A,
}

clone_bounds!(
    [A, D: Clone]
    AxisIterCore[A, D] {
        @copy {
            index,
            end,
            stride,
            ptr,
        }
        inner_dim,
        inner_strides,
    }
);

impl<A, D: Dimension> AxisIterCore<A, D> {
    /// Constructs a new iterator over the specified axis.
    fn new<S, Di>(v: ArrayBase<S, Di>, axis: Axis) -> Self
    where
        Di: RemoveAxis<Smaller = D>,
        S: Data<Elem = A>,
    {
        AxisIterCore {
            index: 0,
            end: v.len_of(axis),
            stride: v.stride_of(axis),
            inner_dim: v.dim.remove_axis(axis),
            inner_strides: v.strides.remove_axis(axis),
            ptr: v.ptr.as_ptr(),
        }
    }

    #[inline]
    unsafe fn offset(&self, index: usize) -> *mut A {
        debug_assert!(
            index < self.end,
            "index={}, end={}, stride={}",
            index,
            self.end,
            self.stride
        );
        self.ptr.offset(index as isize * self.stride)
    }

    /// Splits the iterator at `index`, yielding two disjoint iterators.
    ///
    /// `index` is relative to the current state of the iterator (which is not
    /// necessarily the start of the axis).
    ///
    /// **Panics** if `index` is strictly greater than the iterator's remaining
    /// length.
    fn split_at(self, index: usize) -> (Self, Self) {
        assert!(index <= self.len());
        let mid = self.index + index;
        let left = AxisIterCore {
            index: self.index,
            end: mid,
            stride: self.stride,
            inner_dim: self.inner_dim.clone(),
            inner_strides: self.inner_strides.clone(),
            ptr: self.ptr,
        };
        let right = AxisIterCore {
            index: mid,
            end: self.end,
            stride: self.stride,
            inner_dim: self.inner_dim,
            inner_strides: self.inner_strides,
            ptr: self.ptr,
        };
        (left, right)
    }

    /// Does the same thing as `.next()` but also returns the index of the item
    /// relative to the start of the axis.
    fn next_with_index(&mut self) -> Option<(usize, *mut A)> {
        let index = self.index;
        self.next().map(|ptr| (index, ptr))
    }

    /// Does the same thing as `.next_back()` but also returns the index of the
    /// item relative to the start of the axis.
    fn next_back_with_index(&mut self) -> Option<(usize, *mut A)> {
        self.next_back().map(|ptr| (self.end, ptr))
    }
}

impl<A, D> Iterator for AxisIterCore<A, D>
where
    D: Dimension,
{
    type Item = *mut A;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            None
        } else {
            let ptr = unsafe { self.offset(self.index) };
            self.index += 1;
            Some(ptr)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<A, D> DoubleEndedIterator for AxisIterCore<A, D>
where
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            None
        } else {
            let ptr = unsafe { self.offset(self.end - 1) };
            self.end -= 1;
            Some(ptr)
        }
    }
}

impl<A, D> ExactSizeIterator for AxisIterCore<A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.end - self.index
    }
}

/// An iterator that traverses over an axis and
/// and yields each subview.
///
/// The outermost dimension is `Axis(0)`, created with `.outer_iter()`,
/// but you can traverse arbitrary dimension with `.axis_iter()`.
///
/// For example, in a 3 × 5 × 5 array, with `axis` equal to `Axis(2)`,
/// the iterator element is a 3 × 5 subview (and there are 5 in total).
///
/// Iterator element type is `ArrayView<'a, A, D>`.
///
/// See [`.outer_iter()`](../struct.ArrayBase.html#method.outer_iter)
/// or [`.axis_iter()`](../struct.ArrayBase.html#method.axis_iter)
/// for more information.
#[derive(Debug)]
pub struct AxisIter<'a, A, D> {
    iter: AxisIterCore<A, D>,
    life: PhantomData<&'a A>,
}

clone_bounds!(
    ['a, A, D: Clone]
    AxisIter['a, A, D] {
        @copy {
            life,
        }
        iter,
    }
);

impl<'a, A, D: Dimension> AxisIter<'a, A, D> {
    /// Creates a new iterator over the specified axis.
    pub(crate) fn new<Di>(v: ArrayView<'a, A, Di>, axis: Axis) -> Self
    where
        Di: RemoveAxis<Smaller = D>,
    {
        AxisIter {
            iter: AxisIterCore::new(v, axis),
            life: PhantomData,
        }
    }

    /// Splits the iterator at `index`, yielding two disjoint iterators.
    ///
    /// `index` is relative to the current state of the iterator (which is not
    /// necessarily the start of the axis).
    ///
    /// **Panics** if `index` is strictly greater than the iterator's remaining
    /// length.
    pub fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.iter.split_at(index);
        (
            AxisIter {
                iter: left,
                life: self.life,
            },
            AxisIter {
                iter: right,
                life: self.life,
            },
        )
    }
}

impl<'a, A, D> Iterator for AxisIter<'a, A, D>
where
    D: Dimension,
{
    type Item = ArrayView<'a, A, D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| unsafe { self.as_ref(ptr) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D> DoubleEndedIterator for AxisIter<'a, A, D>
where
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|ptr| unsafe { self.as_ref(ptr) })
    }
}

impl<'a, A, D> ExactSizeIterator for AxisIter<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

/// An iterator that traverses over an axis and
/// and yields each subview (mutable)
///
/// The outermost dimension is `Axis(0)`, created with `.outer_iter()`,
/// but you can traverse arbitrary dimension with `.axis_iter()`.
///
/// For example, in a 3 × 5 × 5 array, with `axis` equal to `Axis(2)`,
/// the iterator element is a 3 × 5 subview (and there are 5 in total).
///
/// Iterator element type is `ArrayViewMut<'a, A, D>`.
///
/// See [`.outer_iter_mut()`](../struct.ArrayBase.html#method.outer_iter_mut)
/// or [`.axis_iter_mut()`](../struct.ArrayBase.html#method.axis_iter_mut)
/// for more information.
pub struct AxisIterMut<'a, A, D> {
    iter: AxisIterCore<A, D>,
    life: PhantomData<&'a mut A>,
}

impl<'a, A, D: Dimension> AxisIterMut<'a, A, D> {
    /// Creates a new iterator over the specified axis.
    pub(crate) fn new<Di>(v: ArrayViewMut<'a, A, Di>, axis: Axis) -> Self
    where
        Di: RemoveAxis<Smaller = D>,
    {
        AxisIterMut {
            iter: AxisIterCore::new(v, axis),
            life: PhantomData,
        }
    }

    /// Splits the iterator at `index`, yielding two disjoint iterators.
    ///
    /// `index` is relative to the current state of the iterator (which is not
    /// necessarily the start of the axis).
    ///
    /// **Panics** if `index` is strictly greater than the iterator's remaining
    /// length.
    pub fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.iter.split_at(index);
        (
            AxisIterMut {
                iter: left,
                life: self.life,
            },
            AxisIterMut {
                iter: right,
                life: self.life,
            },
        )
    }
}

impl<'a, A, D> Iterator for AxisIterMut<'a, A, D>
where
    D: Dimension,
{
    type Item = ArrayViewMut<'a, A, D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| unsafe { self.as_ref(ptr) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D> DoubleEndedIterator for AxisIterMut<'a, A, D>
where
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|ptr| unsafe { self.as_ref(ptr) })
    }
}

impl<'a, A, D> ExactSizeIterator for AxisIterMut<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, A, D: Dimension> NdProducer for AxisIter<'a, A, D> {
    type Item = <Self as Iterator>::Item;
    type Dim = Ix1;
    type Ptr = *mut A;
    type Stride = isize;

    #[doc(hidden)]
    fn layout(&self) -> crate::Layout {
        crate::Layout::one_dimensional()
    }
    #[doc(hidden)]
    fn raw_dim(&self) -> Self::Dim {
        Ix1(self.len())
    }
    #[doc(hidden)]
    fn as_ptr(&self) -> Self::Ptr {
        if self.len() > 0 {
            // `self.iter.index` is guaranteed to be in-bounds if any of the
            // iterator remains (i.e. if `self.len() > 0`).
            unsafe { self.iter.offset(self.iter.index) }
        } else {
            // In this case, `self.iter.index` may be past the end, so we must
            // not call `.offset()`. It's okay to return a dangling pointer
            // because it will never be used in the length 0 case.
            std::ptr::NonNull::dangling().as_ptr()
        }
    }

    fn contiguous_stride(&self) -> isize {
        self.iter.stride
    }

    #[doc(hidden)]
    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
        ArrayView::new_(
            ptr,
            self.iter.inner_dim.clone(),
            self.iter.inner_strides.clone(),
        )
    }
    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        self.iter.offset(self.iter.index + i[0])
    }

    #[doc(hidden)]
    fn stride_of(&self, _axis: Axis) -> isize {
        self.contiguous_stride()
    }

    #[doc(hidden)]
    fn split_at(self, _axis: Axis, index: usize) -> (Self, Self) {
        self.split_at(index)
    }
    private_impl! {}
}

impl<'a, A, D: Dimension> NdProducer for AxisIterMut<'a, A, D> {
    type Item = <Self as Iterator>::Item;
    type Dim = Ix1;
    type Ptr = *mut A;
    type Stride = isize;

    #[doc(hidden)]
    fn layout(&self) -> crate::Layout {
        crate::Layout::one_dimensional()
    }
    #[doc(hidden)]
    fn raw_dim(&self) -> Self::Dim {
        Ix1(self.len())
    }
    #[doc(hidden)]
    fn as_ptr(&self) -> Self::Ptr {
        if self.len() > 0 {
            // `self.iter.index` is guaranteed to be in-bounds if any of the
            // iterator remains (i.e. if `self.len() > 0`).
            unsafe { self.iter.offset(self.iter.index) }
        } else {
            // In this case, `self.iter.index` may be past the end, so we must
            // not call `.offset()`. It's okay to return a dangling pointer
            // because it will never be used in the length 0 case.
            std::ptr::NonNull::dangling().as_ptr()
        }
    }

    fn contiguous_stride(&self) -> isize {
        self.iter.stride
    }

    #[doc(hidden)]
    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
        ArrayViewMut::new_(
            ptr,
            self.iter.inner_dim.clone(),
            self.iter.inner_strides.clone(),
        )
    }
    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        self.iter.offset(self.iter.index + i[0])
    }

    #[doc(hidden)]
    fn stride_of(&self, _axis: Axis) -> isize {
        self.contiguous_stride()
    }

    #[doc(hidden)]
    fn split_at(self, _axis: Axis, index: usize) -> (Self, Self) {
        self.split_at(index)
    }
    private_impl! {}
}

/// An iterator that traverses over the specified axis
/// and yields views of the specified size on this axis.
///
/// For example, in a 2 × 8 × 3 array, if the axis of iteration
/// is 1 and the chunk size is 2, the yielded elements
/// are 2 × 2 × 3 views (and there are 4 in total).
///
/// Iterator element type is `ArrayView<'a, A, D>`.
///
/// See [`.axis_chunks_iter()`](../struct.ArrayBase.html#method.axis_chunks_iter) for more information.
pub struct AxisChunksIter<'a, A, D> {
    iter: AxisIterCore<A, D>,
    /// Index of the partial chunk (the chunk smaller than the specified chunk
    /// size due to the axis length not being evenly divisible). If the axis
    /// length is evenly divisible by the chunk size, this index is larger than
    /// the maximum valid index.
    partial_chunk_index: usize,
    /// Dimension of the partial chunk.
    partial_chunk_dim: D,
    life: PhantomData<&'a A>,
}

clone_bounds!(
    ['a, A, D: Clone]
    AxisChunksIter['a, A, D] {
        @copy {
            life,
            partial_chunk_index,
        }
        iter,
        partial_chunk_dim,
    }
);

/// Computes the information necessary to construct an iterator over chunks
/// along an axis, given a `view` of the array, the `axis` to iterate over, and
/// the chunk `size`.
///
/// Returns an axis iterator with the correct stride to move between chunks,
/// the number of chunks, and the shape of the last chunk.
///
/// **Panics** if `size == 0`.
fn chunk_iter_parts<A, D: Dimension>(
    v: ArrayView<'_, A, D>,
    axis: Axis,
    size: usize,
) -> (AxisIterCore<A, D>, usize, D) {
    assert_ne!(size, 0, "Chunk size must be nonzero.");
    let axis_len = v.len_of(axis);
    let n_whole_chunks = axis_len / size;
    let chunk_remainder = axis_len % size;
    let iter_len = if chunk_remainder == 0 {
        n_whole_chunks
    } else {
        n_whole_chunks + 1
    };
    let stride = if n_whole_chunks == 0 {
        // This case avoids potential overflow when `size > axis_len`.
        0
    } else {
        v.stride_of(axis) * size as isize
    };

    let axis = axis.index();
    let mut inner_dim = v.dim.clone();
    inner_dim[axis] = size;

    let mut partial_chunk_dim = v.dim;
    partial_chunk_dim[axis] = chunk_remainder;
    let partial_chunk_index = n_whole_chunks;

    let iter = AxisIterCore {
        index: 0,
        end: iter_len,
        stride,
        inner_dim,
        inner_strides: v.strides,
        ptr: v.ptr.as_ptr(),
    };

    (iter, partial_chunk_index, partial_chunk_dim)
}

impl<'a, A, D: Dimension> AxisChunksIter<'a, A, D> {
    pub(crate) fn new(v: ArrayView<'a, A, D>, axis: Axis, size: usize) -> Self {
        let (iter, partial_chunk_index, partial_chunk_dim) = chunk_iter_parts(v, axis, size);
        AxisChunksIter {
            iter,
            partial_chunk_index,
            partial_chunk_dim,
            life: PhantomData,
        }
    }
}

macro_rules! chunk_iter_impl {
    ($iter:ident, $array:ident) => {
        impl<'a, A, D> $iter<'a, A, D>
        where
            D: Dimension,
        {
            fn get_subview(&self, index: usize, ptr: *mut A) -> $array<'a, A, D> {
                if index != self.partial_chunk_index {
                    unsafe {
                        $array::new_(
                            ptr,
                            self.iter.inner_dim.clone(),
                            self.iter.inner_strides.clone(),
                        )
                    }
                } else {
                    unsafe {
                        $array::new_(
                            ptr,
                            self.partial_chunk_dim.clone(),
                            self.iter.inner_strides.clone(),
                        )
                    }
                }
            }

            /// Splits the iterator at index, yielding two disjoint iterators.
            ///
            /// `index` is relative to the current state of the iterator (which is not
            /// necessarily the start of the axis).
            ///
            /// **Panics** if `index` is strictly greater than the iterator's remaining
            /// length.
            pub fn split_at(self, index: usize) -> (Self, Self) {
                let (left, right) = self.iter.split_at(index);
                (
                    Self {
                        iter: left,
                        partial_chunk_index: self.partial_chunk_index,
                        partial_chunk_dim: self.partial_chunk_dim.clone(),
                        life: self.life,
                    },
                    Self {
                        iter: right,
                        partial_chunk_index: self.partial_chunk_index,
                        partial_chunk_dim: self.partial_chunk_dim,
                        life: self.life,
                    },
                )
            }
        }

        impl<'a, A, D> Iterator for $iter<'a, A, D>
        where
            D: Dimension,
        {
            type Item = $array<'a, A, D>;

            fn next(&mut self) -> Option<Self::Item> {
                self.iter
                    .next_with_index()
                    .map(|(index, ptr)| self.get_subview(index, ptr))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.iter.size_hint()
            }
        }

        impl<'a, A, D> DoubleEndedIterator for $iter<'a, A, D>
        where
            D: Dimension,
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.iter
                    .next_back_with_index()
                    .map(|(index, ptr)| self.get_subview(index, ptr))
            }
        }

        impl<'a, A, D> ExactSizeIterator for $iter<'a, A, D> where D: Dimension {}
    };
}

chunk_iter_impl!(AxisChunksIter, ArrayView);
chunk_iter_impl!(AxisChunksIterMut, ArrayViewMut);

/// An iterator that traverses over the specified axis
/// and yields mutable views of the specified size on this axis.
///
/// For example, in a 2 × 8 × 3 array, if the axis of iteration
/// is 1 and the chunk size is 2, the yielded elements
/// are 2 × 2 × 3 views (and there are 4 in total).
///
/// Iterator element type is `ArrayViewMut<'a, A, D>`.
///
/// See [`.axis_chunks_iter_mut()`](../struct.ArrayBase.html#method.axis_chunks_iter_mut)
/// for more information.
pub struct AxisChunksIterMut<'a, A, D> {
    iter: AxisIterCore<A, D>,
    partial_chunk_index: usize,
    partial_chunk_dim: D,
    life: PhantomData<&'a mut A>,
}

impl<'a, A, D: Dimension> AxisChunksIterMut<'a, A, D> {
    pub(crate) fn new(v: ArrayViewMut<'a, A, D>, axis: Axis, size: usize) -> Self {
        let (iter, partial_chunk_index, partial_chunk_dim) =
            chunk_iter_parts(v.into_view(), axis, size);
        AxisChunksIterMut {
            iter,
            partial_chunk_index,
            partial_chunk_dim,
            life: PhantomData,
        }
    }
}

