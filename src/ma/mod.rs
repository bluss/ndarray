use std::cmp::PartialEq;
use std::marker::PhantomData;
use std::ops::{Add, Index};
use crate::{ArrayBase, Array1, RawData, Data, DataOwned, Dimension, NdIndex, Array, DataMut};
use crate::iter::IndexedIter;

/// Enum that represents a value that can potentially be masked.
/// We could potentially use `Option<T>` for that, but that produces
/// weird `Option<Option<T>>` return types in iterators.
/// This type can be converted to `Option<T>` using `into` method.
/// There is also a `PartialEq` implementation just to be able to
/// use it in `assert_eq!` statements.
#[derive(Clone, Copy, Debug, Eq)]
pub enum Masked<T> {
    Value(T),
    Empty,
}

impl<T> Masked<&T> {
    fn cloned(&self) -> Masked<T>
    where
        T: Clone
    {
        match self {
            Masked::Value(v) => Masked::Value((*v).clone()),
            Masked::Empty => Masked::Empty,
        }
    }
}

impl<T> PartialEq for Masked<T>
where
    T: PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Masked::Value(v1), Masked::Value(v2)) => v1.eq(v2),
            (Masked::Empty, Masked::Empty) => true,
            _ => false,
        }
    }
}

impl<T> From<Masked<T>> for Option<T> {
    fn from(other: Masked<T>) -> Option<T> {
        match other {
            Masked::Value(v) => Some(v),
            Masked::Empty => None,
        }
    }
}

/// Every struct that can be used as a mask should implement this trait.
/// It has two generic parameters:
///     A - type of the values to be masked
///     D - dimension of the mask
/// The trait is implemented in such a way so that it could be implemented
/// by different types, not just variations of `ArrayBase`. For example,
/// we can implement a mask as a whitelist/blacklist of indices or as a
/// struct which treats some value or range of values as a mask.
pub trait Mask<A, D> {
    /// Given an index of the element and a reference to it, return masked
    /// version of the reference. Accepting a pair allows masking by index,
    /// value or both.
    fn mask_ref<'a, I: NdIndex<D>>(&self, pair: (I, &'a A)) -> Masked<&'a A>;

    // Probably we will need two more methods to be able to mask by value and
    // by mutable reference:

    // fn mask<I: NdIndex<D>>(&self, pair: (I, A)) -> Masked<A>;
    // fn mask_ref_mut<'a, I: NdIndex<D>>(&self, pair: (I, &'a mut A)) -> Masked<&'a mut A>;

    fn mask_iter<'a, 'b: 'a, I>(&'b self, iter: I) -> MaskedIter<'a, A, Self, I, D>
    where
        I: Iterator<Item = (D::Pattern, &'a A)>,
        D: Dimension,
        D::Pattern: NdIndex<D>,
    {
        MaskedIter::new(self, iter)
    }
}

/// Given two masks, generate their intersection. This may be required for any
/// binary operations with two masks.
pub trait JoinMask<A, D, M> : Mask<A, D>
where
    M: Mask<A, D>
{
    type Output: Mask<A, D>;

    fn join(&self, other: &M) -> Self::Output;
}

pub struct MaskedIter<'a, A: 'a, M, I, D>
where
    I: Iterator<Item = (D::Pattern, &'a A)>,
    D: Dimension,
    D::Pattern: NdIndex<D>,
    M: ?Sized + Mask<A, D>
{
    mask: &'a M,
    iter: I,
    _dim: PhantomData<D>,
}

impl<'a, A, M, I, D> MaskedIter<'a, A, M, I, D>
where
    I: Iterator<Item = (D::Pattern, &'a A)>,
    D: Dimension,
    D::Pattern: NdIndex<D>,
    M: ?Sized + Mask<A, D>
{
    fn new(mask: &'a M, iter: I) -> MaskedIter<'a, A, M, I, D> {
        MaskedIter { mask, iter, _dim: PhantomData }
    }
}

impl<'a, A, M, I, D> Iterator for MaskedIter<'a, A, M, I, D>
where
    I: Iterator<Item = (D::Pattern, &'a A)>,
    D: Dimension,
    D::Pattern: NdIndex<D>,
    M: Mask<A, D>
{
    type Item = Masked<&'a A>;

    fn next(&mut self) -> Option<Self::Item> {
        let nex_val = self.iter.next()?;
        Some(self.mask.mask_ref(nex_val))
    }
}

/// First implementation of the mask as a bool array of the same shape.
impl<A, S, D> Mask<A, D> for ArrayBase<S, D>
where
    D: Dimension,
    S: Data<Elem = bool>,
{
    fn mask_ref<'a, I: NdIndex<D>>(&self, pair: (I, &'a A)) -> Masked<&'a A> {
        if *self.index(pair.0) { Masked::Value(pair.1) } else { Masked::Empty }
    }
}

impl<A, S1, S2, D> JoinMask<A, D, ArrayBase<S1, D>> for ArrayBase<S2, D>
where
    D: Dimension,
    S1: Data<Elem = bool>,
    S2: Data<Elem = bool>,
{
    type Output = Array<bool, D>;

    fn join(&self, other: &ArrayBase<S1, D>) -> Self::Output {
        self & other
    }
}

/// Base type for masked array. `S` and `D` types are exactly the ones
/// of `ArrayBase`, `M` is a mask type.
pub struct MaskedArrayBase<S, D, M>
where
    S: RawData,
    M: Mask<S::Elem, D>,
{
    data: ArrayBase<S, D>,
    mask: M,
}

impl<S, D, M> MaskedArrayBase<S, D, M>
where
    S: RawData,
    D: Dimension,
    M: Mask<S::Elem, D>,
{
    pub fn compressed(&self) -> Array1<S::Elem>
    where
        S::Elem: Clone,
        S: Data,
        D::Pattern: NdIndex<D>,
    {
        self.iter()
            .filter_map(|mv: Masked<&S::Elem>| mv.cloned().into())
            .collect()
    }

    pub fn iter(&self) -> MaskedIter<'_, S::Elem, M, IndexedIter<'_, S::Elem, D>, D>
    where
        S: Data,
        D::Pattern: NdIndex<D>,
    {
        self.mask.mask_iter(self.data.indexed_iter())
    }
}

impl<A, S1, S2, D, M> Add<MaskedArrayBase<S2, D, M>> for MaskedArrayBase<S1, D, M>
where
    A: Clone + Add<A, Output = A>,
    S1: DataOwned<Elem = A> + DataMut,
    S2: Data<Elem = A>,
    D: Dimension,
    M: Mask<A, D> + JoinMask<A, D, M>,
{
    type Output = MaskedArrayBase<S1, D, <M as JoinMask<A, D, M>>::Output>;

    fn add(self, rhs: MaskedArrayBase<S2, D, M>) -> Self::Output {
        MaskedArrayBase {
            data: self.data + rhs.data,
            mask: self.mask.join(&rhs.mask),
        }
    }
}

pub fn array<S, D, M>(data: ArrayBase<S, D>, mask: M) -> MaskedArrayBase<S, D, M>
where
    S: RawData,
    M: Mask<S::Elem, D>,
{
    MaskedArrayBase { data, mask }
}
