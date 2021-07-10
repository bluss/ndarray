// Copyright 2016-2019 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Constructors for randomized arrays: `rand` integration for `ndarray`.
//!
//! See **[`RandomExt`]** for usage examples.
//!
//! ## Note
//!
//! `ndarray-rand` depends on [`rand` 0.8][rand].
//!
//! [`rand`][rand] and [`rand_distr`][rand_distr]
//! are re-exported as sub-modules, [`ndarray_rand::rand`](rand)
//! and [`ndarray_rand::rand_distr`](rand_distr) respectively.
//! You can use these submodules for guaranteed version compatibility or
//! convenience.
//!
//! [rand]: https://docs.rs/rand/0.8
//! [rand_distr]: https://docs.rs/rand_distr/0.4
//!
//! If you want to use a random number generator or distribution from another crate
//! with `ndarray-rand`, you need to make sure that the other crate also depends on the
//! same version of `rand`. Otherwise, the compiler will return errors saying
//! that the items are not compatible (e.g. that a type doesn't implement a
//! necessary trait).

use crate::rand::distributions::{Distribution, Uniform};
use crate::rand::rngs::SmallRng;
use crate::rand::seq::index;
use crate::rand::{thread_rng, Rng, SeedableRng};

use ndarray::{Array, Axis, RemoveAxis, ShapeBuilder};
use ndarray::{ArrayBase, DataOwned, RawData, Data, Dimension};
#[cfg(feature = "quickcheck")]
use quickcheck::{Arbitrary, Gen};

/// `rand`, re-exported for convenience and version-compatibility.
pub mod rand {
    pub use rand::*;
}

/// `rand-distr`, re-exported for convenience and version-compatibility.
pub mod rand_distr {
    pub use rand_distr::*;
}

/// Constructors for n-dimensional arrays with random elements.
///
/// This trait extends ndarray’s `ArrayBase` and can not be implemented
/// for other types.
///
/// The default RNG is a fast automatically seeded rng (currently
/// [`rand::rngs::SmallRng`], seeded from [`rand::thread_rng`]).
///
/// Note that `SmallRng` is cheap to initialize and fast, but it may generate
/// low-quality random numbers, and reproducibility is not guaranteed. See its
/// documentation for information. You can select a different RNG with
/// [`.random_using()`](Self::random_using).
pub trait RandomExt<S, A, D>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    /// Create an array with shape `dim` with elements drawn from
    /// `distribution` using the default RNG.
    ///
    /// ***Panics*** if creation of the RNG fails or if the number of elements
    /// overflows usize.
    ///
    /// ```
    /// use ndarray::Array;
    /// use ndarray_rand::RandomExt;
    /// use ndarray_rand::rand_distr::Uniform;
    ///
    /// # fn main() {
    /// let a = Array::random((2, 5), Uniform::new(0., 10.));
    /// println!("{:8.4}", a);
    /// // Example Output:
    /// // [[  8.6900,   6.9824,   3.8922,   6.5861,   2.4890],
    /// //  [  0.0914,   5.5186,   5.8135,   5.2361,   3.1879]]
    /// # }
    fn random<Sh, IdS>(shape: Sh, distribution: IdS) -> ArrayBase<S, D>
    where
        IdS: Distribution<S::Elem>,
        S: DataOwned<Elem = A>,
        Sh: ShapeBuilder<Dim = D>;

    /// Create an array with shape `dim` with elements drawn from
    /// `distribution`, using a specific Rng `rng`.
    ///
    /// ***Panics*** if the number of elements overflows usize.
    ///
    /// ```
    /// use ndarray::Array;
    /// use ndarray_rand::RandomExt;
    /// use ndarray_rand::rand::SeedableRng;
    /// use ndarray_rand::rand_distr::Uniform;
    /// use rand_isaac::isaac64::Isaac64Rng;
    ///
    /// # fn main() {
    /// // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
    /// let seed = 42;
    /// let mut rng = Isaac64Rng::seed_from_u64(seed);
    ///
    /// // Generate a random array using `rng`
    /// let a = Array::random_using((2, 5), Uniform::new(0., 10.), &mut rng);
    /// println!("{:8.4}", a);
    /// // Example Output:
    /// // [[  8.6900,   6.9824,   3.8922,   6.5861,   2.4890],
    /// //  [  0.0914,   5.5186,   5.8135,   5.2361,   3.1879]]
    /// # }
    fn random_using<Sh, IdS, R>(shape: Sh, distribution: IdS, rng: &mut R) -> ArrayBase<S, D>
    where
        IdS: Distribution<S::Elem>,
        R: Rng + ?Sized,
        S: DataOwned<Elem = A>,
        Sh: ShapeBuilder<Dim = D>;

    /// Sample `n_samples` lanes slicing along `axis` using the default RNG.
    ///
    /// If `strategy==SamplingStrategy::WithoutReplacement`, each lane can only be sampled once.
    /// If `strategy==SamplingStrategy::WithReplacement`, each lane can be sampled multiple times.
    ///
    /// ***Panics*** when:
    /// - creation of the RNG fails;
    /// - `n_samples` is greater than the length of `axis` (if sampling without replacement);
    /// - length of `axis` is 0.
    ///
    /// ```
    /// use ndarray::{array, Axis};
    /// use ndarray_rand::{RandomExt, SamplingStrategy};
    ///
    /// # fn main() {
    /// let a = array![
    ///     [1., 2., 3.],
    ///     [4., 5., 6.],
    ///     [7., 8., 9.],
    ///     [10., 11., 12.],
    /// ];
    /// // Sample 2 rows, without replacement
    /// let sample_rows = a.sample_axis(Axis(0), 2, SamplingStrategy::WithoutReplacement);
    /// println!("{:?}", sample_rows);
    /// // Example Output: (1st and 3rd rows)
    /// // [
    /// //  [1., 2., 3.],
    /// //  [7., 8., 9.]
    /// // ]
    /// // Sample 2 columns, with replacement
    /// let sample_columns = a.sample_axis(Axis(1), 1, SamplingStrategy::WithReplacement);
    /// println!("{:?}", sample_columns);
    /// // Example Output: (2nd column, sampled twice)
    /// // [
    /// //  [2., 2.],
    /// //  [5., 5.],
    /// //  [8., 8.],
    /// //  [11., 11.]
    /// // ]
    /// # }
    /// ```
    fn sample_axis(&self, axis: Axis, n_samples: usize, strategy: SamplingStrategy) -> Array<A, D>
    where
        A: Copy,
        S: Data<Elem = A>,
        D: RemoveAxis;

    /// Sample `n_samples` lanes slicing along `axis` using the specified RNG `rng`.
    ///
    /// If `strategy==SamplingStrategy::WithoutReplacement`, each lane can only be sampled once.
    /// If `strategy==SamplingStrategy::WithReplacement`, each lane can be sampled multiple times.
    ///
    /// ***Panics*** when:
    /// - creation of the RNG fails;
    /// - `n_samples` is greater than the length of `axis` (if sampling without replacement);
    /// - length of `axis` is 0.
    ///
    /// ```
    /// use ndarray::{array, Axis};
    /// use ndarray_rand::{RandomExt, SamplingStrategy};
    /// use ndarray_rand::rand::SeedableRng;
    /// use rand_isaac::isaac64::Isaac64Rng;
    ///
    /// # fn main() {
    /// // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
    /// let seed = 42;
    /// let mut rng = Isaac64Rng::seed_from_u64(seed);
    ///
    /// let a = array![
    ///     [1., 2., 3.],
    ///     [4., 5., 6.],
    ///     [7., 8., 9.],
    ///     [10., 11., 12.],
    /// ];
    /// // Sample 2 rows, without replacement
    /// let sample_rows = a.sample_axis_using(Axis(0), 2, SamplingStrategy::WithoutReplacement, &mut rng);
    /// println!("{:?}", sample_rows);
    /// // Example Output: (1st and 3rd rows)
    /// // [
    /// //  [1., 2., 3.],
    /// //  [7., 8., 9.]
    /// // ]
    ///
    /// // Sample 2 columns, with replacement
    /// let sample_columns = a.sample_axis_using(Axis(1), 1, SamplingStrategy::WithReplacement, &mut rng);
    /// println!("{:?}", sample_columns);
    /// // Example Output: (2nd column, sampled twice)
    /// // [
    /// //  [2., 2.],
    /// //  [5., 5.],
    /// //  [8., 8.],
    /// //  [11., 11.]
    /// // ]
    /// # }
    /// ```
    fn sample_axis_using<R>(
        &self,
        axis: Axis,
        n_samples: usize,
        strategy: SamplingStrategy,
        rng: &mut R,
    ) -> Array<A, D>
    where
        R: Rng + ?Sized,
        A: Copy,
        S: Data<Elem = A>,
        D: RemoveAxis;
}

impl<S, A, D> RandomExt<S, A, D> for ArrayBase<S, D>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    fn random<Sh, IdS>(shape: Sh, dist: IdS) -> ArrayBase<S, D>
    where
        IdS: Distribution<S::Elem>,
        S: DataOwned<Elem = A>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::random_using(shape, dist, &mut get_rng())
    }

    fn random_using<Sh, IdS, R>(shape: Sh, dist: IdS, rng: &mut R) -> ArrayBase<S, D>
    where
        IdS: Distribution<S::Elem>,
        R: Rng + ?Sized,
        S: DataOwned<Elem = A>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_shape_simple_fn(shape, move || dist.sample(rng))
    }

    fn sample_axis(&self, axis: Axis, n_samples: usize, strategy: SamplingStrategy) -> Array<A, D>
    where
        A: Copy,
        S: Data<Elem = A>,
        D: RemoveAxis,
    {
        self.sample_axis_using(axis, n_samples, strategy, &mut get_rng())
    }

    fn sample_axis_using<R>(
        &self,
        axis: Axis,
        n_samples: usize,
        strategy: SamplingStrategy,
        rng: &mut R,
    ) -> Array<A, D>
    where
        R: Rng + ?Sized,
        A: Copy,
        S: Data<Elem = A>,
        D: RemoveAxis,
    {
        let indices: Vec<_> = match strategy {
            SamplingStrategy::WithReplacement => {
                let distribution = Uniform::from(0..self.len_of(axis));
                (0..n_samples).map(|_| distribution.sample(rng)).collect()
            }
            SamplingStrategy::WithoutReplacement => {
                index::sample(rng, self.len_of(axis), n_samples).into_vec()
            }
        };
        self.select(axis, &indices)
    }
}

/// Used as parameter in [`sample_axis`] and [`sample_axis_using`] to determine
/// if lanes from the original array should only be sampled once (*without replacement*) or
/// multiple times (*with replacement*).
///
/// [`sample_axis`]: RandomExt::sample_axis
/// [`sample_axis_using`]: RandomExt::sample_axis_using
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    WithReplacement,
    WithoutReplacement,
}

// `Arbitrary` enables `quickcheck` to generate random `SamplingStrategy` values for testing.
#[cfg(feature = "quickcheck")]
impl Arbitrary for SamplingStrategy {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        if bool::arbitrary(g) {
            SamplingStrategy::WithReplacement
        } else {
            SamplingStrategy::WithoutReplacement
        }
    }
}

fn get_rng() -> SmallRng {
    SmallRng::from_rng(thread_rng()).expect("create SmallRng from thread_rng failed")
}

/// A wrapper type that allows casting f64 distributions to f32
///
/// ```
/// use ndarray::Array;
/// use ndarray_rand::{RandomExt, F32};
/// use ndarray_rand::rand_distr::Normal;
///
/// # fn main() {
/// let distribution_f64 = Normal::new(0., 1.).expect("Failed to create normal distribution");
/// let a = Array::random((2, 5), F32(distribution_f64));
/// println!("{:8.4}", a);
/// // Example Output:
/// // [[ -0.6910,   1.1730,   1.0902,  -0.4092,  -1.7340],
/// //  [ -0.6810,   0.1678,  -0.9487,   0.3150,   1.2981]]
/// # }
#[derive(Copy, Clone, Debug)]
#[deprecated(since="0.14.0", note="Redundant with rand 0.8")]
pub struct F32<S>(pub S);

#[allow(deprecated)]
impl<S> Distribution<f32> for F32<S>
where
    S: Distribution<f64>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        self.0.sample(rng) as f32
    }
}
