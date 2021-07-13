//! Element-wise methods for ndarray

use num_traits::Float;

use crate::imp_prelude::*;

macro_rules! boolean_op {
    ($(#[$meta1:meta])* fn $id1:ident $(#[$meta2:meta])* fn $id2:ident -> $func:ident) => {
        $(#[$meta1])*
        pub fn $id1(&self) -> Array<bool, D> {
            self.mapv(A::$func)
        }
        $(#[$meta2])*
        pub fn $id2(&self) -> bool {
            self.mapv(A::$func).iter().any(|&b|b)
        }
    };
}

macro_rules! map_op {
    ($(#[$meta:meta])* fn $id:ident) => {
        $(#[$meta])*
        pub fn $id(&self) -> Array<A, D> {
            self.mapv(A::$id)
        }
    };
}

macro_rules! bin_op {
    ($(#[$meta:meta])* fn $id:ident($ty:ty)) => {
        $(#[$meta])*
        pub fn $id(&self, rhs: $ty) -> Array<A, D> {
            self.mapv(|v| A::$id(v, rhs))
        }
    };
}

/// # Element-wise methods for Float Array
///
/// Element-wise math functions for any array type that contains float number.
impl<A, S, D> ArrayBase<S, D>
where
    A: Float,
    S: RawData<Elem = A> + Data,
    D: Dimension,
{
    boolean_op! {
        /// If the number is `NaN` (not a number), then `true` is returned for each element.
        fn is_nan
        /// Return `true` if any element is `NaN` (not a number).
        fn is_nan_any -> is_nan
    }
    boolean_op! {
        /// If the number is infinity, then `true` is returned for each element.
        fn is_infinite
        /// Return `true` if any element is infinity.
        fn is_infinite_any -> is_infinite
    }
    map_op! {
        /// The largest integer less than or equal to each element.
        fn floor
    }
    map_op! {
        /// The smallest integer less than or equal to each element.
        fn ceil
    }
    map_op! {
        /// The nearest integer of each element.
        fn round
    }
    map_op! {
        /// The integer part of each element.
        fn trunc
    }
    map_op! {
        /// The fractional part of each element.
        fn fract
    }
    map_op! {
        /// Absolute of each element.
        fn abs
    }
    map_op! {
        /// Sign number of each element.
        ///
        /// + `1.0` for all positive numbers.
        /// + `-1.0` for all negative numbers.
        /// + `NaN` for all `NaN` (not a number).
        fn signum
    }
    map_op! {
        /// The reciprocal (inverse) of each element, `1/x`.
        fn recip
    }
    bin_op! {
        /// Integer power of each element.
        ///
        /// This function is generally faster than using float power.
        fn powi(i32)
    }
    bin_op! {
        /// Float power of each element.
        fn powf(A)
    }

    /// Square of each element.
    pub fn square(&self) -> Array<A, D> {
        self.mapv(|v| v * v)
    }

    map_op! {
        /// Square root of each element.
        fn sqrt
    }
    map_op! {
        /// `e^x` of each element. (Exponential function)
        fn exp
    }
    map_op! {
        /// `2^x` of each element.
        fn exp2
    }
    map_op! {
        /// Natural logarithm of each element.
        fn ln
    }
    bin_op! {
        /// Logarithm of each element with respect to an arbitrary base.
        fn log(A)
    }
    map_op! {
        /// Base 2 logarithm of each element.
        fn log2
    }
    map_op! {
        /// Base 10 logarithm of each element.
        fn log10
    }
    bin_op! {
        /// The positive difference between given number and each element.
        fn abs_sub(A)
    }
    map_op! {
        /// Cubic root of each element.
        fn cbrt
    }
    map_op! {
        /// Sine of each element. (in radians)
        fn sin
    }
    map_op! {
        /// Cosine of each element. (in radians)
        fn cos
    }
    map_op! {
        /// Tangent of each element. (in radians)
        fn tan
    }
    map_op! {
        /// Converts radians to degrees for each element.
        fn to_degrees
    }
    map_op! {
        /// Converts degrees to radians for each element.
        fn to_radians
    }

    /// Limit the values for each element.
    ///
    /// ```
    /// use ndarray::{Array1, array};
    ///
    /// let a = Array1::range(0., 10., 1.);
    /// assert_eq!(a.clip(1., 8.), array![1., 1., 2., 3., 4., 5., 6., 7., 8., 8.]);
    /// assert_eq!(a.clip(8., 1.), array![1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]);
    /// assert_eq!(a.clip(3., 6.), array![3., 3., 3., 3., 4., 5., 6., 6., 6., 6.]);
    /// ```
    pub fn clip(&self, min: A, max: A) -> Array<A, D> {
        self.mapv(|v| A::max(v, min)).mapv(|v| A::min(v, max))
    }
}
