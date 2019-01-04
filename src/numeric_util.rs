// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use std::cmp;
use std::ops::Add;
use num_traits::{self, Zero};
use crate::LinalgScalar;

pub(crate) fn pairwise_sum<A>(v: &[A]) -> A
where
    A: Clone + Add<Output=A> + Zero,
{
    let n = v.len();
    if n <= 512 {
        return unrolled_fold(v, A::zero, A::add);
    } else {
        let mid_index = n / 2;
        let (v1, v2) = v.split_at(mid_index);
        pairwise_sum(v1) + pairwise_sum(v2)
    }
}

/// Fold over the manually unrolled `xs` with `f`
pub fn unrolled_fold<A, I, F>(mut xs: &[A], init: I, f: F) -> A
    where A: Clone,
    I: Fn() -> A,
    F: Fn(A, A) -> A,
{
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let mut acc = init();
    let (mut p0, mut p1, mut p2, mut p3,
         mut p4, mut p5, mut p6, mut p7) =
        (init(), init(), init(), init(),
         init(), init(), init(), init());
    while xs.len() >= 8 {
        p0 = f(p0, xs[0].clone());
        p1 = f(p1, xs[1].clone());
        p2 = f(p2, xs[2].clone());
        p3 = f(p3, xs[3].clone());
        p4 = f(p4, xs[4].clone());
        p5 = f(p5, xs[5].clone());
        p6 = f(p6, xs[6].clone());
        p7 = f(p7, xs[7].clone());

        xs = &xs[8..];
    }
    acc = f(acc.clone(), f(p0, p4));
    acc = f(acc.clone(), f(p1, p5));
    acc = f(acc.clone(), f(p2, p6));
    acc = f(acc.clone(), f(p3, p7));

    // make it clear to the optimizer that this loop is short
    // and can not be autovectorized.
    for i in 0..xs.len() {
        if i >= 7 { break; }
        acc = f(acc.clone(), xs[i].clone())
    }
    acc
}

/// Compute the dot product.
///
/// `xs` and `ys` must be the same length
pub fn unrolled_dot<A>(xs: &[A], ys: &[A]) -> A
    where A: LinalgScalar,
{
    debug_assert_eq!(xs.len(), ys.len());
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let len = cmp::min(xs.len(), ys.len());
    let mut xs = &xs[..len];
    let mut ys = &ys[..len];
    let mut sum = A::zero();
    let (mut p0, mut p1, mut p2, mut p3,
         mut p4, mut p5, mut p6, mut p7) =
        (A::zero(), A::zero(), A::zero(), A::zero(),
         A::zero(), A::zero(), A::zero(), A::zero());
    while xs.len() >= 8 {
        p0 = p0 + xs[0] * ys[0];
        p1 = p1 + xs[1] * ys[1];
        p2 = p2 + xs[2] * ys[2];
        p3 = p3 + xs[3] * ys[3];
        p4 = p4 + xs[4] * ys[4];
        p5 = p5 + xs[5] * ys[5];
        p6 = p6 + xs[6] * ys[6];
        p7 = p7 + xs[7] * ys[7];

        xs = &xs[8..];
        ys = &ys[8..];
    }
    sum = sum + (p0 + p4);
    sum = sum + (p1 + p5);
    sum = sum + (p2 + p6);
    sum = sum + (p3 + p7);

    for i in 0..xs.len() {
        if i >= 7 { break; }
        unsafe {
            // get_unchecked is needed to avoid the bounds check
            sum = sum + xs[i] * *ys.get_unchecked(i);
        }
    }
    sum
}

/// Compute pairwise equality
///
/// `xs` and `ys` must be the same length
pub fn unrolled_eq<A>(xs: &[A], ys: &[A]) -> bool
    where A: PartialEq
{
    debug_assert_eq!(xs.len(), ys.len());
    // eightfold unrolled for performance (this is not done by llvm automatically)
    let len = cmp::min(xs.len(), ys.len());
    let mut xs = &xs[..len];
    let mut ys = &ys[..len];

    while xs.len() >= 8 {
        if (xs[0] != ys[0])
        | (xs[1] != ys[1])
        | (xs[2] != ys[2])
        | (xs[3] != ys[3])
        | (xs[4] != ys[4])
        | (xs[5] != ys[5])
        | (xs[6] != ys[6])
        | (xs[7] != ys[7]) { return false; }
        xs = &xs[8..];
        ys = &ys[8..];
    }

    for i in 0..xs.len() {
        if xs[i] != ys[i] {
            return false;
        }
    }

    true
}
