#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]
#![cfg(feature = "std")]
use ndarray::linalg::general_mat_mul;
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray::{rcarr1, rcarr2};
use ndarray::{Data, LinalgScalar};
use ndarray::{Ix, Ixs};
use num_traits::Zero;

use approx::assert_abs_diff_eq;
use defmac::defmac;

fn test_oper(op: &str, a: &[f32], b: &[f32], c: &[f32]) {
    let aa = rcarr1(a);
    let bb = rcarr1(b);
    let cc = rcarr1(c);
    test_oper_arr::<f32, _>(op, aa.clone(), bb.clone(), cc.clone());
    let dim = (2, 2);
    let aa = aa.reshape(dim);
    let bb = bb.reshape(dim);
    let cc = cc.reshape(dim);
    test_oper_arr::<f32, _>(op, aa.clone(), bb.clone(), cc.clone());
    let dim = (1, 2, 1, 2);
    let aa = aa.reshape(dim);
    let bb = bb.reshape(dim);
    let cc = cc.reshape(dim);
    test_oper_arr::<f32, _>(op, aa.clone(), bb.clone(), cc.clone());
}


fn test_oper_arr<A, D>(op: &str, mut aa: ArcArray<f32, D>, bb: ArcArray<f32, D>, cc: ArcArray<f32, D>)
where
    D: Dimension,
{
    match op {
        "+" => {
            assert_eq!(&aa + &bb, cc);
            aa += &bb;
            assert_eq!(aa, cc);
        }
        "-" => {
            assert_eq!(&aa - &bb, cc);
            aa -= &bb;
            assert_eq!(aa, cc);
        }
        "*" => {
            assert_eq!(&aa * &bb, cc);
            aa *= &bb;
            assert_eq!(aa, cc);
        }
        "/" => {
            assert_eq!(&aa / &bb, cc);
            aa /= &bb;
            assert_eq!(aa, cc);
        }
        "%" => {
            assert_eq!(&aa % &bb, cc);
            aa %= &bb;
            assert_eq!(aa, cc);
        }
        "neg" => {
            assert_eq!(-&aa, cc);
            assert_eq!(-aa.clone(), cc);
        }
        _ => panic!(),
    }
}

#[test]
fn operations() {
    test_oper(
        "+",
        &[1.0, 2.0, 3.0, 4.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[1.0, 3.0, 5.0, 7.0],
    );
    test_oper(
        "-",
        &[1.0, 2.0, 3.0, 4.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[1.0, 1.0, 1.0, 1.0],
    );
    test_oper(
        "*",
        &[1.0, 2.0, 3.0, 4.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 2.0, 6.0, 12.0],
    );
    test_oper(
        "/",
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 1.0, 2.0, 3.0],
        &[1.0, 2.0, 3.0 / 2.0, 4.0 / 3.0],
    );
    test_oper(
        "%",
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 1.0, 2.0, 3.0],
        &[0.0, 0.0, 1.0, 1.0],
    );
    test_oper(
        "neg",
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 1.0, 2.0, 3.0],
        &[-1.0, -2.0, -3.0, -4.0],
    );
}

#[test]
fn scalar_operations() {
    let a = arr0::<f32>(1.);
    let b = rcarr1::<f32>(&[1., 1.]);
    let c = rcarr2(&[[1., 1.], [1., 1.]]);

    {
        let mut x = a.clone();
        let mut y = arr0(0.);
        x += 1.;
        y.fill(2.);
        assert_eq!(x, a + arr0(1.));
        assert_eq!(x, y);
    }

    {
        let mut x = b.clone();
        let mut y = rcarr1(&[0., 0.]);
        x += 1.;
        y.fill(2.);
        assert_eq!(x, b + arr0(1.));
        assert_eq!(x, y);
    }

    {
        let mut x = c.clone();
        let mut y = ArcArray::zeros((2, 2));
        x += 1.;
        y.fill(2.);
        assert_eq!(x, c + arr0(1.));
        assert_eq!(x, y);
    }
}

fn reference_dot<'a, V1, V2>(a: V1, b: V2) -> f32
where
    V1: AsArray<'a, f32>,
    V2: AsArray<'a, f32>,
{
    let a = a.into();
    let b = b.into();
    a.iter()
        .zip(b.iter())
        .fold(f32::zero(), |acc, (&x, &y)| acc + x * y)
}

#[test]
fn dot_product() {
    let a = Array::range(0., 69., 1.);
    let b = &a * 2. - 7.;
    let dot = 197846.;
    assert_abs_diff_eq!(a.dot(&b), reference_dot(&a, &b), epsilon = 1e-5);

    // test different alignments
    let max = 8 as Ixs;
    for i in 1..max {
        let a1 = a.slice(s![i..]);
        let b1 = b.slice(s![i..]);
        assert_abs_diff_eq!(a1.dot(&b1), reference_dot(&a1, &b1), epsilon = 1e-5);
        let a2 = a.slice(s![..-i]);
        let b2 = b.slice(s![i..]);
        assert_abs_diff_eq!(a2.dot(&b2), reference_dot(&a2, &b2), epsilon = 1e-5);
    }

    let a = a.map(|f| *f as f32);
    let b = b.map(|f| *f as f32);
    assert_abs_diff_eq!(a.dot(&b), dot as f32, epsilon = 1e-5);

    let max = 8 as Ixs;
    for i in 1..max {
        let a1 = a.slice(s![i..]);
        let b1 = b.slice(s![i..]);
        assert_abs_diff_eq!(a1.dot(&b1), reference_dot(&a1, &b1), epsilon = 1e-5);
        let a2 = a.slice(s![..-i]);
        let b2 = b.slice(s![i..]);
        assert_abs_diff_eq!(a2.dot(&b2), reference_dot(&a2, &b2), epsilon = 1e-5);
    }

    let a = a.map(|f| *f as i32);
    let b = b.map(|f| *f as i32);
    assert_eq!(a.dot(&b), dot as i32);
}

// test that we can dot product with a broadcast array
#[test]
fn dot_product_0() {
    let a = Array::range(0., 69., 1.);
    let x = 1.5;
    let b = aview0(&x);
    let b = b.broadcast(a.dim()).unwrap();
    assert_abs_diff_eq!(a.dot(&b), reference_dot(&a, &b), epsilon = 1e-5);

    // test different alignments
    let max = 8 as Ixs;
    for i in 1..max {
        let a1 = a.slice(s![i..]);
        let b1 = b.slice(s![i..]);
        assert_abs_diff_eq!(a1.dot(&b1), reference_dot(&a1, &b1), epsilon = 1e-5);
        let a2 = a.slice(s![..-i]);
        let b2 = b.slice(s![i..]);
        assert_abs_diff_eq!(a2.dot(&b2), reference_dot(&a2, &b2), epsilon = 1e-5);
    }
}

#[test]
fn dot_product_neg_stride() {
    // test that we can dot with negative stride
    let a = Array::range(0., 69., 1.);
    let b = &a * 2. - 7.;
    for stride in -10..0 {
        // both negative
        let a = a.slice(s![..;stride]);
        let b = b.slice(s![..;stride]);
        assert_abs_diff_eq!(a.dot(&b), reference_dot(&a, &b), epsilon = 1e-5);
    }
    for stride in -10..0 {
        // mixed
        let a = a.slice(s![..;-stride]);
        let b = b.slice(s![..;stride]);
        assert_abs_diff_eq!(a.dot(&b), reference_dot(&a, &b), epsilon = 1e-5);
    }
}

#[test]
fn fold_and_sum() {
    let a = Array::linspace(0., 127., 128).into_shape((8, 16)).unwrap();
    assert_abs_diff_eq!(a.fold(0., |acc, &x| acc + x), a.sum(), epsilon = 1e-5);

    // test different strides
    let max = 8 as Ixs;
    for i in 1..max {
        for j in 1..max {
            let a1 = a.slice(s![..;i, ..;j]);
            let mut sum = 0.;
            for elt in a1.iter() {
                sum += *elt;
            }
            assert_abs_diff_eq!(a1.fold(0., |acc, &x| acc + x), sum, epsilon = 1e-5);
            assert_abs_diff_eq!(sum, a1.sum(), epsilon = 1e-5);
        }
    }

    // skip a few elements
    let max = 8 as Ixs;
    for i in 1..max {
        for skip in 1..max {
            let a1 = a.slice(s![.., ..;i]);
            let mut iter1 = a1.iter();
            for _ in 0..skip {
                iter1.next();
            }
            let iter2 = iter1.clone();

            let mut sum = 0.;
            for elt in iter1 {
                sum += *elt;
            }
            assert_abs_diff_eq!(iter2.fold(0., |acc, &x| acc + x), sum, epsilon = 1e-5);
        }
    }
}

#[test]
fn product() {
    let a = Array::linspace(0.5, 2., 128).into_shape((8, 16)).unwrap();
    assert_abs_diff_eq!(a.fold(1., |acc, &x| acc * x), a.product(), epsilon = 1e-5);

    // test different strides
    let max = 8 as Ixs;
    for i in 1..max {
        for j in 1..max {
            let a1 = a.slice(s![..;i, ..;j]);
            let mut prod = 1.;
            for elt in a1.iter() {
                prod *= *elt;
            }
            assert_abs_diff_eq!(a1.fold(1., |acc, &x| acc * x), prod, epsilon = 1e-5);
            assert_abs_diff_eq!(prod, a1.product(), epsilon = 1e-5);
        }
    }
}

fn range_mat(m: Ix, n: Ix) -> Array2<f32> {
    Array::linspace(0., (m * n) as f32 - 1., m * n)
        .into_shape((m, n))
        .unwrap()
}

fn range_mat64(m: Ix, n: Ix) -> Array2<f64> {
    Array::linspace(0., (m * n) as f64 - 1., m * n)
        .into_shape((m, n))
        .unwrap()
}

#[cfg(feature = "approx")]
fn range1_mat64(m: Ix) -> Array1<f64> {
    Array::linspace(0., m as f64 - 1., m)
}

fn range_i32(m: Ix, n: Ix) -> Array2<i32> {
    Array::from_iter(0..(m * n) as i32)
        .into_shape((m, n))
        .unwrap()
}

// simple, slow, correct (hopefully) mat mul
fn reference_mat_mul<A, S, S2>(lhs: &ArrayBase<S, Ix2>, rhs: &ArrayBase<S2, Ix2>) -> Array2<A>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let ((m, k), (k2, n)) = (lhs.dim(), rhs.dim());
    assert!(m.checked_mul(n).is_some());
    assert_eq!(k, k2);
    let mut res_elems = Vec::<A>::with_capacity(m * n);
    unsafe {
        res_elems.set_len(m * n);
    }

    let mut i = 0;
    let mut j = 0;
    for rr in &mut res_elems {
        unsafe {
            *rr = (0..k).fold(A::zero(), move |s, x| {
                s + *lhs.uget((i, x)) * *rhs.uget((x, j))
            });
        }
        j += 1;
        if j == n {
            j = 0;
            i += 1;
        }
    }
    unsafe { ArrayBase::from_shape_vec_unchecked((m, n), res_elems) }
}

#[test]
fn mat_mul() {
    let (m, n, k) = (8, 8, 8);
    let a = range_mat(m, n);
    let b = range_mat(n, k);
    let mut b = b / 4.;
    {
        let mut c = b.column_mut(0);
        c += 1.0;
    }
    let ab = a.dot(&b);

    let mut af = Array::zeros(a.dim().f());
    let mut bf = Array::zeros(b.dim().f());
    af.assign(&a);
    bf.assign(&b);

    assert_eq!(ab, a.dot(&bf));
    assert_eq!(ab, af.dot(&b));
    assert_eq!(ab, af.dot(&bf));

    let (m, n, k) = (10, 5, 11);
    let a = range_mat(m, n);
    let b = range_mat(n, k);
    let mut b = b / 4.;
    {
        let mut c = b.column_mut(0);
        c += 1.0;
    }
    let ab = a.dot(&b);

    let mut af = Array::zeros(a.dim().f());
    let mut bf = Array::zeros(b.dim().f());
    af.assign(&a);
    bf.assign(&b);

    assert_eq!(ab, a.dot(&bf));
    assert_eq!(ab, af.dot(&b));
    assert_eq!(ab, af.dot(&bf));

    let (m, n, k) = (10, 8, 1);
    let a = range_mat(m, n);
    let b = range_mat(n, k);
    let mut b = b / 4.;
    {
        let mut c = b.column_mut(0);
        c += 1.0;
    }
    let ab = a.dot(&b);

    let mut af = Array::zeros(a.dim().f());
    let mut bf = Array::zeros(b.dim().f());
    af.assign(&a);
    bf.assign(&b);

    assert_eq!(ab, a.dot(&bf));
    assert_eq!(ab, af.dot(&b));
    assert_eq!(ab, af.dot(&bf));
}

// Check that matrix multiplication of contiguous matrices returns a
// matrix with the same order
#[test]
fn mat_mul_order() {
    let (m, n, k) = (8, 8, 8);
    let a = range_mat(m, n);
    let b = range_mat(n, k);
    let mut af = Array::zeros(a.dim().f());
    let mut bf = Array::zeros(b.dim().f());
    af.assign(&a);
    bf.assign(&b);

    let cc = a.dot(&b);
    let ff = af.dot(&bf);

    assert_eq!(cc.strides()[1], 1);
    assert_eq!(ff.strides()[0], 1);
}

// test matrix multiplication shape mismatch
#[test]
#[should_panic]
fn mat_mul_shape_mismatch() {
    let (m, k, k2, n) = (8, 8, 9, 8);
    let a = range_mat(m, k);
    let b = range_mat(k2, n);
    a.dot(&b);
}

// test matrix multiplication shape mismatch
#[test]
#[should_panic]
fn mat_mul_shape_mismatch_2() {
    let (m, k, k2, n) = (8, 8, 8, 8);
    let a = range_mat(m, k);
    let b = range_mat(k2, n);
    let mut c = range_mat(m, n + 1);
    general_mat_mul(1., &a, &b, 1., &mut c);
}

// Check that matrix multiplication
// supports broadcast arrays.
#[test]
fn mat_mul_broadcast() {
    let (m, n, k) = (16, 16, 16);
    let a = range_mat(m, n);
    let x1 = 1.;
    let x = Array::from(vec![x1]);
    let b0 = x.broadcast((n, k)).unwrap();
    let b1 = Array::from_elem(n, x1);
    let b1 = b1.broadcast((n, k)).unwrap();
    let b2 = Array::from_elem((n, k), x1);

    let c2 = a.dot(&b2);
    let c1 = a.dot(&b1);
    let c0 = a.dot(&b0);
    assert_eq!(c2, c1);
    assert_eq!(c2, c0);
}

// Check that matrix multiplication supports reversed axes
#[test]
fn mat_mul_rev() {
    let (m, n, k) = (16, 16, 16);
    let a = range_mat(m, n);
    let b = range_mat(n, k);
    let mut rev = Array::zeros(b.dim());
    let mut rev = rev.slice_mut(s![..;-1, ..]);
    rev.assign(&b);
    println!("{:.?}", rev);

    let c1 = a.dot(&b);
    let c2 = a.dot(&rev);
    assert_eq!(c1, c2);
}

// Check that matrix multiplication supports arrays with zero rows or columns
#[test]
fn mat_mut_zero_len() {
    defmac!(mat_mul_zero_len range_mat_fn => {
        for n in 0..4 {
            for m in 0..4 {
                let a = range_mat_fn(m, n);
                let b = range_mat_fn(n, 0);
                assert_eq!(a.dot(&b), Array2::zeros((m, 0)));
            }
            for k in 0..4 {
                let a = range_mat_fn(0, n);
                let b = range_mat_fn(n, k);
                assert_eq!(a.dot(&b), Array2::zeros((0, k)));
            }
        }
    });
    mat_mul_zero_len!(range_mat);
    mat_mul_zero_len!(range_mat64);
    mat_mul_zero_len!(range_i32);
}

#[test]
fn scaled_add() {
    let a = range_mat(16, 15);
    let mut b = range_mat(16, 15);
    b.mapv_inplace(f32::exp);

    let alpha = 0.2_f32;
    let mut c = a.clone();
    c.scaled_add(alpha, &b);

    let d = alpha * &b + &a;
    assert_eq!(c, d);
}

#[cfg(feature = "approx")]
#[test]
fn scaled_add_2() {
    let beta = -2.3;
    let sizes = vec![
        (4, 4, 1, 4),
        (8, 8, 1, 8),
        (17, 15, 17, 15),
        (4, 17, 4, 17),
        (17, 3, 1, 3),
        (19, 18, 19, 18),
        (16, 17, 16, 17),
        (15, 16, 15, 16),
        (67, 63, 1, 63),
    ];
    // test different strides
    for &s1 in &[1, 2, -1, -2] {
        for &s2 in &[1, 2, -1, -2] {
            for &(m, k, n, q) in &sizes {
                let mut a = range_mat64(m, k);
                let mut answer = a.clone();
                let c = range_mat64(n, q);

                {
                    let mut av = a.slice_mut(s![..;s1, ..;s2]);
                    let c = c.slice(s![..;s1, ..;s2]);

                    let mut answerv = answer.slice_mut(s![..;s1, ..;s2]);
                    answerv += &(beta * &c);
                    av.scaled_add(beta, &c);
                }
                approx::assert_relative_eq!(a, answer, epsilon = 1e-12, max_relative = 1e-7);
            }
        }
    }
}

#[cfg(feature = "approx")]
#[test]
fn scaled_add_3() {
    use approx::assert_relative_eq;
    use ndarray::{Slice, SliceInfo, SliceInfoElem};
    use std::convert::TryFrom;

    let beta = -2.3;
    let sizes = vec![
        (4, 4, 1, 4),
        (8, 8, 1, 8),
        (17, 15, 17, 15),
        (4, 17, 4, 17),
        (17, 3, 1, 3),
        (19, 18, 19, 18),
        (16, 17, 16, 17),
        (15, 16, 15, 16),
        (67, 63, 1, 63),
    ];
    // test different strides
    for &s1 in &[1, 2, -1, -2] {
        for &s2 in &[1, 2, -1, -2] {
            for &(m, k, n, q) in &sizes {
                let mut a = range_mat64(m, k);
                let mut answer = a.clone();
                let cdim = if n == 1 { vec![q] } else { vec![n, q] };
                let cslice: Vec<SliceInfoElem> = if n == 1 {
                    vec![Slice::from(..).step_by(s2).into()]
                } else {
                    vec![
                        Slice::from(..).step_by(s1).into(),
                        Slice::from(..).step_by(s2).into(),
                    ]
                };

                let c = range_mat64(n, q).into_shape(cdim).unwrap();

                {
                    let mut av = a.slice_mut(s![..;s1, ..;s2]);
                    let c = c.slice(SliceInfo::<_, IxDyn, IxDyn>::try_from(cslice).unwrap());

                    let mut answerv = answer.slice_mut(s![..;s1, ..;s2]);
                    answerv += &(beta * &c);
                    av.scaled_add(beta, &c);
                }
                assert_relative_eq!(a, answer, epsilon = 1e-12, max_relative = 1e-7);
            }
        }
    }
}

#[cfg(feature = "approx")]
#[test]
fn gen_mat_mul() {
    let alpha = -2.3;
    let beta = 3.14;
    let sizes = vec![
        (4, 4, 4),
        (8, 8, 8),
        (17, 15, 16),
        (4, 17, 3),
        (17, 3, 22),
        (19, 18, 2),
        (16, 17, 15),
        (15, 16, 17),
        (67, 63, 62),
    ];
    // test different strides
    for &s1 in &[1, 2, -1, -2] {
        for &s2 in &[1, 2, -1, -2] {
            for &(m, k, n) in &sizes {
                let a = range_mat64(m, k);
                let b = range_mat64(k, n);
                let mut c = range_mat64(m, n);
                let mut answer = c.clone();

                {
                    let a = a.slice(s![..;s1, ..;s2]);
                    let b = b.slice(s![..;s2, ..;s2]);
                    let mut cv = c.slice_mut(s![..;s1, ..;s2]);

                    let answer_part = alpha * reference_mat_mul(&a, &b) + beta * &cv;
                    answer.slice_mut(s![..;s1, ..;s2]).assign(&answer_part);

                    general_mat_mul(alpha, &a, &b, beta, &mut cv);
                }
                approx::assert_relative_eq!(c, answer, epsilon = 1e-12, max_relative = 1e-7);
            }
        }
    }
}

// Test y = A x where A is f-order
#[cfg(feature = "approx")]
#[test]
fn gemm_64_1_f() {
    let a = range_mat64(64, 64).reversed_axes();
    let (m, n) = a.dim();
    // m x n  times n x 1  == m x 1
    let x = range_mat64(n, 1);
    let mut y = range_mat64(m, 1);
    let answer = reference_mat_mul(&a, &x) + &y;
    general_mat_mul(1.0, &a, &x, 1.0, &mut y);
    approx::assert_relative_eq!(y, answer, epsilon = 1e-12, max_relative = 1e-7);
}

#[test]
fn gen_mat_mul_i32() {
    let alpha = -1;
    let beta = 2;
    let sizes = if cfg!(miri) {
        vec![(4, 4, 4), (4, 7, 3)]
    } else {
        vec![
            (4, 4, 4),
            (8, 8, 8),
            (17, 15, 16),
            (4, 17, 3),
            (17, 3, 22),
            (19, 18, 2),
            (16, 17, 15),
            (15, 16, 17),
            (67, 63, 62),
        ]
    };
    for &(m, k, n) in &sizes {
        let a = range_i32(m, k);
        let b = range_i32(k, n);
        let mut c = range_i32(m, n);

        let answer = alpha * reference_mat_mul(&a, &b) + beta * &c;
        general_mat_mul(alpha, &a, &b, beta, &mut c);
        assert_eq!(&c, &answer);
    }
}

#[cfg(feature = "approx")]
#[test]
fn gen_mat_vec_mul() {
    use approx::assert_relative_eq;
    use ndarray::linalg::general_mat_vec_mul;

    // simple, slow, correct (hopefully) mat mul
    fn reference_mat_vec_mul<A, S, S2>(
        lhs: &ArrayBase<S, Ix2>,
        rhs: &ArrayBase<S2, Ix1>,
    ) -> Array1<A>
    where
        A: LinalgScalar,
        S: Data<Elem = A>,
        S2: Data<Elem = A>,
    {
        let ((m, _), k) = (lhs.dim(), rhs.dim());
        reference_mat_mul(lhs, &rhs.as_standard_layout().into_shape((k, 1)).unwrap())
            .into_shape(m)
            .unwrap()
    }

    let alpha = -2.3;
    let beta = 3.14;
    let sizes = vec![
        (4, 4),
        (8, 8),
        (17, 15),
        (4, 17),
        (17, 3),
        (19, 18),
        (16, 17),
        (15, 16),
        (67, 63),
    ];
    // test different strides
    for &s1 in &[1, 2, -1, -2] {
        for &s2 in &[1, 2, -1, -2] {
            for &(m, k) in &sizes {
                for &rev in &[false, true] {
                    let mut a = range_mat64(m, k);
                    if rev {
                        a = a.reversed_axes();
                    }
                    let (m, k) = a.dim();
                    let b = range1_mat64(k);
                    let mut c = range1_mat64(m);
                    let mut answer = c.clone();

                    {
                        let a = a.slice(s![..;s1, ..;s2]);
                        let b = b.slice(s![..;s2]);
                        let mut cv = c.slice_mut(s![..;s1]);

                        let answer_part = alpha * reference_mat_vec_mul(&a, &b) + beta * &cv;
                        answer.slice_mut(s![..;s1]).assign(&answer_part);

                        general_mat_vec_mul(alpha, &a, &b, beta, &mut cv);
                    }
                    assert_relative_eq!(c, answer, epsilon = 1e-12, max_relative = 1e-7);
                }
            }
        }
    }
}

#[cfg(feature = "approx")]
#[test]
fn vec_mat_mul() {
    use approx::assert_relative_eq;

    // simple, slow, correct (hopefully) mat mul
    fn reference_vec_mat_mul<A, S, S2>(
        lhs: &ArrayBase<S, Ix1>,
        rhs: &ArrayBase<S2, Ix2>,
    ) -> Array1<A>
    where
        A: LinalgScalar,
        S: Data<Elem = A>,
        S2: Data<Elem = A>,
    {
        let (m, (_, n)) = (lhs.dim(), rhs.dim());
        reference_mat_mul(&lhs.as_standard_layout().into_shape((1, m)).unwrap(), rhs)
            .into_shape(n)
            .unwrap()
    }

    let sizes = vec![
        (4, 4),
        (8, 8),
        (17, 15),
        (4, 17),
        (17, 3),
        (19, 18),
        (16, 17),
        (15, 16),
        (67, 63),
    ];
    // test different strides
    for &s1 in &[1, 2, -1, -2] {
        for &s2 in &[1, 2, -1, -2] {
            for &(m, n) in &sizes {
                for &rev in &[false, true] {
                    let mut b = range_mat64(m, n);
                    if rev {
                        b = b.reversed_axes();
                    }
                    let (m, n) = b.dim();
                    let a = range1_mat64(m);
                    let mut c = range1_mat64(n);
                    let mut answer = c.clone();

                    {
                        let b = b.slice(s![..;s1, ..;s2]);
                        let a = a.slice(s![..;s1]);

                        let answer_part = reference_vec_mat_mul(&a, &b);
                        answer.slice_mut(s![..;s2]).assign(&answer_part);

                        c.slice_mut(s![..;s2]).assign(&a.dot(&b));
                    }
                    assert_relative_eq!(c, answer, epsilon = 1e-12, max_relative = 1e-7);
                }
            }
        }
    }
}

#[test]
fn kron_square_f64() {
    let a = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let b = arr2(&[[0.0, 1.0], [1.0, 0.0]]);

    assert_eq!(
        kron(&a, &b),
        arr2(&[
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ]),
    );

    assert_eq!(
        kron(&b, &a),
        arr2(&[
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]),
    )
}

#[test]
fn kron_square_i64() {
    let a = arr2(&[[1, 0], [0, 1]]);
    let b = arr2(&[[0, 1], [1, 0]]);

    assert_eq!(
        kron(&a, &b),
        arr2(&[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
    );

    assert_eq!(
        kron(&b, &a),
        arr2(&[[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]),
    )
}

#[test]
fn kron_i64() {
    let a = arr2(&[[1, 0]]);
    let b = arr2(&[[0, 1], [1, 0]]);
    let r = arr2(&[[0, 1, 0, 0], [1, 0, 0, 0]]);
    assert_eq!(kron(&a, &b), r);

    let a = arr2(&[[1, 0], [0, 0], [0, 1]]);
    let b = arr2(&[[0, 1], [1, 0]]);
    let r = arr2(&[
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]);
    assert_eq!(kron(&a, &b), r);
}
