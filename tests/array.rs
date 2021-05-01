#![allow(non_snake_case)]
#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::float_cmp
)]

use defmac::defmac;
use itertools::{enumerate, zip, Itertools};
use ndarray::indices;
use ndarray::prelude::*;
use ndarray::{arr3, rcarr2};
use ndarray::{Slice, SliceInfo, SliceInfoElem};
use std::convert::TryFrom;

macro_rules! assert_panics {
    ($body:expr) => {
        if let Ok(v) = ::std::panic::catch_unwind(|| $body) {
            panic!("assertion failed: should_panic; \
            non-panicking result: {:?}", v);
        }
    };
    ($body:expr, $($arg:tt)*) => {
        if let Ok(_) = ::std::panic::catch_unwind(|| $body) {
            panic!($($arg)*);
        }
    };
}

#[test]
fn test_matmul_arcarray() {
    let mut A = ArcArray::<usize, _>::zeros((2, 3));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let mut B = ArcArray::<usize, _>::zeros((3, 4));
    for (i, elt) in B.iter_mut().enumerate() {
        *elt = i;
    }

    let c = A.dot(&B);
    println!("A = \n{:?}", A);
    println!("B = \n{:?}", B);
    println!("A x B = \n{:?}", c);
    unsafe {
        let result =
            ArcArray::from_shape_vec_unchecked((2, 4), vec![20, 23, 26, 29, 56, 68, 80, 92]);
        assert_eq!(c.shape(), result.shape());
        assert!(c.iter().zip(result.iter()).all(|(a, b)| a == b));
        assert!(c == result);
    }
}

#[allow(unused)]
fn arrayview_shrink_lifetime<'a, 'b: 'a>(view: ArrayView1<'b, f64>) -> ArrayView1<'a, f64> {
    view.reborrow()
}

#[allow(unused)]
fn arrayviewmut_shrink_lifetime<'a, 'b: 'a>(
    view: ArrayViewMut1<'b, f64>,
) -> ArrayViewMut1<'a, f64> {
    view.reborrow()
}

#[test]
#[cfg(feature = "std")]
fn test_mat_mul() {
    // smoke test, a big matrix multiplication of uneven size
    let (n, m) = (45, 33);
    let a = ArcArray::linspace(0., ((n * m) - 1) as f32, n as usize * m as usize).reshape((n, m));
    let b = ArcArray::eye(m);
    assert_eq!(a.dot(&b), a);
    let c = ArcArray::eye(n);
    assert_eq!(c.dot(&a), a);
}

#[deny(unsafe_code)]
#[test]
fn test_slice() {
    let mut A = ArcArray::<usize, _>::zeros((3, 4, 5));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let vi = A.slice(s![1.., ..;2, NewAxis, Slice::new(0, None, 2)]);
    assert_eq!(vi.shape(), &[2, 2, 1, 3]);
    let vi = A.slice(s![.., .., ..]);
    assert_eq!(vi.shape(), A.shape());
    assert!(vi.iter().zip(A.iter()).all(|(a, b)| a == b));
}

#[deny(unsafe_code)]
#[test]
fn test_slice_ix0() {
    let arr = arr0(5);
    assert_eq!(arr.slice(s![]), aview0(&5));
}

#[test]
fn test_slice_edge_cases() {
    let mut arr = Array3::<u8>::zeros((3, 4, 5));
    arr.slice_collapse(s![0..0;-1, .., ..]);
    assert_eq!(arr.shape(), &[0, 4, 5]);
    let mut arr = Array2::<u8>::from_shape_vec((1, 1).strides((10, 1)), vec![5]).unwrap();
    arr.slice_collapse(s![1..1, ..]);
    assert_eq!(arr.shape(), &[0, 1]);
}

#[test]
fn test_slice_inclusive_range() {
    let arr = array![[1, 2, 3], [4, 5, 6]];
    assert_eq!(arr.slice(s![1..=1, 1..=2]), array![[5, 6]]);
    assert_eq!(arr.slice(s![1..=-1, -2..=2;-1]), array![[6, 5]]);
    assert_eq!(arr.slice(s![0..=-1, 0..=2;2]), array![[1, 3], [4, 6]]);
}

/// Test that the compiler can infer a type for a sliced array from the
/// arguments to `s![]`.
///
/// This test relies on the fact that `.dot()` is implemented for both
/// `ArrayView1` and `ArrayView2`, so the compiler needs to determine which
/// type is the correct result for the `.slice()` call.
#[test]
fn test_slice_infer() {
    let a = array![1., 2.];
    let b = array![[3., 4.], [5., 6.]];
    b.slice(s![..-1, ..]).dot(&a);
    // b.slice(s![0, ..]).dot(&a);
}

#[test]
fn test_slice_with_many_dim() {
    let mut A = ArcArray::<usize, _>::zeros(&[3, 1, 4, 1, 3, 2, 1][..]);
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let vi = A.slice(s![..2, NewAxis, .., ..;2, NewAxis, ..1, ..1, 1.., ..]);
    let new_shape = &[2, 1, 1, 2, 1, 1, 1, 1, 1][..];
    assert_eq!(vi.shape(), new_shape);
    let correct = array![
        [A[&[0, 0, 0, 0, 0, 1, 0][..]], A[&[0, 0, 2, 0, 0, 1, 0][..]]],
        [A[&[1, 0, 0, 0, 0, 1, 0][..]], A[&[1, 0, 2, 0, 0, 1, 0][..]]]
    ]
    .into_shape(new_shape)
    .unwrap();
    assert_eq!(vi, correct);

    let vi = A.slice(s![..2, 0, ..;2, 0, 0, 1, 0]);
    assert_eq!(vi.shape(), &[2, 2][..]);
    let correct = array![
        [A[&[0, 0, 0, 0, 0, 1, 0][..]], A[&[0, 0, 2, 0, 0, 1, 0][..]]],
        [A[&[1, 0, 0, 0, 0, 1, 0][..]], A[&[1, 0, 2, 0, 0, 1, 0][..]]]
    ];
    assert_eq!(vi, correct);
}

#[test]
fn test_slice_range_variable() {
    let range = 1..4;
    let arr = array![0, 1, 2, 3, 4];
    assert_eq!(arr.slice(s![range]), array![1, 2, 3]);
}

#[test]
fn test_slice_args_eval_range_once() {
    let mut eval_count = 0;
    {
        let mut range = || {
            eval_count += 1;
            1..4
        };
        let arr = array![0, 1, 2, 3, 4];
        assert_eq!(arr.slice(s![range()]), array![1, 2, 3]);
    }
    assert_eq!(eval_count, 1);
}

#[test]
fn test_slice_args_eval_step_once() {
    let mut eval_count = 0;
    {
        let mut step = || {
            eval_count += 1;
            -1
        };
        let arr = array![0, 1, 2, 3, 4];
        assert_eq!(arr.slice(s![1..4;step()]), array![3, 2, 1]);
    }
    assert_eq!(eval_count, 1);
}

#[test]
fn test_slice_array_fixed() {
    let mut arr = Array3::<f64>::zeros((5, 2, 5));
    let info = s![1.., 1, NewAxis, ..;2];
    arr.slice(info);
    arr.slice_mut(info);
    arr.view().slice_move(info);
    let info2 = s![1.., 1, ..;2];
    arr.view().slice_collapse(info2);
}

#[test]
fn test_slice_dyninput_array_fixed() {
    let mut arr = Array3::<f64>::zeros((5, 2, 5)).into_dyn();
    let info = s![1.., 1, NewAxis, ..;2];
    arr.slice(info);
    arr.slice_mut(info);
    arr.view().slice_move(info);
    let info2 = s![1.., 1, ..;2];
    arr.view().slice_collapse(info2);
}

#[test]
fn test_slice_array_dyn() {
    let mut arr = Array3::<f64>::zeros((5, 2, 5));
    let info = SliceInfo::<_, Ix3, IxDyn>::try_from([
        SliceInfoElem::from(1..),
        SliceInfoElem::from(1),
        SliceInfoElem::from(NewAxis),
        SliceInfoElem::from(Slice::from(..).step_by(2)),
    ])
    .unwrap();
    arr.slice(info);
    arr.slice_mut(info);
    arr.view().slice_move(info);
    let info2 = SliceInfo::<_, Ix3, IxDyn>::try_from([
        SliceInfoElem::from(1..),
        SliceInfoElem::from(1),
        SliceInfoElem::from(Slice::from(..).step_by(2)),
    ])
    .unwrap();
    arr.view().slice_collapse(info2);
}

#[test]
fn test_slice_dyninput_array_dyn() {
    let mut arr = Array3::<f64>::zeros((5, 2, 5)).into_dyn();
    let info = SliceInfo::<_, Ix3, IxDyn>::try_from([
        SliceInfoElem::from(1..),
        SliceInfoElem::from(1),
        SliceInfoElem::from(NewAxis),
        SliceInfoElem::from(Slice::from(..).step_by(2)),
    ])
    .unwrap();
    arr.slice(info);
    arr.slice_mut(info);
    arr.view().slice_move(info);
    let info2 = SliceInfo::<_, Ix3, IxDyn>::try_from([
        SliceInfoElem::from(1..),
        SliceInfoElem::from(1),
        SliceInfoElem::from(Slice::from(..).step_by(2)),
    ])
    .unwrap();
    arr.view().slice_collapse(info2);
}

#[test]
fn test_slice_dyninput_vec_fixed() {
    let mut arr = Array3::<f64>::zeros((5, 2, 5)).into_dyn();
    let info = &SliceInfo::<_, Ix3, Ix3>::try_from(vec![
        SliceInfoElem::from(1..),
        SliceInfoElem::from(1),
        SliceInfoElem::from(NewAxis),
        SliceInfoElem::from(Slice::from(..).step_by(2)),
    ])
    .unwrap();
    arr.slice(info);
    arr.slice_mut(info);
    arr.view().slice_move(info);
    let info2 = SliceInfo::<_, Ix3, Ix2>::try_from(vec![
        SliceInfoElem::from(1..),
        SliceInfoElem::from(1),
        SliceInfoElem::from(Slice::from(..).step_by(2)),
    ])
    .unwrap();
    arr.view().slice_collapse(info2);
}

#[test]
fn test_slice_dyninput_vec_dyn() {
    let mut arr = Array3::<f64>::zeros((5, 2, 5)).into_dyn();
    let info = &SliceInfo::<_, Ix3, IxDyn>::try_from(vec![
        SliceInfoElem::from(1..),
        SliceInfoElem::from(1),
        SliceInfoElem::from(NewAxis),
        SliceInfoElem::from(Slice::from(..).step_by(2)),
    ])
    .unwrap();
    arr.slice(info);
    arr.slice_mut(info);
    arr.view().slice_move(info);
    let info2 = SliceInfo::<_, Ix3, IxDyn>::try_from(vec![
        SliceInfoElem::from(1..),
        SliceInfoElem::from(1),
        SliceInfoElem::from(Slice::from(..).step_by(2)),
    ])
    .unwrap();
    arr.view().slice_collapse(info2);
}

#[test]
fn test_slice_with_subview_and_new_axis() {
    let mut arr = ArcArray::<usize, _>::zeros((3, 5, 4));
    for (i, elt) in arr.iter_mut().enumerate() {
        *elt = i;
    }

    let vi = arr.slice(s![NewAxis, 1.., 2, ..;2]);
    assert_eq!(vi.shape(), &[1, 2, 2]);
    assert!(vi
        .iter()
        .zip(
            arr.index_axis(Axis(1), 2)
                .slice(s![1.., ..;2])
                .insert_axis(Axis(0))
                .iter()
        )
        .all(|(a, b)| a == b));

    let vi = arr.slice(s![1, NewAxis, 2, ..;2]);
    assert_eq!(vi.shape(), &[1, 2]);
    assert!(vi
        .iter()
        .zip(
            arr.index_axis(Axis(0), 1)
                .index_axis(Axis(0), 2)
                .slice(s![..;2])
                .insert_axis(Axis(0))
                .iter()
        )
        .all(|(a, b)| a == b));

    let vi = arr.slice(s![1, 2, 3]);
    assert_eq!(vi.shape(), &[]);
    assert_eq!(vi, Array0::from_elem((), arr[(1, 2, 3)]));
}

#[test]
fn test_slice_collapse_with_indices() {
    let mut arr = ArcArray::<usize, _>::zeros((3, 5, 4));
    for (i, elt) in arr.iter_mut().enumerate() {
        *elt = i;
    }

    {
        let mut vi = arr.view();
        vi.slice_collapse(s![1.., 2, ..;2]);
        assert_eq!(vi.shape(), &[2, 1, 2]);
        assert!(vi
            .iter()
            .zip(arr.slice(s![1.., 2..3, ..;2]).iter())
            .all(|(a, b)| a == b));

        let mut vi = arr.view();
        vi.slice_collapse(s![1, 2, ..;2]);
        assert_eq!(vi.shape(), &[1, 1, 2]);
        assert!(vi
            .iter()
            .zip(arr.slice(s![1..2, 2..3, ..;2]).iter())
            .all(|(a, b)| a == b));

        let mut vi = arr.view();
        vi.slice_collapse(s![1, 2, 3]);
        assert_eq!(vi.shape(), &[1, 1, 1]);
        assert_eq!(vi, Array3::from_elem((1, 1, 1), arr[(1, 2, 3)]));
    }

    // Do it to the ArcArray itself
    let elem = arr[(1, 2, 3)];
    let mut vi = arr;
    vi.slice_collapse(s![1, 2, 3]);
    assert_eq!(vi.shape(), &[1, 1, 1]);
    assert_eq!(vi, Array3::from_elem((1, 1, 1), elem));
}

#[test]
#[should_panic]
fn test_slice_collapse_with_newaxis() {
    let mut arr = Array2::<u8>::zeros((2, 3));
    arr.slice_collapse(s![0, 0, NewAxis]);
}

#[test]
fn test_multislice() {
    macro_rules! do_test {
        ($arr:expr, $($s:expr),*) => {
            {
                let arr = $arr;
                let copy = arr.clone();
                assert_eq!(
                    arr.multi_slice_mut(($($s,)*)),
                    ($(copy.clone().slice_mut($s),)*)
                );
            }
        };
    }

    let mut arr = Array1::from_iter(0..48).into_shape((8, 6)).unwrap();

    assert_eq!(
        (arr.clone().view_mut(),),
        arr.multi_slice_mut((s![.., ..],)),
    );
    assert_eq!(arr.multi_slice_mut(()), ());
    do_test!(&mut arr, s![0, ..]);
    do_test!(&mut arr, s![0, ..], s![1, ..]);
    do_test!(&mut arr, s![0, ..], s![-1, ..]);
    do_test!(&mut arr, s![0, ..], s![1.., ..]);
    do_test!(&mut arr, s![1, ..], s![..;2, ..]);
    do_test!(&mut arr, s![..2, ..], s![2.., ..]);
    do_test!(&mut arr, s![1..;2, ..], s![..;2, ..]);
    do_test!(&mut arr, s![..;-2, ..], s![..;2, ..]);
    do_test!(&mut arr, s![..;12, ..], s![3..;3, ..]);
    do_test!(&mut arr, s![3, ..], s![..-1;-2, ..]);
    do_test!(&mut arr, s![0, ..], s![1, ..], s![2, ..]);
    do_test!(&mut arr, s![0, ..], s![1, ..], s![2, ..], s![3, ..]);
}

#[test]
fn test_multislice_intersecting() {
    assert_panics!({
        let mut arr = Array2::<u8>::zeros((8, 6));
        arr.multi_slice_mut((s![3, .., NewAxis], s![3, ..]));
    });
    assert_panics!({
        let mut arr = Array2::<u8>::zeros((8, 6));
        arr.multi_slice_mut((s![3, ..], s![3.., ..]));
    });
    assert_panics!({
        let mut arr = Array2::<u8>::zeros((8, 6));
        arr.multi_slice_mut((s![3, ..], s![..;3, NewAxis, ..]));
    });
    assert_panics!({
        let mut arr = Array2::<u8>::zeros((8, 6));
        arr.multi_slice_mut((s![..;6, ..], s![3..;3, ..]));
    });
    assert_panics!({
        let mut arr = Array2::<u8>::zeros((8, 6));
        arr.multi_slice_mut((s![2, ..], s![..-1;-2, ..]));
    });
    assert_panics!({
        let mut arr = Array2::<u8>::zeros((8, 6));
        arr.multi_slice_mut((s![4, ..], s![3, ..], s![3, ..]));
    });
    assert_panics!({
        let mut arr = Array2::<u8>::zeros((8, 6));
        arr.multi_slice_mut((s![3, ..], s![4, ..], s![3, ..]));
    });
    assert_panics!({
        let mut arr = Array2::<u8>::zeros((8, 6));
        arr.multi_slice_mut((s![3, ..], s![3, ..], s![4, ..]));
    });
    assert_panics!({
        let mut arr = Array2::<u8>::zeros((8, 6));
        arr.multi_slice_mut((s![3, ..], s![3, ..], s![4, ..], s![3, ..]));
    });
}

#[should_panic]
#[test]
fn index_out_of_bounds() {
    let mut a = Array::<i32, _>::zeros((3, 4));
    a[[3, 2]] = 1;
}

#[should_panic]
#[test]
fn slice_oob() {
    let a = ArcArray::<i32, _>::zeros((3, 4));
    let _vi = a.slice(s![..10, ..]);
}

#[should_panic]
#[test]
fn slice_axis_oob() {
    let a = ArcArray::<i32, _>::zeros((3, 4));
    let _vi = a.slice_axis(Axis(0), Slice::new(0, Some(10), 1));
}

#[should_panic]
#[test]
fn slice_wrong_dim() {
    let a = ArcArray::<i32, _>::zeros(vec![3, 4, 5]);
    let _vi = a.slice(s![.., ..]);
}

#[test]
fn test_index() {
    let mut A = ArcArray::<usize, _>::zeros((2, 3));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    for ((i, j), a) in zip(indices((2, 3)), &A) {
        assert_eq!(*a, A[[i, j]]);
    }

    let vi = A.slice(s![1.., ..;2]);
    let mut it = vi.iter();
    for ((i, j), x) in zip(indices((1, 2)), &mut it) {
        assert_eq!(*x, vi[[i, j]]);
    }
    assert!(it.next().is_none());
}

#[test]
fn test_index_arrays() {
    let a = Array1::from_iter(0..12);
    assert_eq!(a[1], a[[1]]);
    let v = a.view().into_shape((3, 4)).unwrap();
    assert_eq!(a[1], v[[0, 1]]);
    let w = v.into_shape((2, 2, 3)).unwrap();
    assert_eq!(a[1], w[[0, 0, 1]]);
}

#[test]
#[allow(clippy::assign_op_pattern)]
fn test_add() {
    let mut A = ArcArray::<usize, _>::zeros((2, 2));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let B = A.clone();
    A = A + &B;
    assert_eq!(A[[0, 0]], 0);
    assert_eq!(A[[0, 1]], 2);
    assert_eq!(A[[1, 0]], 4);
    assert_eq!(A[[1, 1]], 6);
}

#[test]
fn test_multidim() {
    let mut mat = ArcArray::zeros(2 * 3 * 4 * 5 * 6).reshape((2, 3, 4, 5, 6));
    mat[(0, 0, 0, 0, 0)] = 22u8;
    {
        for (i, elt) in mat.iter_mut().enumerate() {
            *elt = i as u8;
        }
    }
    assert_eq!(mat.shape(), &[2, 3, 4, 5, 6]);
}

/*
array([[[ 7,  6],
        [ 5,  4],
        [ 3,  2],
        [ 1,  0]],

       [[15, 14],
        [13, 12],
        [11, 10],
        [ 9,  8]]])
*/
#[test]
fn test_negative_stride_arcarray() {
    let mut mat = ArcArray::zeros((2, 4, 2));
    mat[[0, 0, 0]] = 1.0f32;
    for (i, elt) in mat.iter_mut().enumerate() {
        *elt = i as f32;
    }

    {
        let vi = mat.slice(s![.., ..;-1, ..;-1]);
        assert_eq!(vi.shape(), &[2, 4, 2]);
        // Test against sequential iterator
        let seq = [
            7f32, 6., 5., 4., 3., 2., 1., 0., 15., 14., 13., 12., 11., 10., 9., 8.,
        ];
        for (a, b) in vi.iter().zip(seq.iter()) {
            assert_eq!(*a, *b);
        }
    }
    {
        let vi = mat.slice(s![.., ..;-5, ..]);
        let seq = [6., 7., 14., 15.];
        for (a, b) in vi.iter().zip(seq.iter()) {
            assert_eq!(*a, *b);
        }
    }
}

#[test]
fn test_cow() {
    let mut mat = ArcArray::zeros((2, 2));
    mat[[0, 0]] = 1;
    let n = mat.clone();
    mat[[0, 1]] = 2;
    mat[[1, 0]] = 3;
    mat[[1, 1]] = 4;
    assert_eq!(mat[[0, 0]], 1);
    assert_eq!(mat[[0, 1]], 2);
    assert_eq!(n[[0, 0]], 1);
    assert_eq!(n[[0, 1]], 0);
    assert_eq!(n.get((0, 1)), Some(&0));
    let mut rev = mat.reshape(4);
    rev.slice_collapse(s![..;-1]);
    assert_eq!(rev[0], 4);
    assert_eq!(rev[1], 3);
    assert_eq!(rev[2], 2);
    assert_eq!(rev[3], 1);
    let before = rev.clone();
    // mutation
    rev[0] = 5;
    assert_eq!(rev[0], 5);
    assert_eq!(rev[1], 3);
    assert_eq!(rev[2], 2);
    assert_eq!(rev[3], 1);
    assert_eq!(before[0], 4);
    assert_eq!(before[1], 3);
    assert_eq!(before[2], 2);
    assert_eq!(before[3], 1);
}

#[test]
fn test_cow_shrink() {
    // A test for clone-on-write in the case that
    // mutation shrinks the array and gives it different strides
    //
    let mut mat = ArcArray::zeros((2, 3));
    //mat.slice_collapse(s![.., ..;2]);
    mat[[0, 0]] = 1;
    let n = mat.clone();
    mat[[0, 1]] = 2;
    mat[[0, 2]] = 3;
    mat[[1, 0]] = 4;
    mat[[1, 1]] = 5;
    mat[[1, 2]] = 6;
    assert_eq!(mat[[0, 0]], 1);
    assert_eq!(mat[[0, 1]], 2);
    assert_eq!(n[[0, 0]], 1);
    assert_eq!(n[[0, 1]], 0);
    assert_eq!(n.get((0, 1)), Some(&0));
    // small has non-C strides this way
    let mut small = mat.reshape(6);
    small.slice_collapse(s![4..;-1]);
    assert_eq!(small[0], 6);
    assert_eq!(small[1], 5);
    let before = small.clone();
    // mutation
    // small gets back C strides in CoW.
    small[1] = 9;
    assert_eq!(small[0], 6);
    assert_eq!(small[1], 9);
    assert_eq!(before[0], 6);
    assert_eq!(before[1], 5);
}

#[test]
#[cfg(feature = "std")]
fn test_sub() {
    let mat = ArcArray::linspace(0., 15., 16).reshape((2, 4, 2));
    let s1 = mat.index_axis(Axis(0), 0);
    let s2 = mat.index_axis(Axis(0), 1);
    assert_eq!(s1.shape(), &[4, 2]);
    assert_eq!(s2.shape(), &[4, 2]);
    let n = ArcArray::linspace(8., 15., 8).reshape((4, 2));
    assert_eq!(n, s2);
    let m = ArcArray::from(vec![2., 3., 10., 11.]).reshape((2, 2));
    assert_eq!(m, mat.index_axis(Axis(1), 1));
}

#[should_panic]
#[test]
#[cfg(feature = "std")]
fn test_sub_oob_1() {
    let mat = ArcArray::linspace(0., 15., 16).reshape((2, 4, 2));
    mat.index_axis(Axis(0), 2);
}

#[test]
#[cfg(feature = "approx")]
fn test_select() {
    use approx::assert_abs_diff_eq;

    // test for 2-d array
    let x = arr2(&[
        [0., 1.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [0., 1.],
        [0., 1.],
    ]);
    let r = x.select(Axis(0), &[1, 3, 5]);
    let c = x.select(Axis(1), &[1]);
    let r_target = arr2(&[[1., 0.], [1., 0.], [0., 1.]]);
    let c_target = arr2(&[[1., 0., 0., 0., 0., 1., 1.]]);
    assert_abs_diff_eq!(r, r_target);
    assert_abs_diff_eq!(c, c_target.t());

    // test for 3-d array
    let y = arr3(&[
        [[1., 2., 3.], [1.5, 1.5, 3.]],
        [[1., 2., 8.], [1., 2.5, 3.]],
    ]);
    let r = y.select(Axis(1), &[1]);
    let c = y.select(Axis(2), &[1]);
    let r_target = arr3(&[[[1.5, 1.5, 3.]], [[1., 2.5, 3.]]]);
    let c_target = arr3(&[[[2.], [1.5]], [[2.], [2.5]]]);
    assert_abs_diff_eq!(r, r_target);
    assert_abs_diff_eq!(c, c_target);
}

#[test]
fn test_select_1d() {
    let x = arr1(&[0, 1, 2, 3, 4, 5, 6]);
    let r1 = x.select(Axis(0), &[1, 3, 4, 2, 2, 5]);
    assert_eq!(r1, arr1(&[1, 3, 4, 2, 2, 5]));
    // select nothing
    let r2 = x.select(Axis(0), &[]);
    assert_eq!(r2, arr1(&[]));
    // select nothing from empty
    let r3 = r2.select(Axis(0), &[]);
    assert_eq!(r3, arr1(&[]));
}

#[test]
fn diag() {
    let d = arr2(&[[1., 2., 3.0f32]]).into_diag();
    assert_eq!(d.dim(), 1);
    let a = arr2(&[[1., 2., 3.0f32], [0., 0., 0.]]);
    let d = a.view().into_diag();
    assert_eq!(d.dim(), 2);
    let d = arr2::<f32, 0>(&[[]]).into_diag();
    assert_eq!(d.dim(), 0);
    let d = ArcArray::<f32, _>::zeros(()).into_diag();
    assert_eq!(d.dim(), 1);
}

/// Check that the merged shape is correct.
///
/// Note that this does not check the strides in the "merged" case!
#[test]
#[allow(clippy::cognitive_complexity)]
fn merge_axes() {
    macro_rules! assert_merged {
        ($arr:expr, $slice:expr, $take:expr, $into:expr) => {
            let mut v = $arr.slice($slice);
            let merged_len = v.len_of(Axis($take)) * v.len_of(Axis($into));
            assert!(v.merge_axes(Axis($take), Axis($into)));
            assert_eq!(v.len_of(Axis($take)), if merged_len == 0 { 0 } else { 1 });
            assert_eq!(v.len_of(Axis($into)), merged_len);
        };
    }
    macro_rules! assert_not_merged {
        ($arr:expr, $slice:expr, $take:expr, $into:expr) => {
            let mut v = $arr.slice($slice);
            let old_dim = v.raw_dim();
            let old_strides = v.strides().to_owned();
            assert!(!v.merge_axes(Axis($take), Axis($into)));
            assert_eq!(v.raw_dim(), old_dim);
            assert_eq!(v.strides(), &old_strides[..]);
        };
    }

    let a = Array4::<u8>::zeros((3, 4, 5, 4));

    assert_not_merged!(a, s![.., .., .., ..], 0, 0);
    assert_merged!(a, s![.., .., .., ..], 0, 1);
    assert_not_merged!(a, s![.., .., .., ..], 0, 2);
    assert_not_merged!(a, s![.., .., .., ..], 0, 3);
    assert_not_merged!(a, s![.., .., .., ..], 1, 0);
    assert_not_merged!(a, s![.., .., .., ..], 1, 1);
    assert_merged!(a, s![.., .., .., ..], 1, 2);
    assert_not_merged!(a, s![.., .., .., ..], 1, 3);
    assert_not_merged!(a, s![.., .., .., ..], 2, 1);
    assert_not_merged!(a, s![.., .., .., ..], 2, 2);
    assert_merged!(a, s![.., .., .., ..], 2, 3);
    assert_not_merged!(a, s![.., .., .., ..], 3, 0);
    assert_not_merged!(a, s![.., .., .., ..], 3, 1);
    assert_not_merged!(a, s![.., .., .., ..], 3, 2);
    assert_not_merged!(a, s![.., .., .., ..], 3, 3);

    assert_merged!(a, s![.., .., .., ..;2], 0, 1);
    assert_not_merged!(a, s![.., .., .., ..;2], 1, 0);
    assert_merged!(a, s![.., .., .., ..;2], 1, 2);
    assert_not_merged!(a, s![.., .., .., ..;2], 2, 1);
    assert_merged!(a, s![.., .., .., ..;2], 2, 3);
    assert_not_merged!(a, s![.., .., .., ..;2], 3, 2);

    assert_merged!(a, s![.., .., .., ..3], 0, 1);
    assert_not_merged!(a, s![.., .., .., ..3], 1, 0);
    assert_merged!(a, s![.., .., .., ..3], 1, 2);
    assert_not_merged!(a, s![.., .., .., ..3], 2, 1);
    assert_not_merged!(a, s![.., .., .., ..3], 2, 3);

    assert_merged!(a, s![.., .., ..;2, ..], 0, 1);
    assert_not_merged!(a, s![.., .., ..;2, ..], 1, 0);
    assert_not_merged!(a, s![.., .., ..;2, ..], 1, 2);
    assert_not_merged!(a, s![.., .., ..;2, ..], 2, 3);

    assert_merged!(a, s![.., ..;2, .., ..], 0, 1);
    assert_not_merged!(a, s![.., ..;2, .., ..], 1, 0);
    assert_not_merged!(a, s![.., ..;2, .., ..], 1, 2);
    assert_merged!(a, s![.., ..;2, .., ..], 2, 3);
    assert_not_merged!(a, s![.., ..;2, .., ..], 3, 2);

    let a = Array4::<u8>::zeros((3, 1, 5, 1).f());
    assert_merged!(a, s![.., .., ..;2, ..], 0, 1);
    assert_merged!(a, s![.., .., ..;2, ..], 0, 3);
    assert_merged!(a, s![.., .., ..;2, ..], 1, 0);
    assert_merged!(a, s![.., .., ..;2, ..], 1, 1);
    assert_merged!(a, s![.., .., ..;2, ..], 1, 2);
    assert_merged!(a, s![.., .., ..;2, ..], 1, 3);
    assert_merged!(a, s![.., .., ..;2, ..], 2, 1);
    assert_merged!(a, s![.., .., ..;2, ..], 2, 3);
    assert_merged!(a, s![.., .., ..;2, ..], 3, 0);
    assert_merged!(a, s![.., .., ..;2, ..], 3, 1);
    assert_merged!(a, s![.., .., ..;2, ..], 3, 2);
    assert_merged!(a, s![.., .., ..;2, ..], 3, 3);

    let a = Array4::<u8>::zeros((3, 0, 5, 1));
    assert_merged!(a, s![.., .., ..;2, ..], 0, 1);
    assert_merged!(a, s![.., .., ..;2, ..], 1, 1);
    assert_merged!(a, s![.., .., ..;2, ..], 2, 1);
    assert_merged!(a, s![.., .., ..;2, ..], 3, 1);
    assert_merged!(a, s![.., .., ..;2, ..], 1, 0);
    assert_merged!(a, s![.., .., ..;2, ..], 1, 2);
    assert_merged!(a, s![.., .., ..;2, ..], 1, 3);
}

#[test]
fn swapaxes() {
    let mut a = arr2(&[[1., 2.], [3., 4.0f32]]);
    let b = arr2(&[[1., 3.], [2., 4.0f32]]);
    assert!(a != b);
    a.swap_axes(0, 1);
    assert_eq!(a, b);
    a.swap_axes(1, 1);
    assert_eq!(a, b);
    assert_eq!(a.as_slice_memory_order(), Some(&[1., 2., 3., 4.][..]));
    assert_eq!(b.as_slice_memory_order(), Some(&[1., 3., 2., 4.][..]));
}

#[test]
fn permuted_axes() {
    let a = array![1].index_axis_move(Axis(0), 0);
    let permuted = a.view().permuted_axes([]);
    assert_eq!(a, permuted);

    let a = array![1];
    let permuted = a.view().permuted_axes([0]);
    assert_eq!(a, permuted);

    let a = Array::from_iter(0..24).into_shape((2, 3, 4)).unwrap();
    let permuted = a.view().permuted_axes([2, 1, 0]);
    for ((i0, i1, i2), elem) in a.indexed_iter() {
        assert_eq!(*elem, permuted[(i2, i1, i0)]);
    }
    let permuted = a.view().into_dyn().permuted_axes(&[0, 2, 1][..]);
    for ((i0, i1, i2), elem) in a.indexed_iter() {
        assert_eq!(*elem, permuted[&[i0, i2, i1][..]]);
    }

    let a = Array::from_iter(0..120).into_shape((2, 3, 4, 5)).unwrap();
    let permuted = a.view().permuted_axes([1, 0, 3, 2]);
    for ((i0, i1, i2, i3), elem) in a.indexed_iter() {
        assert_eq!(*elem, permuted[(i1, i0, i3, i2)]);
    }
    let permuted = a.view().into_dyn().permuted_axes(&[1, 2, 3, 0][..]);
    for ((i0, i1, i2, i3), elem) in a.indexed_iter() {
        assert_eq!(*elem, permuted[&[i1, i2, i3, i0][..]]);
    }
}

#[should_panic]
#[test]
fn permuted_axes_repeated_axis() {
    let a = Array::from_iter(0..24).into_shape((2, 3, 4)).unwrap();
    a.view().permuted_axes([1, 0, 1]);
}

#[should_panic]
#[test]
fn permuted_axes_missing_axis() {
    let a = Array::from_iter(0..24)
        .into_shape((2, 3, 4))
        .unwrap()
        .into_dyn();
    a.view().permuted_axes(&[2, 0][..]);
}

#[should_panic]
#[test]
fn permuted_axes_oob() {
    let a = Array::from_iter(0..24).into_shape((2, 3, 4)).unwrap();
    a.view().permuted_axes([1, 0, 3]);
}

#[test]
fn standard_layout() {
    let mut a = arr2(&[[1., 2.], [3., 4.0]]);
    assert!(a.is_standard_layout());
    a.swap_axes(0, 1);
    assert!(!a.is_standard_layout());
    a.swap_axes(0, 1);
    assert!(a.is_standard_layout());
    let x1 = a.index_axis(Axis(0), 0);
    assert!(x1.is_standard_layout());
    let x2 = a.index_axis(Axis(1), 0);
    assert!(!x2.is_standard_layout());
    let x3 = ArrayView1::from_shape(1.strides(2), &[1]).unwrap();
    assert!(x3.is_standard_layout());
    let x4 = ArrayView2::from_shape((0, 2).strides((0, 1)), &[1, 2]).unwrap();
    assert!(x4.is_standard_layout());
}

#[test]
fn iter_size_hint() {
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    {
        let mut it = a.iter();
        assert_eq!(it.size_hint(), (4, Some(4)));
        it.next();
        assert_eq!(it.size_hint().0, 3);
        it.next();
        assert_eq!(it.size_hint().0, 2);
        it.next();
        assert_eq!(it.size_hint().0, 1);
        it.next();
        assert_eq!(it.size_hint().0, 0);
        assert!(it.next().is_none());
        assert_eq!(it.size_hint().0, 0);
    }

    a.swap_axes(0, 1);
    {
        let mut it = a.iter();
        assert_eq!(it.size_hint(), (4, Some(4)));
        it.next();
        assert_eq!(it.size_hint().0, 3);
        it.next();
        assert_eq!(it.size_hint().0, 2);
        it.next();
        assert_eq!(it.size_hint().0, 1);
        it.next();
        assert_eq!(it.size_hint().0, 0);
        assert!(it.next().is_none());
        assert_eq!(it.size_hint().0, 0);
    }
}

#[test]
fn zero_axes() {
    let mut a = arr1::<f32>(&[]);
    for _ in a.iter() {
        panic!();
    }
    a.map(|_| panic!());
    a.map_inplace(|_| panic!());
    a.for_each(|_| panic!());
    println!("{:?}", a);
    let b = arr2::<f32, 0>(&[[], [], [], []]);
    println!("{:?}\n{:?}", b.shape(), b);

    // we can even get a subarray of b
    let bsub = b.index_axis(Axis(0), 2);
    assert_eq!(bsub.dim(), 0);
}

#[test]
fn equality() {
    let a = arr2(&[[1., 2.], [3., 4.]]);
    let mut b = arr2(&[[1., 2.], [2., 4.]]);
    assert!(a != b);
    b[(1, 0)] = 3.;
    assert!(a == b);

    // make sure we can compare different shapes without failure.
    let c = arr2(&[[1., 2.]]);
    assert!(a != c);
}

#[test]
fn map1() {
    let a = arr2(&[[1., 2.], [3., 4.]]);
    let b = a.map(|&x| (x / 3.) as isize);
    assert_eq!(b, arr2(&[[0, 0], [1, 1]]));
    // test map to reference with array's lifetime.
    let c = a.map(|x| x);
    assert_eq!(a[(0, 0)], *c[(0, 0)]);
}

#[test]
fn as_slice_memory_order() {
    // test that mutation breaks sharing
    let a = rcarr2(&[[1., 2.], [3., 4.0f32]]);
    let mut b = a.clone();
    for elt in b.as_slice_memory_order_mut().unwrap() {
        *elt = 0.;
    }
    assert!(a != b, "{:?} != {:?}", a, b);
}

#[test]
fn array0_into_scalar() {
    // With this kind of setup, the `Array`'s pointer is not the same as the
    // underlying `Vec`'s pointer.
    let a: Array0<i32> = array![4, 5, 6, 7].index_axis_move(Axis(0), 2);
    assert_ne!(a.as_ptr(), a.into_raw_vec().as_ptr());
    // `.into_scalar()` should still work correctly.
    let a: Array0<i32> = array![4, 5, 6, 7].index_axis_move(Axis(0), 2);
    assert_eq!(a.into_scalar(), 6);

    // It should work for zero-size elements too.
    let a: Array0<()> = array![(), (), (), ()].index_axis_move(Axis(0), 2);
    assert_eq!(a.into_scalar(), ());
}

#[test]
fn array_view0_into_scalar() {
    // With this kind of setup, the `Array`'s pointer is not the same as the
    // underlying `Vec`'s pointer.
    let a: Array0<i32> = array![4, 5, 6, 7].index_axis_move(Axis(0), 2);
    assert_ne!(a.as_ptr(), a.into_raw_vec().as_ptr());
    // `.into_scalar()` should still work correctly.
    let a: Array0<i32> = array![4, 5, 6, 7].index_axis_move(Axis(0), 2);
    assert_eq!(a.view().into_scalar(), &6);

    // It should work for zero-size elements too.
    let a: Array0<()> = array![(), (), (), ()].index_axis_move(Axis(0), 2);
    assert_eq!(a.view().into_scalar(), &());
}

#[test]
fn array_view_mut0_into_scalar() {
    // With this kind of setup, the `Array`'s pointer is not the same as the
    // underlying `Vec`'s pointer.
    let a: Array0<i32> = array![4, 5, 6, 7].index_axis_move(Axis(0), 2);
    assert_ne!(a.as_ptr(), a.into_raw_vec().as_ptr());
    // `.into_scalar()` should still work correctly.
    let mut a: Array0<i32> = array![4, 5, 6, 7].index_axis_move(Axis(0), 2);
    assert_eq!(a.view_mut().into_scalar(), &6);

    // It should work for zero-size elements too.
    let mut a: Array0<()> = array![(), (), (), ()].index_axis_move(Axis(0), 2);
    assert_eq!(a.view_mut().into_scalar(), &());
}

#[test]
fn owned_array1() {
    let mut a = Array::from(vec![1, 2, 3, 4]);
    for elt in a.iter_mut() {
        *elt = 2;
    }
    for elt in a.iter() {
        assert_eq!(*elt, 2);
    }
    assert_eq!(a.shape(), &[4]);

    let mut a = Array::zeros((2, 2));
    let mut b = ArcArray::zeros((2, 2));
    a[(1, 1)] = 3;
    b[(1, 1)] = 3;
    assert_eq!(a, b);

    let c = a.clone();

    let d1 = &a + &b;
    let d2 = a + b;
    assert!(c != d1);
    assert_eq!(d1, d2);
}

#[test]
fn owned_array_with_stride() {
    let v: Vec<_> = (0..12).collect();
    let dim = (2, 3, 2);
    let strides = (1, 4, 2);

    let a = Array::from_shape_vec(dim.strides(strides), v).unwrap();
    assert_eq!(a.strides(), &[1, 4, 2]);
}

#[test]
fn owned_array_discontiguous() {
    use std::iter::repeat;
    let v: Vec<_> = (0..12).flat_map(|x| repeat(x).take(2)).collect();
    let dim = (3, 2, 2);
    let strides = (8, 4, 2);

    let a = Array::from_shape_vec(dim.strides(strides), v).unwrap();
    assert_eq!(a.strides(), &[8, 4, 2]);
    println!("{:?}", a.iter().cloned().collect::<Vec<_>>());
    itertools::assert_equal(a.iter().cloned(), 0..12);
}

#[test]
fn owned_array_discontiguous_drop() {
    use std::cell::RefCell;
    use std::collections::BTreeSet;
    use std::rc::Rc;

    struct InsertOnDrop<T: Ord>(Rc<RefCell<BTreeSet<T>>>, Option<T>);
    impl<T: Ord> Drop for InsertOnDrop<T> {
        fn drop(&mut self) {
            let InsertOnDrop(ref set, ref mut value) = *self;
            set.borrow_mut().insert(value.take().expect("double drop!"));
        }
    }

    let set = Rc::new(RefCell::new(BTreeSet::new()));
    {
        let v: Vec<_> = (0..12)
            .map(|x| InsertOnDrop(set.clone(), Some(x)))
            .collect();
        let mut a = Array::from_shape_vec((2, 6), v).unwrap();
        // discontiguous and non-zero offset
        a.slice_collapse(s![.., 1..]);
    }
    // each item was dropped exactly once
    itertools::assert_equal(set.borrow().iter().cloned(), 0..12);
}

macro_rules! assert_matches {
    ($value:expr, $pat:pat) => {
        match $value {
            $pat => {}
            ref err => panic!(
                "assertion failed: `{}` matches `{}` found: {:?}",
                stringify!($value),
                stringify!($pat),
                err
            ),
        }
    };
}

#[test]
fn from_vec_dim_stride_empty_1d() {
    let empty: [f32; 0] = [];
    assert_matches!(Array::from_shape_vec(0.strides(1), empty.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_0d() {
    let empty: [f32; 0] = [];
    let one = [1.];
    let two = [1., 2.];
    // too few elements
    assert_matches!(
        Array::from_shape_vec(().strides(()), empty.to_vec()),
        Err(_)
    );
    // exact number of elements
    assert_matches!(Array::from_shape_vec(().strides(()), one.to_vec()), Ok(_));
    // too many are ok
    assert_matches!(Array::from_shape_vec(().strides(()), two.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_1() {
    let two = [1., 2.];
    let d = Ix2(2, 1);
    let s = d.default_strides();
    assert_matches!(Array::from_shape_vec(d.strides(s), two.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_2() {
    let two = [1., 2.];
    let d = Ix2(1, 2);
    let s = d.default_strides();
    assert_matches!(Array::from_shape_vec(d.strides(s), two.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_3() {
    let a = arr3(&[[[1]], [[2]], [[3]]]);
    let d = a.raw_dim();
    let s = d.default_strides();
    assert_matches!(
        Array::from_shape_vec(d.strides(s), a.as_slice().unwrap().to_vec()),
        Ok(_)
    );
}

#[test]
fn from_vec_dim_stride_2d_4() {
    let a = arr3(&[[[1]], [[2]], [[3]]]);
    let d = a.raw_dim();
    let s = d.fortran_strides();
    assert_matches!(
        Array::from_shape_vec(d.strides(s), a.as_slice().unwrap().to_vec()),
        Ok(_)
    );
}

#[test]
fn from_vec_dim_stride_2d_5() {
    let a = arr3(&[[[1, 2, 3]]]);
    let d = a.raw_dim();
    let s = d.fortran_strides();
    assert_matches!(
        Array::from_shape_vec(d.strides(s), a.as_slice().unwrap().to_vec()),
        Ok(_)
    );
}

#[test]
fn from_vec_dim_stride_2d_6() {
    let a = [1., 2., 3., 4., 5., 6.];
    let d = (2, 1, 1);
    let s = (2, 2, 1);
    assert_matches!(Array::from_shape_vec(d.strides(s), a.to_vec()), Ok(_));

    let d = (1, 2, 1);
    let s = (2, 2, 1);
    assert_matches!(Array::from_shape_vec(d.strides(s), a.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_7() {
    // empty arrays can have 0 strides
    let a: [f32; 0] = [];
    // [[]] shape=[4, 0], strides=[0, 1]
    let d = (4, 0);
    let s = (0, 1);
    assert_matches!(Array::from_shape_vec(d.strides(s), a.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_8() {
    // strides of length 1 axes can be zero
    let a = [1.];
    let d = (1, 1);
    let s = (0, 1);
    assert_matches!(Array::from_shape_vec(d.strides(s), a.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_rejects() {
    let two = [1., 2.];
    let d = (2, 2);
    let s = (1, 0);
    assert_matches!(Array::from_shape_vec(d.strides(s), two.to_vec()), Err(_));

    let d = (2, 2);
    let s = (0, 1);
    assert_matches!(Array::from_shape_vec(d.strides(s), two.to_vec()), Err(_));
}

#[test]
fn views() {
    let a = ArcArray::from(vec![1, 2, 3, 4]).reshape((2, 2));
    let b = a.view();
    assert_eq!(a, b);
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.clone() + a.clone(), &b + &b);
    assert_eq!(a.clone() + b, &b + &b);
    a.clone()[(0, 0)] = 99;
    assert_eq!(b[(0, 0)], 1);

    assert_eq!(
        a.view().into_iter().cloned().collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
}

#[test]
fn view_mut() {
    let mut a = ArcArray::from(vec![1, 2, 3, 4]).reshape((2, 2));
    for elt in &mut a.view_mut() {
        *elt = 0;
    }
    assert_eq!(a, Array::zeros((2, 2)));
    {
        let mut b = a.view_mut();
        b[(0, 0)] = 7;
    }
    assert_eq!(a[(0, 0)], 7);

    for elt in a.view_mut() {
        *elt = 2;
    }
    assert_eq!(a, ArcArray::from_elem((2, 2), 2));
}

#[test]
fn slice_mut() {
    let mut a = ArcArray::from(vec![1, 2, 3, 4]).reshape((2, 2));
    for elt in a.slice_mut(s![.., ..]) {
        *elt = 0;
    }
    assert_eq!(a, aview2(&[[0, 0], [0, 0]]));

    let mut b = arr2(&[[1, 2, 3], [4, 5, 6]]);
    let c = b.clone(); // make sure we can mutate b even if it has to be unshared first
    for elt in b.slice_mut(s![.., ..1]) {
        *elt = 0;
    }
    assert_eq!(b, aview2(&[[0, 2, 3], [0, 5, 6]]));
    assert!(c != b);

    for elt in b.slice_mut(s![.., ..;2]) {
        *elt = 99;
    }
    assert_eq!(b, aview2(&[[99, 2, 99], [99, 5, 99]]));
}

#[test]
fn assign_ops() {
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    let b = arr2(&[[1., 3.], [2., 4.]]);
    (*&mut a.view_mut()) += &b;
    assert_eq!(a, arr2(&[[2., 5.], [5., 8.]]));

    a -= &b;
    a -= &b;
    assert_eq!(a, arr2(&[[0., -1.,], [1., 0.]]));

    a += 1.;
    assert_eq!(a, arr2(&[[1., 0.,], [2., 1.]]));
    a *= 10.;
    a /= 5.;
    assert_eq!(a, arr2(&[[2., 0.,], [4., 2.]]));
}

#[test]
fn aview() {
    let a = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
    let data = [[1., 2., 3.], [4., 5., 6.]];
    let b = aview2(&data);
    assert_eq!(a, b);
    assert_eq!(b.shape(), &[2, 3]);
}

#[test]
fn aview_mut() {
    let mut data = [0; 16];
    {
        let mut a = aview_mut1(&mut data).into_shape((4, 4)).unwrap();
        {
            let mut slc = a.slice_mut(s![..2, ..;2]);
            slc += 1;
        }
    }
    assert_eq!(data, [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn transpose_view() {
    let a = arr2(&[[1, 2], [3, 4]]);
    let at = a.view().reversed_axes();
    assert_eq!(at, arr2(&[[1, 3], [2, 4]]));

    let a = arr2(&[[1, 2, 3], [4, 5, 6]]);
    let at = a.view().reversed_axes();
    assert_eq!(at, arr2(&[[1, 4], [2, 5], [3, 6]]));
}

#[test]
fn transpose_view_mut() {
    let mut a = arr2(&[[1, 2], [3, 4]]);
    let mut at = a.view_mut().reversed_axes();
    at[[0, 1]] = 5;
    assert_eq!(at, arr2(&[[1, 5], [2, 4]]));

    let mut a = arr2(&[[1, 2, 3], [4, 5, 6]]);
    let mut at = a.view_mut().reversed_axes();
    at[[2, 1]] = 7;
    assert_eq!(at, arr2(&[[1, 4], [2, 5], [3, 7]]));
}

#[test]
fn reshape() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8];
    let v = aview1(&data);
    let u = v.into_shape((3, 3));
    assert!(u.is_err());
    let u = v.into_shape((2, 2, 2));
    assert!(u.is_ok());
    let u = u.unwrap();
    assert_eq!(u.shape(), &[2, 2, 2]);
    let s = u.into_shape((4, 2)).unwrap();
    assert_eq!(s.shape(), &[4, 2]);
    assert_eq!(s, aview2(&[[1, 2], [3, 4], [5, 6], [7, 8]]));
}

#[test]
#[should_panic(expected = "IncompatibleShape")]
fn reshape_error1() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8];
    let v = aview1(&data);
    let _u = v.into_shape((2, 5)).unwrap();
}

#[test]
#[should_panic(expected = "IncompatibleLayout")]
fn reshape_error2() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8];
    let v = aview1(&data);
    let mut u = v.into_shape((2, 2, 2)).unwrap();
    u.swap_axes(0, 1);
    let _s = u.into_shape((2, 4)).unwrap();
}

#[test]
fn reshape_f() {
    let mut u = Array::zeros((3, 4).f());
    for (i, elt) in enumerate(u.as_slice_memory_order_mut().unwrap()) {
        *elt = i as i32;
    }
    let v = u.view();
    println!("{:?}", v);

    // noop ok
    let v2 = v.into_shape((3, 4));
    assert!(v2.is_ok());
    assert_eq!(v, v2.unwrap());

    let u = v.into_shape((3, 2, 2));
    assert!(u.is_ok());
    let u = u.unwrap();
    println!("{:?}", u);
    assert_eq!(u.shape(), &[3, 2, 2]);
    let s = u.into_shape((4, 3)).unwrap();
    println!("{:?}", s);
    assert_eq!(s.shape(), &[4, 3]);
    assert_eq!(s, aview2(&[[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]));
}

#[test]
#[allow(clippy::cognitive_complexity)]
fn insert_axis() {
    defmac!(test_insert orig, index, new => {
        let res = orig.insert_axis(Axis(index));
        assert_eq!(res, new);
        assert!(res.is_standard_layout());
    });

    let v = 1;
    test_insert!(aview0(&v), 0, arr1(&[1]));
    assert!(::std::panic::catch_unwind(|| aview0(&v).insert_axis(Axis(1))).is_err());

    test_insert!(arr1(&[1, 2, 3]), 0, arr2(&[[1, 2, 3]]));
    test_insert!(arr1(&[1, 2, 3]), 1, arr2(&[[1], [2], [3]]));
    assert!(::std::panic::catch_unwind(|| arr1(&[1, 2, 3]).insert_axis(Axis(2))).is_err());

    test_insert!(
        arr2(&[[1, 2, 3], [4, 5, 6]]),
        0,
        arr3(&[[[1, 2, 3], [4, 5, 6]]])
    );
    test_insert!(
        arr2(&[[1, 2, 3], [4, 5, 6]]),
        1,
        arr3(&[[[1, 2, 3]], [[4, 5, 6]]])
    );
    test_insert!(
        arr2(&[[1, 2, 3], [4, 5, 6]]),
        2,
        arr3(&[[[1], [2], [3]], [[4], [5], [6]]])
    );
    assert!(
        ::std::panic::catch_unwind(|| arr2(&[[1, 2, 3], [4, 5, 6]]).insert_axis(Axis(3))).is_err()
    );

    test_insert!(
        Array3::<u8>::zeros((3, 4, 5)),
        0,
        Array4::<u8>::zeros((1, 3, 4, 5))
    );
    test_insert!(
        Array3::<u8>::zeros((3, 4, 5)),
        1,
        Array4::<u8>::zeros((3, 1, 4, 5))
    );
    test_insert!(
        Array3::<u8>::zeros((3, 4, 5)),
        3,
        Array4::<u8>::zeros((3, 4, 5, 1))
    );
    assert!(
        ::std::panic::catch_unwind(|| Array3::<u8>::zeros((3, 4, 5)).insert_axis(Axis(4))).is_err()
    );

    test_insert!(
        Array6::<u8>::zeros((2, 3, 4, 3, 2, 3)),
        0,
        ArrayD::<u8>::zeros(vec![1, 2, 3, 4, 3, 2, 3])
    );
    test_insert!(
        Array6::<u8>::zeros((2, 3, 4, 3, 2, 3)),
        3,
        ArrayD::<u8>::zeros(vec![2, 3, 4, 1, 3, 2, 3])
    );
    test_insert!(
        Array6::<u8>::zeros((2, 3, 4, 3, 2, 3)),
        6,
        ArrayD::<u8>::zeros(vec![2, 3, 4, 3, 2, 3, 1])
    );
    assert!(::std::panic::catch_unwind(
        || Array6::<u8>::zeros((2, 3, 4, 3, 2, 3)).insert_axis(Axis(7))
    )
    .is_err());

    test_insert!(
        ArrayD::<u8>::zeros(vec![3, 4, 5]),
        0,
        ArrayD::<u8>::zeros(vec![1, 3, 4, 5])
    );
    test_insert!(
        ArrayD::<u8>::zeros(vec![3, 4, 5]),
        1,
        ArrayD::<u8>::zeros(vec![3, 1, 4, 5])
    );
    test_insert!(
        ArrayD::<u8>::zeros(vec![3, 4, 5]),
        3,
        ArrayD::<u8>::zeros(vec![3, 4, 5, 1])
    );
    assert!(
        ::std::panic::catch_unwind(|| ArrayD::<u8>::zeros(vec![3, 4, 5]).insert_axis(Axis(4)))
            .is_err()
    );
}

#[test]
fn insert_axis_f() {
    defmac!(test_insert_f orig, index, new => {
        let res = orig.insert_axis(Axis(index));
        assert_eq!(res, new);
        assert!(res.t().is_standard_layout());
    });

    test_insert_f!(
        Array0::from_shape_vec(().f(), vec![1]).unwrap(),
        0,
        arr1(&[1])
    );
    assert!(
        ::std::panic::catch_unwind(|| Array0::from_shape_vec(().f(), vec![1])
            .unwrap()
            .insert_axis(Axis(1)))
        .is_err()
    );

    test_insert_f!(Array1::<u8>::zeros((3).f()), 0, Array2::<u8>::zeros((1, 3)));
    test_insert_f!(Array1::<u8>::zeros((3).f()), 1, Array2::<u8>::zeros((3, 1)));
    assert!(
        ::std::panic::catch_unwind(|| Array1::<u8>::zeros((3).f()).insert_axis(Axis(2))).is_err()
    );

    test_insert_f!(
        Array3::<u8>::zeros((3, 4, 5).f()),
        1,
        Array4::<u8>::zeros((3, 1, 4, 5))
    );
    assert!(
        ::std::panic::catch_unwind(|| Array3::<u8>::zeros((3, 4, 5).f()).insert_axis(Axis(4)))
            .is_err()
    );

    test_insert_f!(
        ArrayD::<u8>::zeros(vec![3, 4, 5].f()),
        1,
        ArrayD::<u8>::zeros(vec![3, 1, 4, 5])
    );
    assert!(::std::panic::catch_unwind(
        || ArrayD::<u8>::zeros(vec![3, 4, 5].f()).insert_axis(Axis(4))
    )
    .is_err());
}

#[test]
fn insert_axis_view() {
    let a = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]];

    assert_eq!(
        a.index_axis(Axis(1), 0).insert_axis(Axis(0)),
        array![[[1, 2], [5, 6], [9, 10]]]
    );
    assert_eq!(
        a.index_axis(Axis(1), 0).insert_axis(Axis(1)),
        array![[[1, 2]], [[5, 6]], [[9, 10]]]
    );
    assert_eq!(
        a.index_axis(Axis(1), 0).insert_axis(Axis(2)),
        array![[[1], [2]], [[5], [6]], [[9], [10]]]
    );
}

#[test]
fn arithmetic_broadcast() {
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    let b = a.clone() * aview0(&1.);
    assert_eq!(a, b);
    a.swap_axes(0, 1);
    let b = a.clone() / aview0(&1.);
    assert_eq!(a, b);

    // reference
    let a = arr2(&[[2], [3], [4]]);
    let b = arr1(&[5, 6, 7]);
    assert_eq!(&a + &b, arr2(&[[7, 8, 9], [8, 9, 10], [9, 10, 11]]));
    assert_eq!(
        a.clone() - &b,
        arr2(&[[-3, -4, -5], [-2, -3, -4], [-1, -2, -3]])
    );
    assert_eq!(
        a.clone() * b.clone(),
        arr2(&[[10, 12, 14], [15, 18, 21], [20, 24, 28]])
    );
    assert_eq!(&b / a, arr2(&[[2, 3, 3], [1, 2, 2], [1, 1, 1]]));

    // Negative strides and non-contiguous memory
    let s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let s = Array3::from_shape_vec((2, 3, 2).strides((1, 4, 2)), s.to_vec()).unwrap();
    let a = s.slice(s![..;-1,..;2,..]);
    let b = s.slice(s![..2, -1, ..]);
    let mut c = s.clone();
    c.collapse_axis(Axis(2), 1);
    let c = c.slice(s![1,..;2,..]);
    assert_eq!(
        &a.to_owned() + &b,
        arr3(&[[[11, 15], [20, 24]], [[10, 14], [19, 23]]])
    );
    assert_eq!(
        &a + b.into_owned() + c,
        arr3(&[[[15, 19], [32, 36]], [[14, 18], [31, 35]]])
    );

    // shared array
    let sa = a.to_shared();
    let sa2 = sa.to_shared();
    let sb = b.to_shared();
    let sb2 = sb.to_shared();
    let sc = c.to_shared();
    let sc2 = sc.into_shared();
    assert_eq!(
        sa2 + &sb2 + sc2.into_owned(),
        arr3(&[[[15, 19], [32, 36]], [[14, 18], [31, 35]]])
    );

    // Same shape
    let a = s.slice(s![..;-1, ..;2, ..]);
    let b = s.slice(s![.., ..;2, ..]);
    assert_eq!(a.shape(), b.shape());
    assert_eq!(&a + &b, arr3(&[[[3, 7], [19, 23]], [[3, 7], [19, 23]]]));
}

#[test]
fn char_array() {
    // test compilation & basics of non-numerical array
    let cc = ArcArray::from_iter("alphabet".chars()).reshape((4, 2));
    assert!(cc.index_axis(Axis(1), 0) == ArcArray::from_iter("apae".chars()));
}

#[test]
fn scalar_ops() {
    let a = Array::<i32, _>::zeros((5, 5));
    let b = &a + 1;
    let c = (&a + &a + 2) - 3;
    println!("{:?}", b);
    println!("{:?}", c);

    let a = Array::<f32, _>::zeros((2, 2));
    let b = (1. + a) * 3.;
    assert_eq!(b, arr2(&[[3., 3.], [3., 3.]]));

    let a = arr1(&[false, true, true]);
    let b = &a ^ true;
    let c = true ^ &a;
    assert_eq!(b, c);
    assert_eq!(true & &a, a);
    assert_eq!(b, arr1(&[true, false, false]));
    assert_eq!(true ^ &a, !a);

    let zero = Array::<f32, _>::zeros((2, 2));
    let one = &zero + 1.;
    assert_eq!(0. * &one, zero);
    assert_eq!(&one * 0., zero);
    assert_eq!((&one + &one).sum(), 8.);
    assert_eq!(&one / 2., 0.5 * &one);
    assert_eq!(&one % 1., zero);

    let zero = Array::<i32, _>::zeros((2, 2));
    let one = &zero + 1;
    assert_eq!(one.clone() << 3, 8 * &one);
    assert_eq!(3 << one.clone(), 6 * &one);

    assert_eq!(&one << 3, 8 * &one);
    assert_eq!(3 << &one, 6 * &one);
}

#[test]
#[cfg(feature = "std")]
fn split_at() {
    let mut a = arr2(&[[1., 2.], [3., 4.]]);

    {
        let (c0, c1) = a.view().split_at(Axis(1), 1);

        assert_eq!(c0, arr2(&[[1.], [3.]]));
        assert_eq!(c1, arr2(&[[2.], [4.]]));
    }

    {
        let (mut r0, mut r1) = a.view_mut().split_at(Axis(0), 1);
        r0[[0, 1]] = 5.;
        r1[[0, 0]] = 8.;
    }
    assert_eq!(a, arr2(&[[1., 5.], [8., 4.]]));

    let b = ArcArray::linspace(0., 59., 60).reshape((3, 4, 5));

    let (left, right) = b.view().split_at(Axis(2), 2);
    assert_eq!(left.shape(), [3, 4, 2]);
    assert_eq!(right.shape(), [3, 4, 3]);
    assert_eq!(
        left,
        arr3(&[
            [[0., 1.], [5., 6.], [10., 11.], [15., 16.]],
            [[20., 21.], [25., 26.], [30., 31.], [35., 36.]],
            [[40., 41.], [45., 46.], [50., 51.], [55., 56.]]
        ])
    );

    // we allow for an empty right view when index == dim[axis]
    let (_, right) = b.view().split_at(Axis(1), 4);
    assert_eq!(right.shape(), [3, 0, 5]);
}

#[test]
#[should_panic]
fn deny_split_at_axis_out_of_bounds() {
    let a = arr2(&[[1., 2.], [3., 4.]]);
    a.view().split_at(Axis(2), 0);
}

#[test]
#[should_panic]
fn deny_split_at_index_out_of_bounds() {
    let a = arr2(&[[1., 2.], [3., 4.]]);
    a.view().split_at(Axis(1), 3);
}

#[test]
#[cfg(feature = "std")]
fn test_range() {
    let a = Array::range(0., 5., 1.);
    assert_eq!(a.len(), 5);
    assert_eq!(a[0], 0.);
    assert_eq!(a[4], 4.);

    let b = Array::range(0., 2.2, 1.);
    assert_eq!(b.len(), 3);
    assert_eq!(b[0], 0.);
    assert_eq!(b[2], 2.);

    let c = Array::range(0., 5., 2.);
    assert_eq!(c.len(), 3);
    assert_eq!(c[0], 0.);
    assert_eq!(c[1], 2.);
    assert_eq!(c[2], 4.);

    let d = Array::range(1.0, 2.2, 0.1);
    assert_eq!(d.len(), 13);
    assert_eq!(d[0], 1.);
    assert_eq!(d[10], 2.);
    assert_eq!(d[12], 2.2);

    let e = Array::range(1., 1., 1.);
    assert_eq!(e.len(), 0);
    assert!(e.is_empty());
}

#[test]
fn test_f_order() {
    // Test that arrays are logically equal in every way,
    // even if the underlying memory order is different
    let c = arr2(&[[1, 2, 3], [4, 5, 6]]);
    let mut f = Array::zeros(c.dim().f());
    f.assign(&c);
    assert_eq!(f, c);
    assert_eq!(f.shape(), c.shape());
    assert_eq!(c.strides(), &[3, 1]);
    assert_eq!(f.strides(), &[1, 2]);
    itertools::assert_equal(f.iter(), c.iter());
    itertools::assert_equal(f.rows(), c.rows());
    itertools::assert_equal(f.outer_iter(), c.outer_iter());
    itertools::assert_equal(f.axis_iter(Axis(0)), c.axis_iter(Axis(0)));
    itertools::assert_equal(f.axis_iter(Axis(1)), c.axis_iter(Axis(1)));

    let dupc = &c + &c;
    let dupf = &f + &f;
    assert_eq!(dupc, dupf);
}

#[test]
fn to_owned_memory_order() {
    // check that .to_owned() makes f-contiguous arrays out of f-contiguous
    // input.
    let c = arr2(&[[1, 2, 3], [4, 5, 6]]);
    let mut f = c.view();

    // transposed array
    f.swap_axes(0, 1);
    let fo = f.to_owned();
    assert_eq!(f, fo);
    assert_eq!(f.strides(), fo.strides());

    // negated stride axis
    f.invert_axis(Axis(1));
    let fo2 = f.to_owned();
    assert_eq!(f, fo2);
    assert_eq!(f.strides(), fo2.strides());
}

#[test]
fn to_owned_neg_stride() {
    let mut c = arr2(&[[1, 2, 3], [4, 5, 6]]);
    c.slice_collapse(s![.., ..;-1]);
    let co = c.to_owned();
    assert_eq!(c, co);
    assert_eq!(c.strides(), co.strides());
}

#[test]
fn discontiguous_owned_to_owned() {
    let mut c = arr2(&[[1, 2, 3], [4, 5, 6]]);
    c.slice_collapse(s![.., ..;2]);

    let co = c.to_owned();
    assert_eq!(c.strides(), &[3, 2]);
    assert_eq!(co.strides(), &[2, 1]);
    assert_eq!(c, co);
}

#[test]
fn map_memory_order() {
    let a = arr3(&[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, -1, -2]]]);
    let mut v = a.view();
    v.swap_axes(0, 1);
    let amap = v.map(|x| *x >= 3);
    assert_eq!(amap.dim(), v.dim());
    assert_eq!(amap.strides(), v.strides());
}

#[test]
fn test_view_from_shape() {
    let s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let a = ArrayView::from_shape((2, 3, 2), &s).unwrap();
    let mut answer = Array::from(s.to_vec()).into_shape((2, 3, 2)).unwrap();
    assert_eq!(a, answer);

    // custom strides (row major)
    let a = ArrayView::from_shape((2, 3, 2).strides((6, 2, 1)), &s).unwrap();
    assert_eq!(a, answer);

    // custom strides (col major)
    let a = ArrayView::from_shape((2, 3, 2).strides((1, 2, 6)), &s).unwrap();
    assert_eq!(a, answer.t());

    // negative strides
    let a = ArrayView::from_shape((2, 3, 2).strides((6, (-2isize) as usize, 1)), &s).unwrap();
    answer.invert_axis(Axis(1));
    assert_eq!(a, answer);
}

#[test]
fn test_contiguous() {
    let c = arr3(&[[[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [7, 7, 7]]]);
    assert!(c.is_standard_layout());
    assert!(c.as_slice_memory_order().is_some());
    let v = c.slice(s![.., 0..1, ..]);
    assert!(!v.is_standard_layout());
    assert!(!v.as_slice_memory_order().is_some());

    let v = c.slice(s![1..2, .., ..]);
    assert!(v.is_standard_layout());
    assert!(v.as_slice_memory_order().is_some());
    let v = v.reversed_axes();
    assert!(!v.is_standard_layout());
    assert!(v.as_slice_memory_order().is_some());
    let mut v = v.reversed_axes();
    v.swap_axes(1, 2);
    assert!(!v.is_standard_layout());
    assert!(v.as_slice_memory_order().is_some());

    let a = Array::<f32, _>::zeros((20, 1));
    let b = Array::<f32, _>::zeros((20, 1).f());
    assert!(a.as_slice().is_some());
    assert!(b.as_slice().is_some());
    assert!(a.as_slice_memory_order().is_some());
    assert!(b.as_slice_memory_order().is_some());
    let a = a.t();
    let b = b.t();
    assert!(a.as_slice().is_some());
    assert!(b.as_slice().is_some());
    assert!(a.as_slice_memory_order().is_some());
    assert!(b.as_slice_memory_order().is_some());
}

#[test]
fn test_contiguous_neg_strides() {
    let s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    let a = ArrayView::from_shape((2, 3, 2).strides((1, 4, 2)), &s).unwrap();
    assert_eq!(
        a,
        arr3(&[[[0, 2], [4, 6], [8, 10]], [[1, 3], [5, 7], [9, 11]]])
    );
    assert!(a.as_slice_memory_order().is_some());

    let mut b = a.slice(s![..;1, ..;-1, ..;-1]);
    assert_eq!(
        b,
        arr3(&[[[10, 8], [6, 4], [2, 0]], [[11, 9], [7, 5], [3, 1]]])
    );
    assert!(b.as_slice_memory_order().is_some());

    b.swap_axes(1, 2);
    assert_eq!(b, arr3(&[[[10, 6, 2], [8, 4, 0]], [[11, 7, 3], [9, 5, 1]]]));
    assert!(b.as_slice_memory_order().is_some());

    b.invert_axis(Axis(0));
    assert_eq!(b, arr3(&[[[11, 7, 3], [9, 5, 1]], [[10, 6, 2], [8, 4, 0]]]));
    assert!(b.as_slice_memory_order().is_some());

    let mut c = b.reversed_axes();
    assert_eq!(
        c,
        arr3(&[[[11, 10], [9, 8]], [[7, 6], [5, 4]], [[3, 2], [1, 0]]])
    );
    assert!(c.as_slice_memory_order().is_some());

    c.merge_axes(Axis(1), Axis(2));
    assert_eq!(c, arr3(&[[[11, 10, 9, 8]], [[7, 6, 5, 4]], [[3, 2, 1, 0]]]));
    assert!(c.as_slice_memory_order().is_some());

    let d = b.remove_axis(Axis(1));
    assert_eq!(d, arr2(&[[11, 7, 3], [10, 6, 2]]));
    assert!(d.as_slice_memory_order().is_none());

    let e = b.remove_axis(Axis(2));
    assert_eq!(e, arr2(&[[11, 9], [10, 8]]));
    assert!(e.as_slice_memory_order().is_some());

    let f = e.insert_axis(Axis(2));
    assert_eq!(f, arr3(&[[[11], [9]], [[10], [8]]]));
    assert!(f.as_slice_memory_order().is_some());

    let mut g = b.clone();
    g.collapse_axis(Axis(1), 0);
    assert_eq!(g, arr3(&[[[11, 7, 3]], [[10, 6, 2]]]));
    assert!(g.as_slice_memory_order().is_none());

    b.collapse_axis(Axis(2), 0);
    assert_eq!(b, arr3(&[[[11], [9]], [[10], [8]]]));
    assert!(b.as_slice_memory_order().is_some());
}

#[test]
fn test_swap() {
    let mut a = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let b = a.clone();

    for i in 0..a.nrows() {
        for j in i + 1..a.ncols() {
            a.swap((i, j), (j, i));
        }
    }
    assert_eq!(a, b.t());
}

#[test]
fn test_uswap() {
    let mut a = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let b = a.clone();

    for i in 0..a.nrows() {
        for j in i + 1..a.ncols() {
            unsafe { a.uswap((i, j), (j, i)) };
        }
    }
    assert_eq!(a, b.t());
}

#[test]
fn test_shape() {
    let data = [0, 1, 2, 3, 4, 5];
    let a = Array::from_shape_vec((1, 2, 3), data.to_vec()).unwrap();
    let b = Array::from_shape_vec((1, 2, 3).f(), data.to_vec()).unwrap();
    let c = Array::from_shape_vec((1, 2, 3).strides((1, 3, 1)), data.to_vec()).unwrap();
    println!("{:?}", a);
    println!("{:?}", b);
    println!("{:?}", c);
    assert_eq!(a.strides(), &[6, 3, 1]);
    assert_eq!(b.strides(), &[1, 1, 2]);
    assert_eq!(c.strides(), &[1, 3, 1]);
}

#[test]
fn test_view_from_shape_ptr() {
    let data = [0, 1, 2, 3, 4, 5];
    let view = unsafe { ArrayView::from_shape_ptr((2, 3), data.as_ptr()) };
    assert_eq!(view, aview2(&[[0, 1, 2], [3, 4, 5]]));

    let mut data = data;
    let mut view = unsafe { ArrayViewMut::from_shape_ptr((2, 3), data.as_mut_ptr()) };
    view[[1, 2]] = 6;
    assert_eq!(view, aview2(&[[0, 1, 2], [3, 4, 6]]));
    view[[0, 1]] = 0;
    assert_eq!(view, aview2(&[[0, 0, 2], [3, 4, 6]]));
}

#[should_panic(expected = "Unsupported")]
#[cfg(debug_assertions)]
#[test]
fn test_view_from_shape_ptr_deny_neg_strides() {
    let data = [0, 1, 2, 3, 4, 5];
    let _view =
        unsafe { ArrayView::from_shape_ptr((2, 3).strides((-3isize as usize, 1)), data.as_ptr()) };
}

#[should_panic(expected = "Unsupported")]
#[cfg(debug_assertions)]
#[test]
fn test_view_mut_from_shape_ptr_deny_neg_strides() {
    let mut data = [0, 1, 2, 3, 4, 5];
    let _view = unsafe {
        ArrayViewMut::from_shape_ptr((2, 3).strides((-3isize as usize, 1)), data.as_mut_ptr())
    };
}

#[should_panic(expected = "Unsupported")]
#[cfg(debug_assertions)]
#[test]
fn test_raw_view_from_shape_ptr_deny_neg_strides() {
    let data = [0, 1, 2, 3, 4, 5];
    let _view = unsafe {
        RawArrayView::from_shape_ptr((2, 3).strides((-3isize as usize, 1)), data.as_ptr())
    };
}

#[should_panic(expected = "Unsupported")]
#[cfg(debug_assertions)]
#[test]
fn test_raw_view_mut_from_shape_ptr_deny_neg_strides() {
    let mut data = [0, 1, 2, 3, 4, 5];
    let _view = unsafe {
        RawArrayViewMut::from_shape_ptr((2, 3).strides((-3isize as usize, 1)), data.as_mut_ptr())
    };
}

#[test]
fn test_default() {
    let a = <Array<f32, Ix2> as Default>::default();
    assert_eq!(a, aview2(&[[0.0; 0]; 0]));

    #[derive(Default, Debug, PartialEq)]
    struct Foo(i32);
    let b = <Array<Foo, Ix0> as Default>::default();
    assert_eq!(b, arr0(Foo::default()));
}

#[test]
fn test_default_ixdyn() {
    let a = <Array<f32, IxDyn> as Default>::default();
    let b = <Array<f32, _>>::zeros(IxDyn(&[0]));
    assert_eq!(a, b);
}

#[test]
fn test_map_axis() {
    let a = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);

    let b = a.map_axis(Axis(0), |view| view.sum());
    let answer1 = arr1(&[22, 26, 30]);
    assert_eq!(b, answer1);
    let c = a.map_axis(Axis(1), |view| view.sum());
    let answer2 = arr1(&[6, 15, 24, 33]);
    assert_eq!(c, answer2);

    // Test zero-length axis case
    let arr = Array3::<f32>::zeros((3, 0, 4));
    let mut counter = 0;
    let result = arr.map_axis(Axis(1), |x| {
        assert_eq!(x.shape(), &[0]);
        counter += 1;
        counter
    });
    assert_eq!(result.shape(), &[3, 4]);
    itertools::assert_equal(result.iter().cloned().sorted(), 1..=3 * 4);

    let mut arr = Array3::<f32>::zeros((3, 0, 4));
    let mut counter = 0;
    let result = arr.map_axis_mut(Axis(1), |x| {
        assert_eq!(x.shape(), &[0]);
        counter += 1;
        counter
    });
    assert_eq!(result.shape(), &[3, 4]);
    itertools::assert_equal(result.iter().cloned().sorted(), 1..=3 * 4);
}

#[test]
fn test_accumulate_axis_inplace_noop() {
    let mut a = Array2::<u8>::zeros((0, 3));
    a.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    assert_eq!(a, Array2::zeros((0, 3)));

    let mut a = Array2::<u8>::zeros((3, 1));
    a.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr += prev);
    assert_eq!(a, Array2::zeros((3, 1)));
}

#[rustfmt::skip] // Allow block array formatting
#[test]
fn test_accumulate_axis_inplace_nonstandard_layout() {
    let a = arr2(&[[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10,11,12]]);

    let mut a_t = a.clone().reversed_axes();
    a_t.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    assert_eq!(a_t, aview2(&[[1, 4, 7, 10],
                             [3, 9, 15, 21],
                             [6, 15, 24, 33]]));

    let mut a0 = a.clone();
    a0.invert_axis(Axis(0));
    a0.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    assert_eq!(a0, aview2(&[[10, 11, 12],
                            [17, 19, 21],
                            [21, 24, 27],
                            [22, 26, 30]]));

    let mut a1 = a.clone();
    a1.invert_axis(Axis(1));
    a1.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr += prev);
    assert_eq!(a1, aview2(&[[3, 5, 6],
                            [6, 11, 15],
                            [9, 17, 24],
                            [12, 23, 33]]));
}

#[test]
fn test_to_vec() {
    let mut a = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);

    a.slice_collapse(s![..;-1, ..]);
    assert_eq!(a.row(3).to_vec(), vec![1, 2, 3]);
    assert_eq!(a.column(2).to_vec(), vec![12, 9, 6, 3]);
    a.slice_collapse(s![.., ..;-1]);
    assert_eq!(a.row(3).to_vec(), vec![3, 2, 1]);
}

#[test]
fn test_array_clone_unalias() {
    let a = Array::<i32, _>::zeros((3, 3));
    let mut b = a.clone();
    b.fill(1);
    assert!(a != b);
    assert_eq!(a, Array::<_, _>::zeros((3, 3)));
}

#[test]
fn test_array_clone_same_view() {
    let mut a = Array::from_iter(0..9).into_shape((3, 3)).unwrap();
    a.slice_collapse(s![..;-1, ..;-1]);
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn test_array2_from_diag() {
    let diag = arr1(&[0, 1, 2]);
    let x = Array2::from_diag(&diag);
    let x_exp = arr2(&[[0, 0, 0], [0, 1, 0], [0, 0, 2]]);
    assert_eq!(x, x_exp);

    // check 0 length array
    let diag = Array1::<f64>::zeros(0);
    let x = Array2::from_diag(&diag);
    assert_eq!(x.ndim(), 2);
    assert_eq!(x.shape(), [0, 0]);
}

#[test]
fn array_macros() {
    // array
    let a1 = array![1, 2, 3];
    assert_eq!(a1, arr1(&[1, 2, 3]));
    let a2 = array![[1, 2], [3, 4], [5, 6]];
    assert_eq!(a2, arr2(&[[1, 2], [3, 4], [5, 6]]));
    let a3 = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
    assert_eq!(a3, arr3(&[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]));
    let a4 = array![[[1, 2,], [3, 4,]], [[5, 6,], [7, 8,],],]; // trailing commas
    assert_eq!(a4, arr3(&[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]));

    let s = String::from("abc");
    let a2s = array![
        [String::from("w"), s],
        [String::from("x"), String::from("y")]
    ];
    assert_eq!(a2s[[0, 0]], "w");
    assert_eq!(a2s[[0, 1]], "abc");
    assert_eq!(a2s[[1, 0]], "x");
    assert_eq!(a2s[[1, 1]], "y");

    let empty1: Array<f32, Ix1> = array![];
    assert_eq!(empty1, array![]);
    let empty2: Array<f32, Ix2> = array![[]];
    assert_eq!(empty2, array![[]]);
}

#[cfg(test)]
mod as_standard_layout_tests {
    use super::*;
    use ndarray::Data;
    use std::fmt::Debug;

    fn test_as_standard_layout_for<S, D>(orig: ArrayBase<S, D>)
    where
        S: Data,
        S::Elem: Clone + Debug + PartialEq,
        D: Dimension,
    {
        let orig_is_standard = orig.is_standard_layout();
        let out = orig.as_standard_layout();
        assert!(out.is_standard_layout());
        assert_eq!(out, orig);
        assert_eq!(orig_is_standard, out.is_view());
    }

    #[test]
    fn test_f_layout() {
        let shape = (2, 2).f();
        let arr = Array::<i32, Ix2>::from_shape_vec(shape, vec![1, 2, 3, 4]).unwrap();
        assert!(!arr.is_standard_layout());
        test_as_standard_layout_for(arr);
    }

    #[test]
    fn test_c_layout() {
        let arr = Array::<i32, Ix2>::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        assert!(arr.is_standard_layout());
        test_as_standard_layout_for(arr);
    }

    #[test]
    fn test_f_layout_view() {
        let shape = (2, 2).f();
        let arr = Array::<i32, Ix2>::from_shape_vec(shape, vec![1, 2, 3, 4]).unwrap();
        let arr_view = arr.view();
        assert!(!arr_view.is_standard_layout());
        test_as_standard_layout_for(arr);
    }

    #[test]
    fn test_c_layout_view() {
        let arr = Array::<i32, Ix2>::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        let arr_view = arr.view();
        assert!(arr_view.is_standard_layout());
        test_as_standard_layout_for(arr_view);
    }

    #[test]
    fn test_zero_dimensional_array() {
        let arr_view = ArrayView1::<i32>::from(&[]);
        assert!(arr_view.is_standard_layout());
        test_as_standard_layout_for(arr_view);
    }

    #[test]
    fn test_custom_layout() {
        let shape = (1, 2, 3, 2).strides((12, 1, 2, 6));
        let arr_data: Vec<i32> = (0..12).collect();
        let arr = Array::<i32, Ix4>::from_shape_vec(shape, arr_data).unwrap();
        assert!(!arr.is_standard_layout());
        test_as_standard_layout_for(arr);
    }
}

#[cfg(test)]
mod array_cow_tests {
    use super::*;

    #[test]
    fn test_is_variant() {
        let arr: Array<i32, Ix2> = array![[1, 2], [3, 4]];
        let arr_cow = CowArray::<i32, Ix2>::from(arr.view());
        assert!(arr_cow.is_view());
        assert!(!arr_cow.is_owned());
        let arr_cow = CowArray::<i32, Ix2>::from(arr);
        assert!(arr_cow.is_owned());
        assert!(!arr_cow.is_view());
    }

    fn run_with_various_layouts(mut f: impl FnMut(Array2<i32>)) {
        for all in vec![
            Array2::from_shape_vec((7, 8), (0..7 * 8).collect()).unwrap(),
            Array2::from_shape_vec((7, 8).f(), (0..7 * 8).collect()).unwrap(),
        ] {
            f(all.clone());
            f(all.clone().slice_move(s![.., 2..5]));
            f(all.clone().slice_move(s![3..5, 2..5]));
            f(all.clone().slice_move(s![.., ..;2]));
            f(all.clone().slice_move(s![..;3, ..]));
            f(all.clone().slice_move(s![.., ..;-1]));
            f(all.clone().slice_move(s![..;-2, ..;-1]));
            f(all.clone().slice_move(s![2..5;-2, 3..6]));
            f(all.clone().slice_move(s![2..5;-2, 3..6;-1]));
        }
    }

    #[test]
    fn test_element_mutation() {
        run_with_various_layouts(|arr: Array2<i32>| {
            let mut expected = arr.clone();
            expected[(1, 1)] = 2;

            let mut arr_cow = CowArray::<i32, Ix2>::from(arr.view());
            arr_cow[(1, 1)] = 2;
            assert!(arr_cow.is_owned());
            assert_eq!(arr_cow, expected);

            let ptr = arr.as_ptr();
            let mut arr_cow = CowArray::<i32, Ix2>::from(arr);
            assert_eq!(arr_cow.as_ptr(), ptr);
            arr_cow[(1, 1)] = 2;
            assert_eq!(arr_cow.as_ptr(), ptr);
            assert_eq!(arr_cow, expected);
        });
    }

    #[test]
    fn test_clone() {
        run_with_various_layouts(|arr: Array2<i32>| {
            let arr_cow = CowArray::<i32, Ix2>::from(arr.view());
            let arr_cow_clone = arr_cow.clone();
            assert!(arr_cow_clone.is_view());
            assert_eq!(arr_cow, arr_cow_clone);
            assert_eq!(arr_cow.dim(), arr_cow_clone.dim());
            assert_eq!(arr_cow.strides(), arr_cow_clone.strides());

            let arr_cow = CowArray::<i32, Ix2>::from(arr);
            let arr_cow_clone = arr_cow.clone();
            assert!(arr_cow_clone.is_owned());
            assert_eq!(arr_cow, arr_cow_clone);
            assert_eq!(arr_cow.dim(), arr_cow_clone.dim());
            assert_eq!(arr_cow.strides(), arr_cow_clone.strides());
        });
    }

    #[test]
    fn test_clone_from() {
        fn assert_eq_contents_and_layout(
            arr1: &CowArray<'_, i32, Ix2>,
            arr2: &CowArray<'_, i32, Ix2>,
        ) {
            assert_eq!(arr1, arr2);
            assert_eq!(arr1.dim(), arr2.dim());
            assert_eq!(arr1.strides(), arr2.strides());
        }

        run_with_various_layouts(|arr: Array2<i32>| {
            run_with_various_layouts(|other_arr: Array2<i32>| {
                let arr_cow_src = CowArray::<i32, Ix2>::from(arr.view());
                let mut arr_cow_dst = CowArray::<i32, Ix2>::from(other_arr.clone());
                arr_cow_dst.clone_from(&arr_cow_src);
                assert!(arr_cow_dst.is_view());
                assert_eq_contents_and_layout(&arr_cow_src, &arr_cow_dst);

                let arr_cow_src = CowArray::<i32, Ix2>::from(arr.view());
                let mut arr_cow_dst = CowArray::<i32, Ix2>::from(other_arr.view());
                arr_cow_dst.clone_from(&arr_cow_src);
                assert!(arr_cow_dst.is_view());
                assert_eq_contents_and_layout(&arr_cow_src, &arr_cow_dst);

                let arr_cow_src = CowArray::<i32, Ix2>::from(arr.clone());
                let mut arr_cow_dst = CowArray::<i32, Ix2>::from(other_arr.view());
                arr_cow_dst.clone_from(&arr_cow_src);
                assert!(arr_cow_dst.is_owned());
                assert_eq_contents_and_layout(&arr_cow_src, &arr_cow_dst);

                let arr_cow_src = CowArray::<i32, Ix2>::from(arr.clone());
                let mut arr_cow_dst = CowArray::<i32, Ix2>::from(other_arr.clone());
                arr_cow_dst.clone_from(&arr_cow_src);
                assert!(arr_cow_dst.is_owned());
                assert_eq_contents_and_layout(&arr_cow_src, &arr_cow_dst);
            });
        });
    }

    #[test]
    fn test_into_owned() {
        run_with_various_layouts(|arr: Array2<i32>| {
            let before = CowArray::<i32, Ix2>::from(arr.view());
            let after = before.into_owned();
            assert_eq!(arr, after);

            let before = CowArray::<i32, Ix2>::from(arr.clone());
            let ptr = before.as_ptr();
            let after = before.into_owned();
            assert_eq!(after.as_ptr(), ptr);
            assert_eq!(arr, after);
        });
    }
}

#[test]
fn test_remove_index() {
    let mut a = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
    a.remove_index(Axis(0), 1);
    a.remove_index(Axis(1), 2);
    assert_eq!(a.shape(), &[3, 2]);
    assert_eq!(a, array![[1, 2], [7, 8], [10, 11]]);

    let mut a = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
    a.invert_axis(Axis(0));
    a.remove_index(Axis(0), 1);
    a.remove_index(Axis(1), 2);
    assert_eq!(a.shape(), &[3, 2]);
    assert_eq!(a, array![[10, 11], [4, 5], [1, 2]]);

    a.remove_index(Axis(1), 1);

    assert_eq!(a.shape(), &[3, 1]);
    assert_eq!(a, array![[10], [4], [1]]);
    a.remove_index(Axis(1), 0);
    assert_eq!(a.shape(), &[3, 0]);
    assert_eq!(a, array![[], [], []]);
}

#[should_panic(expected = "must be less")]
#[test]
fn test_remove_index_oob1() {
    let mut a = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
    a.remove_index(Axis(0), 4);
}

#[should_panic(expected = "must be less")]
#[test]
fn test_remove_index_oob2() {
    let mut a = array![[10], [4], [1]];
    a.remove_index(Axis(1), 0);
    assert_eq!(a.shape(), &[3, 0]);
    assert_eq!(a, array![[], [], []]);
    a.remove_index(Axis(0), 1); // ok
    assert_eq!(a, array![[], []]);
    a.remove_index(Axis(1), 0); // oob
}

#[should_panic(expected = "index out of bounds")]
#[test]
fn test_remove_index_oob3() {
    let mut a = array![[10], [4], [1]];
    a.remove_index(Axis(2), 0);
}
