#![allow(non_snake_case)]

extern crate ndarray;

use ndarray::{Array, S, Si};
use ndarray::{arr0, arr1, arr2};
use ndarray::Indexes;
use ndarray::SliceRange;

#[test]
fn test_matmul_rcarray()
{
    let mut A = Array::<usize, _>::zeros((2, 3));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let mut B = Array::<usize, _>::zeros((3, 4));
    for (i, elt) in B.iter_mut().enumerate() {
        *elt = i;
    }

    let c = A.mat_mul(&B);
    println!("A = \n{:?}", A);
    println!("B = \n{:?}", B);
    println!("A x B = \n{:?}", c);
    unsafe {
        let result = Array::from_vec_dim((2, 4), vec![20, 23, 26, 29, 56, 68, 80, 92]);
        assert_eq!(c.shape(), result.shape());
        assert!(c.iter().zip(result.iter()).all(|(a,b)| a == b));
        assert!(c == result);
    }
}

#[test]
fn test_slice()
{
    let mut A = Array::<usize, _>::zeros((3, 4));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let vi = A.slice(&[(1..).slice(), (0..).step(2)]);
    assert_eq!(vi.dim(), (2, 2));
    let vi = A.slice(&[S, S]);
    assert_eq!(vi.shape(), A.shape());
    assert!(vi.iter().zip(A.iter()).all(|(a, b)| a == b));
}

#[test]
fn test_index()
{
    let mut A = Array::<usize, _>::zeros((2, 3));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    for ((i, j), a) in Indexes::new((2, 3)).zip(A.iter()) {
        assert_eq!(*a, A[(i, j)]);
    }

    let vi = A.slice(&[Si(1, None, 1), Si(0, None, 2)]);
    let mut it = vi.iter();
    for ((i, j), x) in Indexes::new((1, 2)).zip(it.by_ref()) {
        assert_eq!(*x, vi[(i, j)]);
    }
    assert!(it.next().is_none());
}

#[test]
fn test_add()
{
    let mut A = Array::<usize, _>::zeros((2, 2));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let B = A.clone();
    A.iadd(&B);
    assert_eq!(A[(0,0)], 0);
    assert_eq!(A[(0,1)], 2);
    assert_eq!(A[(1,0)], 4);
    assert_eq!(A[(1,1)], 6);
}

#[test]
fn test_multidim()
{
    let mut mat = Array::zeros(2*3*4*5*6).reshape_into((2,3,4,5,6));
    mat[(0,0,0,0,0)] = 22u8;
    {
        for (i, elt) in mat.iter_mut().enumerate() {
            *elt = i as u8;
        }
    }
    //println!("shape={:?}, strides={:?}", mat.shape(), mat.strides);
    assert_eq!(mat.dim(), (2,3,4,5,6));
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
fn test_negative_stride_rcarray()
{
    let mut mat = Array::zeros((2, 4, 2));
    mat[(0, 0, 0)] = 1.0f32;
    for (i, elt) in mat.iter_mut().enumerate() {
        *elt = i as f32;
    }

    {
        let vi = mat.slice(&[S, Si(0, None, -1), Si(0, None, -1)]);
        assert_eq!(vi.dim(), (2,4,2));
        // Test against sequential iterator
        let seq = [7f32,6., 5.,4.,3.,2.,1.,0.,15.,14.,13., 12.,11.,  10.,   9.,   8.];
        for (a, b) in vi.clone().iter().zip(seq.iter()) {
            assert_eq!(*a, *b);
        }
    }
    {
        let vi = mat.slice(&[S, Si(0, None, -5), S]);
        let seq = [6_f32, 7., 14., 15.];
        for (a, b) in vi.iter().zip(seq.iter()) {
            assert_eq!(*a, *b);
        }
    }
}

// Removed copy on write test, only makes sense with Rc storage
// #[test]
// fn test_cow()
// {
//     let mut mat = Array::<isize, _>::zeros((2,2));
//     mat[(0, 0)] = 1;
//     let n = mat.clone();
//     mat[(0, 1)] = 2;
//     mat[(1, 0)] = 3;
//     mat[(1, 1)] = 4;
//     assert_eq!(mat[(0,0)], 1);
//     assert_eq!(mat[(0,1)], 2);
//     assert_eq!(n[(0,0)], 1);
//     assert_eq!(n[(0,1)], 0);
//     let mat = mat.reshape_into(4);
//     let mut rev = mat.slice(&[Si(0, None, -1)]);
//     assert_eq!(rev[0], 4);
//     assert_eq!(rev[1], 3);
//     assert_eq!(rev[2], 2);
//     assert_eq!(rev[3], 1);
//     let before = rev.clone();
//     // mutation
//     rev[0] = 5;
//     assert_eq!(rev[0], 5);
//     assert_eq!(rev[1], 3);
//     assert_eq!(rev[2], 2);
//     assert_eq!(rev[3], 1);
//     assert_eq!(before[0], 4);
//     assert_eq!(before[1], 3);
//     assert_eq!(before[2], 2);
//     assert_eq!(before[3], 1);
// }

#[test]
fn test_sub()
{
    let mat = Array::range(0.0f32, 16.0).reshape_into((2, 4, 2));
    let s1 = mat.subview(0,0);
    let s2 = mat.subview(0,1);
    assert_eq!(s1.dim(), (4, 2));
    assert_eq!(s2.dim(), (4, 2));
    let n = Array::range(8.0f32, 16.0).reshape_into((4,2));
    assert_eq!(n.view(), s2);
    let m = Array::from_vec(vec![2f32, 3., 10., 11.]).reshape_into((2, 2));
    assert_eq!(m.view(), mat.subview(1, 1));
}

#[test]
fn diag()
{
    let a = arr2(&[[1., 2., 3.0f32]]);
    assert_eq!(a.diag().dim(), 1);
    let a = arr2(&[[1., 2., 3.0f32], [0., 0., 0.]]);
    assert_eq!(a.diag().dim(), 2);
    let a = arr2::<f32, _>(&[[]]);
    assert_eq!(a.diag().dim(), 0);
    let a = Array::<f32, _>::zeros(());
    assert_eq!(a.diag().dim(), 1);
}

#[test]
fn swapaxes()
{
    let mut a = arr2(&[[1., 2.], [3., 4.0f32]]);
    let     b = arr2(&[[1., 3.], [2., 4.0f32]]);
    assert!(a != b);
    a.swap_axes(0, 1);
    assert_eq!(a, b);
    a.swap_axes(1, 1);
    assert_eq!(a, b);
    assert!(a.raw_data() == [1., 2., 3., 4.]);
    assert!(b.raw_data() == [1., 3., 2., 4.]);
}

#[test]
fn standard_layout()
{
    let mut a = arr2(&[[1., 2.], [3., 4.0]]);
    assert!(a.is_standard_layout());
    a.swap_axes(0, 1);
    assert!(!a.is_standard_layout());
    a.swap_axes(0, 1);
    assert!(a.is_standard_layout());
    let x1 = a.subview(0, 0);
    assert!(x1.is_standard_layout());
    let x2 = a.subview(1, 0);
    assert!(!x2.is_standard_layout());
}

#[test]
fn assign()
{
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    let     b = arr2(&[[1., 3.], [2., 4.]]);
    a.assign(&b);
    assert_eq!(a, b);

    /* Test broadcasting */
    a.assign(&Array::zeros(1));
    assert_eq!(a, Array::zeros((2, 2)));
}

#[test]
fn dyn_dimension()
{
    let a = arr2(&[[1., 2.], [3., 4.0]]).reshape_into(vec![2, 2]);
    assert_eq!(&a - &a, Array::zeros(vec![2, 2]));

    let mut dim = vec![1; 1024];
    dim[16] = 4;
    dim[17] = 3;
    let z = Array::<f32, _>::zeros(dim.clone());
    assert_eq!(z.shape(), &dim[..]);
}

#[test]
fn sum_mean()
{
    let a = arr2(&[[1., 2.], [3., 4.]]);
    assert_eq!(a.sum(0), arr1(&[4., 6.]));
    assert_eq!(a.sum(1), arr1(&[3., 7.]));
    assert_eq!(a.mean(0), arr1(&[2., 3.]));
    assert_eq!(a.mean(1), arr1(&[1.5, 3.5]));
    assert_eq!(a.sum(1).sum(0), arr0(10.));
}

#[test]
fn iter_size_hint()
{
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
fn zero_axes()
{
    let a = arr1::<f32>(&[]);
    for _ in a.iter() {
        assert!(false);
    }
    println!("{:?}", a);
    let b = arr2::<f32, _>(&[[], [], [], []]);
    println!("{:?}\n{:?}", b.shape(), b);

    // we can even get a subarray of b
    let bsub = b.subview(0, 2);
    assert_eq!(bsub.dim(), 0);
}

#[test]
fn equality()
{
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
fn map1()
{
    let a = arr2(&[[1., 2.], [3., 4.]]);
    let b = a.map(|&x| (x / 3.) as isize);
    assert_eq!(b, arr2(&[[0, 0], [1, 1]]));
    // test map to reference with array's lifetime.
    let c = a.map(|x| x);
    assert_eq!(a[(0, 0)], *c[(0, 0)]);
}
