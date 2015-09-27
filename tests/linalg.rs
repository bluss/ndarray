#![allow(non_snake_case)]

extern crate ndarray;

use ndarray::Array;
use ndarray::{arr1, arr2};

#[test]
fn identity()
{
    let e = ndarray::linalg::eye::<f32>(4);
    let e2 = e.mat_mul(&e);
    assert_eq!(e2, e);

    let e = ndarray::linalg::eye::<f32>(2);
    let ans = 
        arr2(&[[3.16227770, 0.00000000],
               [4.42718887, 0.63245525]]);

    assert!(ans.allclose(&ans.mat_mul(&e), 0.000001));
}

#[test]
fn chol()
{
    let _ = arr2(&[[1., 2.], [3., 4.]]); // not pos. def.
    let a = arr2(&[[10., 14.], [14., 20.]]); // aT a is pos def

    let chol = ndarray::linalg::cholesky(a);
    let ans = 
        arr2(&[[3.16227770, 0.00000000],
               [4.42718887, 0.63245525]]);

    assert!(ans.allclose(&chol, 0.001));

    // Compute bT b for a pos def matrix
    let b = Array::range(0.0f32, 9.).reshape_into((3, 3));
    let mut bt = b.clone();
    bt.swap_axes(0, 1);
    let bpd = bt.mat_mul(&b);
    println!("bpd=\n{:?}", bpd);
    let chol = ndarray::linalg::cholesky(bpd);
    println!("chol=\n{:.8?}", chol);

    let ans =
        arr2(&[[6.70820379, 0.00000000, 0.00000000],
               [8.04984474, 1.09544373, 0.00000000],
               [9.39148617, 2.19088745, 0.00000000]]);
    assert!(ans.allclose(&chol, 0.001));

    let a =
        arr2(&[[ 0.05201001,  0.22982409,  0.1014132 ],
               [ 0.22982409,  1.105822  ,  0.37946544],
               [ 0.1014132 ,  0.37946544,  1.16199134]]);
    let chol = ndarray::linalg::cholesky(a);

    let ans =
        arr2(&[[ 0.22805704,  0.        ,  0.        ],
               [ 1.00774829,  0.30044197,  0.        ],
               [ 0.44468348, -0.2285419 ,  0.95499557]]);
    assert!(ans.allclose(&chol, 0.001));
}

#[test]
fn subst()
{
    let lll =
        arr2(&[[ 0.22805704,  0.        ,  0.        ],
               [ 1.00774829,  0.30044197,  0.        ],
               [ 0.44468348, -0.2285419 ,  0.95499557]]);
    let ans = arr1::<f32>(&[4.384868, -8.050947, -0.827078]);

    assert!(ans.allclose(&ndarray::linalg::subst_fw(&lll, &arr1(&[1., 2., 3.])),
                         0.001));
}

#[test]
fn lst_squares()
{
    let xs =
        arr2(&[[ 2.,  3.],
               [-2., -1.],
               [ 1.,  5.],
               [-1.,  2.]]);
    let b = arr1(&[1., -1., 2., 1.]);
    let x_lstsq = ndarray::linalg::least_squares(&xs, &b);
    let ans = arr1(&[0.070632, 0.390335]);
    assert!(x_lstsq.allclose(&ans, 0.001));
}
