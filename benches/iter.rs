#![feature(test)]
#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]

extern crate test;
use rawpointer::PointerExt;
use test::black_box;
use test::Bencher;

use ndarray::prelude::*;
use ndarray::Slice;
use ndarray::{FoldWhile, Zip};

#[bench]
fn iter_sum_2d_regular(bench: &mut Bencher) {
    let a = Array::<i32, _>::zeros((64, 64));
    bench.iter(|| a.iter().sum::<i32>());
}

#[bench]
fn iter_sum_2d_cutout(bench: &mut Bencher) {
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = av;
    bench.iter(|| a.iter().sum::<i32>());
}

#[bench]
fn iter_all_2d_cutout(bench: &mut Bencher) {
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = av;
    bench.iter(|| a.iter().all(|&x| x >= 0));
}

#[bench]
fn iter_sum_2d_transpose(bench: &mut Bencher) {
    let a = Array::<i32, _>::zeros((66, 66));
    let a = a.t();
    bench.iter(|| a.iter().sum::<i32>());
}

#[bench]
fn iter_filter_sum_2d_u32(bench: &mut Bencher) {
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a.mapv(|x| (x * 100.) as u32);
    bench.iter(|| b.iter().filter(|&&x| x < 75).sum::<u32>());
}

#[bench]
fn iter_filter_sum_2d_f32(bench: &mut Bencher) {
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a * 100.;
    bench.iter(|| b.iter().filter(|&&x| x < 75.).sum::<f32>());
}

#[bench]
fn iter_filter_sum_2d_stride_u32(bench: &mut Bencher) {
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a.mapv(|x| (x * 100.) as u32);
    let b = b.slice(s![.., ..;2]);
    bench.iter(|| b.iter().filter(|&&x| x < 75).sum::<u32>());
}

#[bench]
fn iter_filter_sum_2d_stride_f32(bench: &mut Bencher) {
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a * 100.;
    let b = b.slice(s![.., ..;2]);
    bench.iter(|| b.iter().filter(|&&x| x < 75.).sum::<f32>());
}

#[bench]
fn iter_sum_2d_row_matrix(bench: &mut Bencher) {
    let a = Array::from_iter(0i32..64 * 64);
    let v = a.view().insert_axis(Axis(1));
    bench.iter(|| {
        let mut s = 0;
        for &elt in v.iter() {
            s += elt;
        }
        s
    });
}

#[bench]
fn iter_sum_2d_row_matrix_for_strided(bench: &mut Bencher) {
    let a = Array::from_iter(0i32..64 * 64).slice_move(s![..;2]);
    let v = a.view().insert_axis(Axis(1));
    bench.iter(|| {
        let mut s = 0;
        for &elt in v.iter() {
            s += elt;
        }
        s
    });
}

#[bench]
fn iter_sum_2d_row_matrix_sum_strided(bench: &mut Bencher) {
    let a = Array::from_iter(0i32..64 * 64).slice_move(s![..;2]);
    let v = a.view().insert_axis(Axis(1));
    bench.iter(|| {
        v.iter().sum::<i32>()
    });
}

#[bench]
fn iter_sum_2d_col_matrix(bench: &mut Bencher) {
    let a = Array::from_iter(0i32..64 * 64);
    let v = a.view().insert_axis(Axis(0));
    bench.iter(|| {
        let mut s = 0;
        for &elt in v.iter() {
            s += elt;
        }
        s
    });
}

#[bench]
fn iter_sum_2d_col_matrix_for_strided(bench: &mut Bencher) {
    let a = Array::from_iter(0i32..64 * 64).slice_move(s![..;2]);
    let v = a.view().insert_axis(Axis(0));
    bench.iter(|| {
        let mut s = 0;
        for &elt in v.iter() {
            s += elt;
        }
        s
    });
}

#[bench]
fn iter_sum_2d_col_matrix_sum_strided(bench: &mut Bencher) {
    let a = Array::from_iter(0i32..64 * 64).slice_move(s![..;2]);
    let v = a.view().insert_axis(Axis(0));
    bench.iter(|| {
        v.iter().sum::<i32>()
    });
}

#[bench]
fn iter_rev_step_by_contiguous(bench: &mut Bencher) {
    let a = Array::linspace(0., 1., 512);
    bench.iter(|| {
        a.iter().rev().step_by(2).for_each(|x| {
            black_box(x);
        })
    });
}

#[bench]
fn iter_rev_step_by_discontiguous(bench: &mut Bencher) {
    let mut a = Array::linspace(0., 1., 1024);
    a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    bench.iter(|| {
        a.iter().rev().step_by(2).for_each(|x| {
            black_box(x);
        })
    });
}

const ZIPSZ: usize = 10_000;

#[bench]
fn sum_3_std_zip1(bench: &mut Bencher) {
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c = vec![1; ZIPSZ];
    bench.iter(|| {
        a.iter()
            .zip(b.iter().zip(&c))
            .fold(0, |acc, (&a, (&b, &c))| acc + a + b + c)
    });
}

#[bench]
fn sum_3_std_zip2(bench: &mut Bencher) {
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c = vec![1; ZIPSZ];
    bench.iter(|| {
        a.iter()
            .zip(b.iter())
            .zip(&c)
            .fold(0, |acc, ((&a, &b), &c)| acc + a + b + c)
    });
}

#[bench]
fn sum_3_std_zip3(bench: &mut Bencher) {
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c = vec![1; ZIPSZ];
    bench.iter(|| {
        let mut s = 0;
        for ((&a, &b), &c) in a.iter().zip(b.iter()).zip(&c) {
            s += a + b + c
        }
        s
    });
}

#[bench]
fn vector_sum_3_std_zip(bench: &mut Bencher) {
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c = vec![1.; ZIPSZ];
    bench.iter(|| {
        for ((&a, &b), c) in a.iter().zip(b.iter()).zip(&mut c) {
            *c += a + b;
        }
    });
}

#[bench]
fn sum_3_azip(bench: &mut Bencher) {
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c = vec![1; ZIPSZ];
    bench.iter(|| {
        let mut s = 0;
        azip!((&a in &a, &b in &b, &c in &c) {
            s += a + b + c;
        });
        s
    });
}

#[bench]
fn sum_3_azip_fold(bench: &mut Bencher) {
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c = vec![1; ZIPSZ];
    bench.iter(|| {
        Zip::from(&a)
            .and(&b)
            .and(&c)
            .fold_while(0, |acc, &a, &b, &c| FoldWhile::Continue(acc + a + b + c))
            .into_inner()
    });
}

#[bench]
fn vector_sum_3_azip(bench: &mut Bencher) {
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c = vec![1.; ZIPSZ];
    bench.iter(|| {
        azip!((&a in &a, &b in &b, c in &mut c) {
            *c += a + b;
        });
    });
}

fn vector_sum3_unchecked(a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..c.len() {
        unsafe {
            *c.get_unchecked_mut(i) += *a.get_unchecked(i) + *b.get_unchecked(i);
        }
    }
}

#[bench]
fn vector_sum_3_zip_unchecked(bench: &mut Bencher) {
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c = vec![1.; ZIPSZ];
    bench.iter(move || {
        vector_sum3_unchecked(&a, &b, &mut c);
    });
}

#[bench]
fn vector_sum_3_zip_unchecked_manual(bench: &mut Bencher) {
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c = vec![1.; ZIPSZ];
    bench.iter(move || unsafe {
        let mut ap = a.as_ptr();
        let mut bp = b.as_ptr();
        let mut cp = c.as_mut_ptr();
        let cend = cp.add(c.len());
        while cp != cend {
            *cp.post_inc() += *ap.post_inc() + *bp.post_inc();
        }
    });
}

// index iterator size
const ISZ: usize = 16;
const I2DSZ: usize = 64;

#[bench]
fn indexed_iter_1d_ix1(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros(I2DSZ * I2DSZ);
    for (i, elt) in a.indexed_iter_mut() {
        *elt = i as _;
    }

    bench.iter(|| {
        for (i, &_elt) in a.indexed_iter() {
            //assert!(a[i] == elt);
            black_box(i);
        }
    })
}

#[bench]
fn indexed_zip_1d_ix1(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros(I2DSZ * I2DSZ);
    for (i, elt) in a.indexed_iter_mut() {
        *elt = i as _;
    }

    bench.iter(|| {
        Zip::indexed(&a).for_each(|i, &_elt| {
            black_box(i);
            //assert!(a[i] == elt);
        });
    })
}

#[bench]
fn indexed_iter_2d_ix2(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros((I2DSZ, I2DSZ));
    for ((i, j), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j) as _;
    }

    bench.iter(|| {
        for (i, &_elt) in a.indexed_iter() {
            //assert!(a[i] == elt);
            black_box(i);
        }
    })
}
#[bench]
fn indexed_zip_2d_ix2(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros((I2DSZ, I2DSZ));
    for ((i, j), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j) as _;
    }

    bench.iter(|| {
        Zip::indexed(&a).for_each(|i, &_elt| {
            black_box(i);
            //assert!(a[i] == elt);
        });
    })
}

#[bench]
fn indexed_iter_3d_ix3(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros((ISZ, ISZ, ISZ));
    for ((i, j, k), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j + 10000 * k) as _;
    }

    bench.iter(|| {
        for (i, &_elt) in a.indexed_iter() {
            //assert!(a[i] == elt);
            black_box(i);
        }
    })
}

#[bench]
fn indexed_zip_3d_ix3(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros((ISZ, ISZ, ISZ));
    for ((i, j, k), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j + 10000 * k) as _;
    }

    bench.iter(|| {
        Zip::indexed(&a).for_each(|i, &_elt| {
            black_box(i);
            //assert!(a[i] == elt);
        });
    })
}

#[bench]
fn indexed_iter_3d_dyn(bench: &mut Bencher) {
    let mut a = Array::<f64, _>::zeros((ISZ, ISZ, ISZ));
    for ((i, j, k), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j + 10000 * k) as _;
    }
    let a = a.into_shape(&[ISZ; 3][..]).unwrap();

    bench.iter(|| {
        for (i, &_elt) in a.indexed_iter() {
            //assert!(a[i] == elt);
            black_box(i);
        }
    })
}

#[bench]
fn iter_sum_1d_strided_fold(bench: &mut Bencher) {
    let mut a = Array::<u64, _>::ones(10240);
    a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    bench.iter(|| a.iter().sum::<u64>());
}

#[bench]
fn iter_sum_1d_strided_rfold(bench: &mut Bencher) {
    let mut a = Array::<u64, _>::ones(10240);
    a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    bench.iter(|| a.iter().rfold(0, |acc, &x| acc + x));
}

#[bench]
fn iter_axis_iter_sum(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((64, 64));
    bench.iter(|| a.axis_iter(Axis(0)).map(|plane| plane.sum()).sum::<f32>());
}

#[bench]
fn iter_axis_chunks_1_iter_sum(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((64, 64));
    bench.iter(|| {
        a.axis_chunks_iter(Axis(0), 1)
            .map(|plane| plane.sum())
            .sum::<f32>()
    });
}

#[bench]
fn iter_axis_chunks_5_iter_sum(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((64, 64));
    bench.iter(|| {
        a.axis_chunks_iter(Axis(0), 5)
            .map(|plane| plane.sum())
            .sum::<f32>()
    });
}

pub fn zip_mut_with(data: &Array3<f32>, out: &mut Array3<f32>) {
    out.zip_mut_with(&data, |o, &i| {
        *o = i;
    });
}

#[bench]
fn zip_mut_with_cc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros((ISZ, ISZ, ISZ));
    let mut out = Array3::zeros(data.dim());
    b.iter(|| zip_mut_with(&data, &mut out));
}

#[bench]
fn zip_mut_with_ff(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros((ISZ, ISZ, ISZ).f());
    let mut out = Array3::zeros(data.dim().f());
    b.iter(|| zip_mut_with(&data, &mut out));
}
