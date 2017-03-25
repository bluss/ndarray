
#![feature(test)]

extern crate num_cpus;
extern crate test;
use test::Bencher;

#[macro_use(s)]
extern crate ndarray;
use ndarray::prelude::*;

extern crate rayon;
use rayon::prelude::*;

const EXP_N: usize = 128;

use std::cmp::max;

fn set_threads() {
    let n = max(1, num_cpus::get() / 2);
    let cfg = rayon::Configuration::new().set_num_threads(n);
    let _ = rayon::initialize(cfg);
}

#[bench]
fn map_exp_regular(bench: &mut Bencher)
{
    let mut a = Array2::<f64>::zeros((EXP_N, EXP_N));
    a.swap_axes(0, 1);
    bench.iter(|| {
        a.mapv_inplace(|x| x.exp());
    });
}

#[bench]
fn rayon_exp_regular(bench: &mut Bencher)
{
    set_threads();
    let mut a = Array2::<f64>::zeros((EXP_N, EXP_N));
    a.swap_axes(0, 1);
    bench.iter(|| {
        a.view_mut().into_par_iter().for_each(|x| *x = x.exp());
    });
}

const FASTEXP: usize = 800;

#[inline]
fn fastexp(x: f64) -> f64 {
    let x = 1. + x/1024.;
    x.powi(1024)
}

#[bench]
fn map_fastexp_regular(bench: &mut Bencher)
{
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        a.mapv_inplace(|x| fastexp(x))
    });
}

#[bench]
fn rayon_fastexp_regular(bench: &mut Bencher)
{
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        a.view_mut().into_par_iter().for_each(|x| *x = fastexp(*x));
    });
}

#[bench]
fn map_fastexp_cut(bench: &mut Bencher)
{
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    let mut a = a.slice_mut(s![.., ..-1]);
    bench.iter(|| {
        a.mapv_inplace(|x| fastexp(x))
    });
}

#[bench]
fn rayon_fastexp_cut(bench: &mut Bencher)
{
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    let mut a = a.slice_mut(s![.., ..-1]);
    bench.iter(|| {
        a.view_mut().into_par_iter().for_each(|x| *x = fastexp(*x));
    });
}

#[bench]
fn map_fastexp_by_axis(bench: &mut Bencher)
{
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        for mut sheet in a.axis_iter_mut(Axis(0)) {
            sheet.mapv_inplace(fastexp)
        }
    });
}

#[bench]
fn rayon_fastexp_by_axis(bench: &mut Bencher)
{
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        a.axis_iter_mut(Axis(0)).into_par_iter()
            .for_each(|mut sheet| sheet.mapv_inplace(fastexp));
    });
}

#[bench]
fn par_map_inplace_fastexp(bench: &mut Bencher)
{
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        a.par_map_inplace(|x| *x = fastexp(*x));
    });
}

#[bench]
fn map_fastexp(bench: &mut Bencher)
{
    set_threads();
    let a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        a.map(|x| fastexp(*x))
    });
}

#[bench]
fn par_map_fastexp(bench: &mut Bencher)
{
    set_threads();
    let a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        a.par_map(|x| fastexp(*x))
    });
}
