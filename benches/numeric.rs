#![feature(test)]

extern crate test;
use test::Bencher;

use ndarray::prelude::*;
use ndarray::{Zip, FoldWhile};

const N: usize = 1024;
const X: usize = 64;
const Y: usize = 16;

#[bench]
fn clip(bench: &mut Bencher) {
    let mut a = Array::linspace(0., 127., N * 2)
        .into_shape([X, Y * 2])
        .unwrap();
    let min = 2.;
    let max = 5.;
    bench.iter(|| {
        a.mapv_inplace(|mut x| {
            if x < min {
                x = min
            }
            if x > max {
                x = max
            }
            x
        })
    });
}

#[bench]
fn max_early_return(bench: &mut Bencher)
{
    fn max(arr: &Array2<f64>) -> Option<&f64>
    {
        if let Some(mut max) = arr.first() {
            if let Some(slc) = arr.as_slice_memory_order() {
                for item in slc.iter().skip(1) {
                    match max.partial_cmp(item) {
                        None => return None,
                        Some(::std::cmp::Ordering::Less) => max = item,
                        _ => {},
                    }
                }
                Some(max)
            } else {
                Zip::from(arr).fold_while(Some(max), |acc, x| 
                match acc.partial_cmp(&Some(x)) {
                    None => FoldWhile::Done(None),
                    Some(::std::cmp::Ordering::Less) => FoldWhile::Continue(Some(x)),
                    _ => FoldWhile::Continue(acc),
                }).into_inner()
            }
        } else {
            None
        }
    }  
    let mut a = Array::linspace(0., 127., N * 2).into_shape([X, Y * 2]).unwrap();
    bench.iter(|| max(&a));
}

#[bench]
fn max_short(bench: &mut Bencher) {
    fn max(arr: &Array2<f64>) -> Option<&f64> {
        if let Some(first) = arr.first() {
            let max = arr.fold(first, |acc, x| if x>acc {x} else {acc});
            if max == max {
                Some(max)
            } else {
                None
            }
        } else {
            None
        }
    }
    let mut a = Array::linspace(0., 127., N * 2).into_shape([X, Y * 2]).unwrap();
    bench.iter(|| max(&a));
}