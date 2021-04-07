mod layoutfmt;

// Layout it a bitset used for internal layout description of
// arrays, producers and sets of producers.
// The type is public but users don't interact with it.
#[doc(hidden)]
/// Memory layout description
#[derive(Copy, Clone)]
pub struct Layout(u32);

impl Layout {
    #[inline(always)]
    pub(crate) fn is(self, flag: u32) -> bool {
        self.0 & flag != 0
    }

    /// Return layout common to both inputs
    #[inline(always)]
    pub(crate) fn intersect(self, other: Layout) -> Layout {
        Layout(self.0 & other.0)
    }

    /// Return a layout that simultaneously "is" what both of the inputs are
    #[inline(always)]
    pub(crate) fn also(self, other: Layout) -> Layout {
        Layout(self.0 | other.0)
    }

    #[inline(always)]
    pub(crate) fn one_dimensional() -> Layout {
        Layout::c().also(Layout::f())
    }

    #[inline(always)]
    pub(crate) fn c() -> Layout {
        Layout(CORDER | CPREFER)
    }

    #[inline(always)]
    pub(crate) fn f() -> Layout {
        Layout(FORDER | FPREFER)
    }

    #[inline(always)]
    pub(crate) fn cpref() -> Layout {
        Layout(CPREFER)
    }

    #[inline(always)]
    pub(crate) fn fpref() -> Layout {
        Layout(FPREFER)
    }

    #[inline(always)]
    pub(crate) fn none() -> Layout {
        Layout(0)
    }

    /// A simple "score" method which scores positive for preferring C-order, negative for F-order
    /// Subject to change when we can describe other layouts
    pub(crate) fn tendency(self) -> i32 {
        (self.is(CORDER) as i32 - self.is(FORDER) as i32) +
        (self.is(CPREFER) as i32 - self.is(FPREFER) as i32)

    }
}

pub const CORDER: u32 = 0b01;
pub const FORDER: u32 = 0b10;
pub const CPREFER: u32 = 0b0100;
pub const FPREFER: u32 = 0b1000;


#[cfg(test)]
mod tests {
    use super::*;
    use crate::imp_prelude::*;
    use crate::NdProducer;

    type M = Array2<f32>;
    type M1 = Array1<f32>;
    type M0 = Array0<f32>;

    macro_rules! assert_layouts {
        ($mat:expr, $($layout:expr),*) => {{
            let layout = $mat.view().layout();
            $(
            assert!(layout.is($layout), "Assertion failed: array {:?} is not layout {}",
                $mat,
                stringify!($layout));
            )*
        }}
    }

    macro_rules! assert_not_layouts {
        ($mat:expr, $($layout:expr),*) => {{
            let layout = $mat.view().layout();
            $(
            assert!(!layout.is($layout), "Assertion failed: array {:?} show not have layout {}",
                $mat,
                stringify!($layout));
            )*
        }}
    }

    #[test]
    fn contig_layouts() {
        let a = M::zeros((5, 5));
        let b = M::zeros((5, 5).f());
        let ac = a.view().layout();
        let af = b.view().layout();
        assert!(ac.is(CORDER) && ac.is(CPREFER));
        assert!(!ac.is(FORDER) && !ac.is(FPREFER));
        assert!(!af.is(CORDER) && !af.is(CPREFER));
        assert!(af.is(FORDER) && af.is(FPREFER));
    }

    #[test]
    fn contig_cf_layouts() {
        let a = M::zeros((5, 1));
        let b = M::zeros((1, 5).f());
        assert_layouts!(a, CORDER, CPREFER, FORDER, FPREFER);
        assert_layouts!(b, CORDER, CPREFER, FORDER, FPREFER);

        let a = M1::zeros(5);
        let b = M1::zeros(5.f());
        assert_layouts!(a, CORDER, CPREFER, FORDER, FPREFER);
        assert_layouts!(b, CORDER, CPREFER, FORDER, FPREFER);

        let a = M0::zeros(());
        assert_layouts!(a, CORDER, CPREFER, FORDER, FPREFER);

        let a = M::zeros((5, 5));
        let b = M::zeros((5, 5).f());
        let arow = a.slice(s![..1, ..]);
        let bcol = b.slice(s![.., ..1]);
        assert_layouts!(arow, CORDER, CPREFER, FORDER, FPREFER);
        assert_layouts!(bcol, CORDER, CPREFER, FORDER, FPREFER);

        let acol = a.slice(s![.., ..1]);
        let brow = b.slice(s![..1, ..]);
        assert_not_layouts!(acol, CORDER, CPREFER, FORDER, FPREFER);
        assert_not_layouts!(brow, CORDER, CPREFER, FORDER, FPREFER);
    }

    #[test]
    fn stride_layouts() {
        let a = M::zeros((5, 5));

        {
            let v1 = a.slice(s![1.., ..]).layout();
            let v2 = a.slice(s![.., 1..]).layout();

            assert!(v1.is(CORDER) && v1.is(CPREFER));
            assert!(!v1.is(FORDER) && !v1.is(FPREFER));
            assert!(!v2.is(CORDER) && v2.is(CPREFER));
            assert!(!v2.is(FORDER) && !v2.is(FPREFER));
        }

        let b = M::zeros((5, 5).f());

        {
            let v1 = b.slice(s![1.., ..]).layout();
            let v2 = b.slice(s![.., 1..]).layout();

            assert!(!v1.is(CORDER) && !v1.is(CPREFER));
            assert!(!v1.is(FORDER) && v1.is(FPREFER));
            assert!(!v2.is(CORDER) && !v2.is(CPREFER));
            assert!(v2.is(FORDER) && v2.is(FPREFER));
        }
    }

    #[test]
    fn skip_layouts() {
        let a = M::zeros((5, 5));
        {
            let v1 = a.slice(s![..;2, ..]).layout();
            let v2 = a.slice(s![.., ..;2]).layout();

            assert!(!v1.is(CORDER) && v1.is(CPREFER));
            assert!(!v1.is(FORDER) && !v1.is(FPREFER));
            assert!(!v2.is(CORDER) && !v2.is(CPREFER));
            assert!(!v2.is(FORDER) && !v2.is(FPREFER));
        }

        let b = M::zeros((5, 5).f());
        {
            let v1 = b.slice(s![..;2, ..]).layout();
            let v2 = b.slice(s![.., ..;2]).layout();

            assert!(!v1.is(CORDER) && !v1.is(CPREFER));
            assert!(!v1.is(FORDER) && !v1.is(FPREFER));
            assert!(!v2.is(CORDER) && !v2.is(CPREFER));
            assert!(!v2.is(FORDER) && v2.is(FPREFER));
        }
    }
}
