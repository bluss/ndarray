// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_use]
mod macros;

mod axis;
pub(crate) mod base;
mod chunks;
mod into_iter;
pub mod iter;
mod lanes;
mod trusted;
mod windows;

pub use self::axis::{AxisIter, AxisIterMut, AxisChunksIter, AxisChunksIterMut};
pub use self::base::{Iter, IterMut, IndexedIter, IndexedIterMut};
pub use self::chunks::{ExactChunks, ExactChunksIter, ExactChunksIterMut, ExactChunksMut};
pub use self::lanes::{Lanes, LanesMut, LanesIter, LanesIterMut};
pub use self::windows::Windows;
pub use self::into_iter::IntoIter;
pub(crate) use self::base::{Baseiter, ElementsBase, ElementsBaseMut};
pub(crate) use self::trusted::{TrustedIterator, to_vec, to_vec_mapped};

send_sync_read_only!(Iter);
send_sync_read_only!(IndexedIter);
send_sync_read_only!(LanesIter);
send_sync_read_only!(AxisIter);
send_sync_read_only!(AxisChunksIter);
send_sync_read_only!(ElementsBase);

send_sync_read_write!(IterMut);
send_sync_read_write!(IndexedIterMut);
send_sync_read_write!(LanesIterMut);
send_sync_read_write!(AxisIterMut);
send_sync_read_write!(AxisChunksIterMut);
send_sync_read_write!(ElementsBaseMut);
