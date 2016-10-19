#![feature(question_mark)]

extern crate libc;

mod c_wstr;

pub mod ffi {
    pub use c_wstr::*;
}
