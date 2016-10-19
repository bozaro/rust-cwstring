#![feature(question_mark)]

extern crate libc;
extern crate memchr;

mod c_wstr;

pub mod ffi {
  pub use c_wstr::*;
}
