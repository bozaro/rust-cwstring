// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ascii;
use std::borrow::{Cow, ToOwned, Borrow};
use std::boxed::Box;
use std::convert::{Into, From};
use std::cmp::{PartialEq, Eq, PartialOrd, Ord, Ordering};
use std::error::Error;
use std::fmt::{self, Write, Debug};
use std::io;
use std::iter::Iterator;
use libc;
use memchr;
use std::mem;
use std::ops;
use std::option::Option::{self, Some, None};
use std::os::raw::{c_char, c_ushort};
use std::result::Result::{self, Ok, Err};
use std::slice;
use std::str::{self, Utf8Error};
use std::string::{String, FromUtf16Error, FromUtf8Error};
use std::vec::Vec;

/// A type representing an owned C-compatible string
///
/// This type serves the primary purpose of being able to safely generate a
/// C-compatible string from a Rust byte slice or vector. An instance of this
/// type is a static guarantee that the underlying bytes contain no interior 0
/// bytes and the final byte is 0.
///
/// A `CString` is created from either a byte slice or a byte vector. After
/// being created, a `CString` predominately inherits all of its methods from
/// the `Deref` implementation to `[c_ushort]`. Note that the underlying array
/// is represented as an array of `c_ushort` as opposed to `u8`. A `u8` slice
/// can be obtained with the `as_bytes` method.  Slices produced from a `CString`
/// do *not* contain the trailing nul terminator unless otherwise specified.
///
/// # Examples
///
/// ```no_run
/// extern crate cwstring;
/// # fn main() {
///
/// use cwstring::ffi::CString;
/// use std::os::raw::c_ushort;
///
/// extern {
///     fn my_printer(s: *const c_ushort);
/// }
///
/// let c_to_print = CString::<u16>::from_str("Hello, world!").unwrap();
/// unsafe {
///     my_printer(c_to_print.as_ptr());
/// }
/// # }
/// ```
///
/// # Safety
///
/// `CString` is intended for working with traditional C-style strings
/// (a sequence of non-null bytes terminated by a single null byte); the
/// primary use case for these kinds of strings is interoperating with C-like
/// code. Often you will need to transfer ownership to/from that external
/// code. It is strongly recommended that you thoroughly read through the
/// documentation of `CString` before use, as improper ownership management
/// of `CString` instances can lead to invalid memory accesses, memory leaks,
/// and other memory errors.

#[derive(PartialEq, PartialOrd, Eq, Ord, Hash, Clone)]
pub struct CString<A: CChar = u8> {
    inner: Box<[A]>,
}

/// Representation of a borrowed C string.
///
/// This dynamically sized type is only safely constructed via a borrowed
/// version of an instance of `CString`. This type can be constructed from a raw
/// C string as well and represents a C string borrowed from another location.
///
/// Note that this structure is **not** `repr(C)` and is not recommended to be
/// placed in the signatures of FFI functions. Instead safe wrappers of FFI
/// functions may leverage the unsafe `from_ptr` constructor to provide a safe
/// interface to other consumers.
///
/// # Examples
///
/// Inspecting a foreign C string
///
/// ```no_run
/// extern crate cwstring;
///
/// use cwstring::ffi::CStr;
/// use std::os::raw::c_ushort;
///
/// extern { fn my_string() -> *const c_ushort; }
///
/// fn main() {
///     unsafe {
///         let slice = CStr::<u16>::from_cptr(my_string());
///         println!("string length: {}", slice.to_bytes().len());
///     }
/// }
/// ```
///
/// Passing a Rust-originating C string
///
/// ```no_run
/// extern crate cwstring;
///
/// use cwstring::ffi::{CString, CStr};
/// use std::os::raw::c_ushort;
///
/// fn work(data: &CStr<u16>) {
///     extern { fn work_with(data: *const c_ushort); }
///
///     unsafe { work_with(data.as_ptr()) }
/// }
///
/// fn main() {
///     let s = CString::<u16>::from_str("data data data data").unwrap();
///     work(&s);
/// }
/// ```
///
/// Converting a foreign C string into a Rust `String`
///
/// ```no_run
/// extern crate cwstring;
///
/// use cwstring::ffi::CStr;
/// use std::os::raw::c_ushort;
///
/// extern { fn my_string() -> *const c_ushort; }
///
/// fn my_string_safe() -> String {
///     unsafe {
///         CStr::<u16>::from_cptr(my_string()).to_string_lossy().into_owned()
///     }
/// }
///
/// fn main() {
///     println!("string: {}", my_string_safe());
/// }
/// ```
#[derive(Hash)]
pub struct CStr<A: CChar = u8> {
    // FIXME: this should not be represented with a DST slice but rather with
    //        just a raw `c_ushort` along with some form of marker to make
    //        this an unsized type. Essentially `sizeof(&CStr)` should be the
    //        same as `sizeof(&c_ushort)` but `CStr` should be an unsized type.
    inner: [A]
}

/// An error returned from `CString::new` to indicate that a nul byte was found
/// in the vector provided.
#[derive(Clone, PartialEq, Debug)]
pub struct WideNulError<A: CChar>(usize, Vec<A>);

/// An error returned from `CString::into_string` to indicate that a UTF-8 error
/// was encountered during the conversion.
#[derive(Clone, PartialEq, Debug)]
pub struct IntoStringError<A: CChar> {
    inner: CString<A>,
    error: Utf8Error,
}

pub trait CChar: Sized + PartialEq + PartialOrd + Ord + Default + Clone + Debug {
    type CType;

    type FromError;

    fn memchr(x: Self, text: &[Self]) -> Option<usize>;

    fn vec_to_string(text: &[Self]) -> Result<String, Self::FromError>;
    
    fn vec_to_string_lossy<'a>(text: &'a [Self]) -> Cow<'a, str>;

    fn str_to_vec(text: &str) -> Vec<Self>;

    unsafe fn strlen(ptr: *const Self::CType) -> usize;

    fn write(f: &mut fmt::Formatter, text: &[Self]) -> fmt::Result;
}

impl <A: CChar> CString<A> {
    /// Creates a new C-compatible string from a container of bytes.
    ///
    /// This method will consume the provided data and use the underlying bytes
    /// to construct a new string, ensuring that there is a trailing 0 byte.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// extern crate cwstring;
    ///
    /// use cwstring::ffi::CString;
    /// use std::os::raw::c_ushort;
    ///
    /// extern { fn puts(s: *const c_ushort); }
    ///
    /// fn main() {
    ///     let to_print = CString::new("Hello!".encode_utf16().collect::<Vec<u16>>()).unwrap();
    ///     unsafe {
    ///         puts(to_print.as_ptr());
    ///     }
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// This function will return an error if the bytes yielded contain an
    /// internal 0 byte. The error returned will contain the bytes as well as
    /// the position of the nul byte.
    pub fn new<T: Into<Vec<A>>>(t: T) -> Result<Self, WideNulError<A>> {
        Self::_new(t.into())
    }

    fn _new(bytes: Vec<A>) -> Result<Self, WideNulError<A>> {
        match A::memchr(A::default(), &bytes) {
            Some(i) => Err(WideNulError(i, bytes)),
            None => Ok(unsafe { CString::from_vec_unchecked(bytes) }),
        }
    }

    /// Creates a C-compatible string from a byte vector without checking for
    /// interior 0 bytes.
    ///
    /// This method is equivalent to `new` except that no runtime assertion
    /// is made that `v` contains no 0 bytes, and it requires an actual
    /// byte vector, not anything that can be converted to one with Into.
    pub unsafe fn from_vec_unchecked(mut v: Vec<A>) -> Self {
        v.push(A::default());
        CString { inner: v.into_boxed_slice() }
    }

    /// Retakes ownership of a `CString` that was transferred to C.
    ///
    /// This should only ever be called with a pointer that was earlier
    /// obtained by calling `into_raw` on a `CString`. Additionally, the length
    /// of the string will be recalculated from the pointer.
    pub unsafe fn from_raw(ptr: *mut A::CType) -> Self {
        let len = A::strlen(ptr) + 1; // Including the NUL byte
        let slice = slice::from_raw_parts(ptr, len as usize);
        CString { inner: mem::transmute(slice) }
    }

    /// Creates a new C-compatible string from a container of bytes.
    ///
    /// This method will consume the provided data and use the underlying bytes
    /// to construct a new string, ensuring that there is a trailing 0 byte.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// extern crate cwstring;
    ///
    /// use cwstring::ffi::CString;
    /// use std::os::raw::c_ushort;
    ///
    /// extern { fn puts(s: *const c_ushort); }
    ///
    /// fn main() {
    ///     let to_print = CString::<u16>::from_str("Hello!").unwrap();
    ///     unsafe {
    ///         puts(to_print.as_ptr());
    ///     }
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// This function will return an error if the bytes yielded contain an
    /// internal 0 byte. The error returned will contain the bytes as well as
    /// the position of the nul byte.
    pub fn from_str(v: &str) -> Result<Self, WideNulError<A>> {
        Self::_new(A::str_to_vec(v))
    }

    /// Transfers ownership of the string to a C caller.
    ///
    /// The pointer must be returned to Rust and reconstituted using
    /// `from_raw` to be properly deallocated. Specifically, one
    /// should *not* use the standard C `free` function to deallocate
    /// this string.
    ///
    /// Failure to call `from_raw` will lead to a memory leak.
    pub fn into_raw(self) -> *mut c_ushort {
        Box::into_raw(self.inner) as *mut c_ushort
    }

    /// Returns the underlying byte buffer.
    ///
    /// The returned buffer does **not** contain the trailing nul separator and
    /// it is guaranteed to not have any interior nul bytes.
    pub fn into_bytes(self) -> Vec<A> {
        let mut vec = self.inner.into_vec();
        let _nul = vec.pop();
        debug_assert_eq!(_nul, Some(A::default()));
        vec
    }

    /// Equivalent to the `into_bytes` function except that the returned vector
    /// includes the trailing nul byte.
    pub fn into_bytes_with_nul(self) -> Vec<A> {
        self.inner.into_vec()
    }

    /// Returns the contents of this `CString` as a slice of bytes.
    ///
    /// The returned slice does **not** contain the trailing nul separator and
    /// it is guaranteed to not have any interior nul bytes.
    pub fn as_bytes(&self) -> &[A] {
        &self.inner[..self.inner.len() - 1]
    }

    /// Equivalent to the `as_bytes` function except that the returned slice
    /// includes the trailing nul byte.
    pub fn as_bytes_with_nul(&self) -> &[A] {
        &self.inner
    }
}

impl <A: CChar> ops::Deref for CString<A> {
    type Target = CStr<A>;

    fn deref(&self) -> &CStr<A> {
        unsafe { mem::transmute(self.as_bytes_with_nul()) }
    }
}

impl <A: CChar> fmt::Debug for CString<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}
/*
impl <A: CChar> From<CString<A>> for Vec<A> {
    fn from(s: CString<A>) -> Vec<A> {
        s.into_bytes()
    }
}
*/
impl From<CString<u8>> for Vec<u8> {
    // todo: Replace this method by generic
    fn from(s: CString<u8>) -> Vec<u8> {
        s.into_bytes()
    }
}

impl From<CString<u16>> for Vec<u16> {
    // todo: Replace this method by generic
    fn from(s: CString<u16>) -> Vec<u16> {
        s.into_bytes()
    }
}

impl <A: CChar> fmt::Debug for CStr<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\"")?;
        A::write(f, self.to_bytes())?;
        write!(f, "\"")
    }
}

impl <A: CChar> Borrow<CStr<A>> for CString<A> {
    fn borrow(&self) -> &CStr<A> { self }
}

impl <A: CChar> WideNulError<A> {
    /// Returns the position of the nul byte in the slice that was provided to
    /// `CString::new`.
    pub fn nul_position(&self) -> usize { self.0 }

    /// Consumes this error, returning the underlying vector of bytes which
    /// generated the error in the first place.
    pub fn into_vec(self) -> Vec<A> { self.1 }
}

impl <A: CChar> Error for WideNulError<A> {
    fn description(&self) -> &str { "nul byte found in data" }
}

impl <A: CChar> fmt::Display for WideNulError<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "nul byte found in provided data at position: {}", self.0)
    }
}

impl <A: CChar> From<WideNulError<A>> for io::Error {
    fn from(_: WideNulError<A>) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidInput,
                       "data provided contains a nul byte")
    }
}

impl <A: CChar> CStr<A> {
    /// Casts a raw C string to a safe C string wrapper.
    ///
    /// This function will cast the provided `ptr` to the `CStr` wrapper which
    /// allows inspection and interoperation of non-owned C strings. This method
    /// is unsafe for a number of reasons:
    ///
    /// * There is no guarantee to the validity of `ptr`
    /// * The returned lifetime is not guaranteed to be the actual lifetime of
    ///   `ptr`
    /// * There is no guarantee that the memory pointed to by `ptr` contains a
    ///   valid nul terminator byte at the end of the string.
    ///
    /// > **Note**: This operation is intended to be a 0-cost cast but it is
    /// > currently implemented with an up-front calculation of the length of
    /// > the string. This is not guaranteed to always be the case.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// extern crate cwstring;
    ///
    /// # fn main() {
    /// use cwstring::ffi::CStr;
    /// use std::os::raw::c_ushort;
    ///
    /// extern {
    ///     fn my_string() -> *const c_ushort;
    /// }
    ///
    /// unsafe {
    ///     let slice = CStr::<u16>::from_cptr(my_string());
    ///     println!("string returned: {}", slice.to_string().unwrap());
    /// }
    /// # }
    /// ```
    pub unsafe fn from_cptr<'a>(ptr: *const A::CType) -> &'a CStr<A> {
        let len = A::strlen(ptr);
        mem::transmute(slice::from_raw_parts(ptr, len + 1))
    }

    /// Creates a C string wrapper from a byte slice.
    ///
    /// This function will cast the provided `bytes` to a `CStr` wrapper after
    /// ensuring that it is null terminated and does not contain any interior
    /// nul bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(cstr_from_bytes)]
    /// extern crate cwstring;
    ///
    /// use cwstring::ffi::CStr;
    ///
    /// # fn main() {
    /// let wide = "hello\0".encode_utf16().collect::<Vec<u16>>();
    /// let cstr = CStr::from_bytes_with_nul(&wide);
    /// assert!(cstr.is_some());
    /// # }
    /// ```
    pub fn from_bytes_with_nul(bytes: &[A]) -> Option<&CStr<A>> {
        if bytes.is_empty() || A::memchr(A::default(), bytes) != Some(bytes.len() - 1) {
            None
        } else {
            Some(unsafe { Self::from_bytes_with_nul_unchecked(bytes) })
        }
    }

    /// Unsafely creates a C string wrapper from a byte slice.
    ///
    /// This function will cast the provided `bytes` to a `CStr` wrapper without
    /// performing any sanity checks. The provided slice must be null terminated
    /// and not contain any interior nul bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(cstr_from_bytes)]
    /// extern crate cwstring;
    ///
    /// use cwstring::ffi::{CStr, CString};
    ///
    /// # fn main() {
    /// unsafe {
    ///     let cstring = CString::<u16>::from_str("hello").unwrap();
    ///     let cstr = CStr::from_bytes_with_nul_unchecked(cstring.to_bytes_with_nul());
    ///     assert_eq!(cstr, &*cstring);
    /// }
    /// # }
    /// ```
    pub unsafe fn from_bytes_with_nul_unchecked(bytes: &[A]) -> &CStr<A> {
        mem::transmute(bytes)
    }

    /// Returns the inner pointer to this C string.
    ///
    /// The returned pointer will be valid for as long as `self` is and points
    /// to a contiguous region of memory terminated with a 0 byte to represent
    /// the end of the string.
    pub fn as_ptr(&self) -> *const A::CType {
        self.inner.as_ptr() as *const A::CType
    }

    /// Converts this C string to a byte slice.
    ///
    /// This function will calculate the length of this string (which normally
    /// requires a linear amount of work to be done) and then return the
    /// resulting slice of `u8` elements.
    ///
    /// The returned slice will **not** contain the trailing nul that this C
    /// string has.
    ///
    /// > **Note**: This method is currently implemented as a 0-cost cast, but
    /// > it is planned to alter its definition in the future to perform the
    /// > length calculation whenever this method is called.
    pub fn to_bytes(&self) -> &[A] {
        let bytes = self.to_bytes_with_nul();
        &bytes[..bytes.len() - 1]
    }

    /// Converts this C string to a byte slice containing the trailing 0 byte.
    ///
    /// This function is the equivalent of `to_bytes` except that it will retain
    /// the trailing nul instead of chopping it off.
    ///
    /// > **Note**: This method is currently implemented as a 0-cost cast, but
    /// > it is planned to alter its definition in the future to perform the
    /// > length calculation whenever this method is called.
    pub fn to_bytes_with_nul(&self) -> &[A] {
        unsafe { mem::transmute(&self.inner) }
    }

    /// Yields a `&str` slice if the `CStr` contains valid UTF-8.
    ///
    /// This function will calculate the length of this string and check for
    /// UTF-8 validity, and then return the `&str` if it's valid.
    ///
    /// > **Note**: This method is currently implemented to check for validity
    /// > after a 0-cost cast, but it is planned to alter its definition in the
    /// > future to perform the length calculation in addition to the UTF-8
    /// > check whenever this method is called.
    pub fn to_string(&self) -> Result<String, A::FromError> {
        // NB: When CStr is changed to perform the length check in .to_bytes()
        // instead of in from_ptr(), it may be worth considering if this should
        // be rewritten to do the UTF-8 check inline with the length calculation
        // instead of doing it afterwards.
        A::vec_to_string(self.to_bytes())
    }

    /// Converts a `CStr` into a `Cow<str>`.
    ///
    /// This function will calculate the length of this string (which normally
    /// requires a linear amount of work to be done) and then return the
    /// resulting slice as a `Cow<str>`, replacing any invalid UTF-8 sequences
    /// with `U+FFFD REPLACEMENT CHARACTER`.
    ///
    /// > **Note**: This method is currently implemented to check for validity
    /// > after a 0-cost cast, but it is planned to alter its definition in the
    /// > future to perform the length calculation in addition to the UTF-8
    /// > check whenever this method is called.
    pub fn to_string_lossy<'a>(&'a self) -> Cow<'a, str> {
        A::vec_to_string_lossy(self.to_bytes())
    }
}

impl CStr<u8> {
    pub fn to_str(&self) -> Result<&str, str::Utf8Error> {
        str::from_utf8(self.to_bytes())
    }

    pub unsafe fn from_ptr<'a>(ptr: *const c_char) -> &'a Self {
        CStr::from_cptr(ptr)
    }
}

impl <A: CChar> PartialEq for CStr<A> {
    fn eq(&self, other: &CStr<A>) -> bool {
        self.to_bytes().eq(other.to_bytes())
    }
}

impl <A: CChar> Eq for CStr<A> {}
impl <A: CChar> PartialOrd for CStr<A> {
    fn partial_cmp(&self, other: &CStr<A>) -> Option<Ordering> {
        self.to_bytes().partial_cmp(&other.to_bytes())
    }
}
impl <A: CChar> Ord for CStr<A> {
    fn cmp(&self, other: &CStr<A>) -> Ordering {
        self.to_bytes().cmp(&other.to_bytes())
    }
}

impl <A: CChar> ToOwned for CStr<A> {
    type Owned = CString<A>;

    fn to_owned(&self) -> CString<A> {
        unsafe { CString::from_vec_unchecked(self.to_bytes().to_vec()) }
    }
}

impl <'a, A: CChar> From<&'a CStr<A>> for CString<A> {
    fn from(s: &'a CStr<A>) -> CString<A> {
        s.to_owned()
    }
}

impl <A: CChar> ops::Index<ops::RangeFull> for CString<A> {
    type Output = CStr<A>;

    #[inline]
    fn index(&self, _index: ops::RangeFull) -> &CStr<A> {
        self
    }
}

impl <A: CChar> AsRef<CStr<A>> for CStr<A> {
    fn as_ref(&self) -> &CStr<A> {
        self
    }
}

impl <A: CChar> AsRef<CStr<A>> for CString<A> {
    fn as_ref(&self) -> &CStr<A> {
        self
    }
}

impl CChar for u8 {
    type CType = c_char;
    type FromError = FromUtf8Error;

    #[inline]
    fn memchr(x: Self, text: &[Self]) -> Option<usize> {
        memchr::memchr(x, text)
    }

    #[inline]
    unsafe fn strlen(ptr: *const c_char) -> usize {
        libc::strlen(ptr) as usize
    }

    #[inline]
    fn str_to_vec(text: &str) -> Vec<Self> {
        text.as_bytes().to_vec()
    }

    #[inline]
    fn vec_to_string(text: &[Self]) -> Result<String, FromUtf8Error> {
        String::from_utf8(text.to_vec())
    }
    
    #[inline]
    fn vec_to_string_lossy<'a>(text: &'a [Self]) -> Cow<'a, str> {
        String::from_utf8_lossy(text)
    }

    fn write(f: &mut fmt::Formatter, text: &[Self]) -> fmt::Result {
        for byte in text.iter().flat_map(|b| ascii::escape_default(*b)) {
            f.write_char(byte as char)?;
        }
        Ok(())
    }
}

impl CChar for u16 {
    type CType = c_ushort;
    type FromError = FromUtf16Error;

    #[inline]
    fn memchr(x: Self, text: &[Self]) -> Option<usize> {
        for i in 0..text.len() {
            if text[i] == x {
                return Some(i);
            }
        }
        None
    }

    #[inline]
    unsafe fn strlen(ptr: *const c_ushort) -> usize {
        let mut len = 0;
        loop {
            if *ptr.offset(len as isize) == 0u16 {
                break;
            }
            len += 1;
        }
        len
    }

    #[inline]
    fn str_to_vec(text: &str) -> Vec<Self> {
        text.encode_utf16().collect()
    }

    #[inline]
    fn vec_to_string(text: &[Self]) -> Result<String, FromUtf16Error> {
        String::from_utf16(text)
    }
    
    #[inline]
    fn vec_to_string_lossy<'a>(text: &'a [Self]) -> Cow<'a, str> {
        Cow::Owned(String::from_utf16_lossy(text))
    }

    fn write(f: &mut fmt::Formatter, text: &[Self]) -> fmt::Result {
        use std;
        for byte in text.iter().flat_map(|b| unsafe {std::char::from_u32_unchecked(*b as u32)}.escape_default()) {
            f.write_char(byte as char)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests_u8 {
    use std::prelude::v1::*;
    use super::*;
    use std::os::raw::c_char;
    use std::borrow::Cow::{Borrowed, Owned};
    use std::hash::{SipHasher, Hash, Hasher};

    #[test]
    fn c_to_rust() {
        let data = b"123\0";
        let ptr = data.as_ptr() as *const c_char;
        unsafe {
            assert_eq!(CStr::from_ptr(ptr).to_bytes(), b"123");
            assert_eq!(CStr::from_ptr(ptr).to_bytes_with_nul(), b"123\0");
        }
    }

    #[test]
    fn simple() {
        let s = CString::new("1234").unwrap();
        assert_eq!(s.as_bytes(), b"1234");
        assert_eq!(s.as_bytes_with_nul(), b"1234\0");
    }

    #[test]
    fn build_with_zero1() {
        assert!(CString::new(&b"\0"[..]).is_err());
    }
    #[test]
    fn build_with_zero2() {
        assert!(CString::new(vec![0 as u8]).is_err());
    }

    #[test]
    fn build_with_zero3() {
        unsafe {
            let s = CString::from_vec_unchecked(vec![0]);
            assert_eq!(s.as_bytes(), b"\0");
        }
    }

    #[test]
    fn formatted() {
        let s = CString::new(&b"abc\x01\x02\n\xE2\x80\xA6\xFF"[..]).unwrap();
        assert_eq!(format!("{:?}", s), r#""abc\x01\x02\n\xe2\x80\xa6\xff""#);
    }

    #[test]
    fn borrowed() {
        unsafe {
            let s = CStr::from_ptr(b"12\0".as_ptr() as *const _);
            assert_eq!(s.to_bytes(), b"12");
            assert_eq!(s.to_bytes_with_nul(), b"12\0");
        }
    }

    #[test]
    fn to_str() {
        let data = b"123\xE2\x80\xA6\0";
        let ptr = data.as_ptr() as *const c_char;
        unsafe {
            assert_eq!(CStr::from_ptr(ptr).to_str(), Ok("123…"));
            assert_eq!(CStr::from_ptr(ptr).to_string_lossy(), Borrowed("123…"));
        }
        let data = b"123\xE2\0";
        let ptr = data.as_ptr() as *const c_char;
        unsafe {
            assert!(CStr::from_ptr(ptr).to_str().is_err());
            assert_eq!(CStr::from_ptr(ptr).to_string_lossy(), Owned::<str>(format!("123\u{FFFD}")));
        }
    }

    #[test]
    fn to_owned() {
        let data = b"123\0";
        let ptr = data.as_ptr() as *const c_char;

        let owned = unsafe { CStr::from_ptr(ptr).to_owned() };
        assert_eq!(owned.as_bytes_with_nul(), data);
    }

    #[test]
    fn equal_hash() {
        let data = b"123\xE2\xFA\xA6\0";
        let ptr = data.as_ptr() as *const c_char;
        let cstr: &'static CStr = unsafe { CStr::from_ptr(ptr) };

        let mut s = SipHasher::new_with_keys(0, 0);
        cstr.hash(&mut s);
        let cstr_hash = s.finish();
        let mut s = SipHasher::new_with_keys(0, 0);
        CString::new(&data[..data.len() - 1]).unwrap().hash(&mut s);
        let cstring_hash = s.finish();

        assert_eq!(cstr_hash, cstring_hash);
    }

    #[test]
    fn from_bytes_with_nul() {
        let data = b"123\0";
        let cstr = CStr::from_bytes_with_nul(data);
        assert_eq!(cstr.map(CStr::to_bytes), Some(&b"123"[..]));
        assert_eq!(cstr.map(CStr::to_bytes_with_nul), Some(&b"123\0"[..]));

        unsafe {
            let cstr_unchecked = CStr::from_bytes_with_nul_unchecked(data);
            assert_eq!(cstr, Some(cstr_unchecked));
        }
    }

    #[test]
    fn from_bytes_with_nul_unterminated() {
        let data = b"123";
        let cstr = CStr::from_bytes_with_nul(data);
        assert!(cstr.is_none());
    }

    #[test]
    fn from_bytes_with_nul_interior() {
        let data = b"1\023\0";
        let cstr = CStr::from_bytes_with_nul(data);
        assert!(cstr.is_none());
    }
}

#[cfg(test)]
mod tests_u16 {
    use std::prelude::v1::*;
    use super::*;
    use std::os::raw::c_ushort;
    use std::borrow::Cow::{Borrowed, Owned};
    use std::hash::{SipHasher, Hash, Hasher};

    fn utf16(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn c_to_rust() {
        let data = utf16("123\0");
        let ptr = data.as_ptr();
        unsafe {
            assert_eq!(CStr::<u16>::from_cptr(ptr).to_bytes(), &utf16("123")[..]);
            assert_eq!(CStr::<u16>::from_cptr(ptr).to_bytes_with_nul(), &utf16("123\0")[..]);
        }
    }

    #[test]
    fn simple() {
        let s = CString::<u16>::from_str("1234").unwrap();
        assert_eq!(s.as_bytes(), &utf16("1234")[..]);
        assert_eq!(s.as_bytes_with_nul(), &utf16("1234\0")[..]);
    }

    #[test]
    fn build_with_zero1() {
        assert!(CString::<u16>::from_str("\0").is_err());
    }
    #[test]
    fn build_with_zero2() {
        assert!(CString::new(vec![0 as u16]).is_err());
    }

    #[test]
    fn build_with_zero3() {
        unsafe {
            let s = CString::from_vec_unchecked(vec![0]);
            assert_eq!(s.as_bytes(), &[0u16]);
        }
    }

    #[test]
    fn formatted() {
        let s = CString::new(&['a' as u16, 'b' as u16, 'c' as u16, 0x01u16, 0x02u16, '\n' as u16, 0xD83Du16, 0xDE03u16][..]).unwrap();
        assert_eq!(format!("{:?}", s), r#""abc\u{1}\u{2}\n\u{d83d}\u{de03}""#);
    }

    #[test]
    fn borrowed() {
        unsafe {
            let s = CStr::<u16>::from_cptr(utf16("12\0").as_ptr() as *const _);
            assert_eq!(s.to_bytes(), &utf16("12")[..]);
            assert_eq!(s.to_bytes_with_nul(), &utf16("12\0")[..]);
        }
    }

    #[test]
    fn to_string() {
        let data: Vec<u16> = "123".encode_utf16().chain(Some(0xD83Du16)).chain(Some(0xDE03u16)).chain(Some(0)).collect();
        let ptr = data.as_ptr();
        unsafe {
            assert!(CStr::<u16>::from_cptr(ptr).to_string().is_ok());
            assert_eq!(CStr::<u16>::from_cptr(ptr).to_string().unwrap(), "123😃");
            assert_eq!(CStr::<u16>::from_cptr(ptr).to_string_lossy(), Borrowed("123😃"));
        }
        let data: Vec<u16> = "123".encode_utf16().chain(Some(0xD83Du16)).chain(Some(0)).collect();
        let ptr = data.as_ptr();
        unsafe {
            assert!(CStr::<u16>::from_cptr(ptr).to_string().is_err());
            assert_eq!(CStr::<u16>::from_cptr(ptr).to_string_lossy(), Owned::<str>(format!("123\u{FFFD}")));
        }
    }

    #[test]
    fn from_ptr() {
        let data = utf16("123\0");
        let ptr = data.as_ptr();

        let owned = unsafe { CStr::<u16>::from_cptr(ptr).to_owned() };
        assert_eq!(owned.as_bytes_with_nul(), &data[..]);
    }

    #[test]
    fn equal_hash() {
        let data: Vec<u16> = "123".encode_utf16().chain(Some(0xD83Du16)).chain(Some(0xDE03u16)).chain(Some(0)).collect();
        let ptr = data.as_ptr();
        let cstr: &'static CStr<u16> = unsafe { CStr::<u16>::from_cptr(ptr) };

        let mut s = SipHasher::new_with_keys(0, 0);
        cstr.hash(&mut s);
        let cstr_hash = s.finish();
        let mut s = SipHasher::new_with_keys(0, 0);
        CString::new(&data[..data.len() - 1]).unwrap().hash(&mut s);
        let cstring_hash = s.finish();

        assert_eq!(cstr_hash, cstring_hash);
    }

    #[test]
    fn from_bytes_with_nul() {
        let data = utf16("123\0");
        let cstr = CStr::from_bytes_with_nul(&data);
        assert_eq!(cstr.map(CStr::to_bytes), Some(&utf16("123")[..]));
        assert_eq!(cstr.map(CStr::to_bytes_with_nul), Some(&utf16("123\0")[..]));

        unsafe {
            let cstr_unchecked = CStr::from_bytes_with_nul_unchecked(&data);
            assert_eq!(cstr, Some(cstr_unchecked));
        }
    }

    #[test]
    fn from_bytes_with_nul_unterminated() {
        let data = utf16("123");
        let cstr = CStr::from_bytes_with_nul(&data);
        assert!(cstr.is_none());
    }

    #[test]
    fn from_bytes_with_nul_interior() {
        let data = utf16("1\023\0");
        let cstr = CStr::from_bytes_with_nul(&data);
        assert!(cstr.is_none());
    }
}
