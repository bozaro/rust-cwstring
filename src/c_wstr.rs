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
use std::fmt::{self, Write};
use std::io;
use std::iter::Iterator;
use libc;
use std::mem;
use std::ops;
use std::option::Option::{self, Some, None};
use std::os::raw::c_ushort;
use std::result::Result::{self, Ok, Err};
use std::slice;
use std::str::{self, Utf8Error};
use std::string::{String, FromUtf16Error};
use std::vec::Vec;

/// A type representing an owned C-compatible string
///
/// This type serves the primary purpose of being able to safely generate a
/// C-compatible string from a Rust byte slice or vector. An instance of this
/// type is a static guarantee that the underlying bytes contain no interior 0
/// bytes and the final byte is 0.
///
/// A `CWideString` is created from either a byte slice or a byte vector. After
/// being created, a `CWideString` predominately inherits all of its methods from
/// the `Deref` implementation to `[c_ushort]`. Note that the underlying array
/// is represented as an array of `c_ushort` as opposed to `u8`. A `u8` slice
/// can be obtained with the `as_bytes` method.  Slices produced from a `CWideString`
/// do *not* contain the trailing nul terminator unless otherwise specified.
///
/// # Examples
///
/// ```no_run
/// extern crate cwstring;
/// # fn main() {
///
/// use cwstring::ffi::CWideString;
/// use std::os::raw::c_ushort;
///
/// extern {
///     fn my_printer(s: *const c_ushort);
/// }
///
/// let c_to_print = CWideString::from_str("Hello, world!").unwrap();
/// unsafe {
///     my_printer(c_to_print.as_ptr());
/// }
/// # }
/// ```
///
/// # Safety
///
/// `CWideString` is intended for working with traditional C-style strings
/// (a sequence of non-null bytes terminated by a single null byte); the
/// primary use case for these kinds of strings is interoperating with C-like
/// code. Often you will need to transfer ownership to/from that external
/// code. It is strongly recommended that you thoroughly read through the
/// documentation of `CWideString` before use, as improper ownership management
/// of `CWideString` instances can lead to invalid memory accesses, memory leaks,
/// and other memory errors.

#[derive(PartialEq, PartialOrd, Eq, Ord, Hash, Clone)]
pub struct CWideString {
    inner: Box<[u16]>,
}

/// Representation of a borrowed C string.
///
/// This dynamically sized type is only safely constructed via a borrowed
/// version of an instance of `CWideString`. This type can be constructed from a raw
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
/// use cwstring::ffi::CWideStr;
/// use std::os::raw::c_ushort;
///
/// extern { fn my_string() -> *const c_ushort; }
///
/// fn main() {
///     unsafe {
///         let slice = CWideStr::from_ptr(my_string());
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
/// use cwstring::ffi::{CWideString, CWideStr};
/// use std::os::raw::c_ushort;
///
/// fn work(data: &CWideStr) {
///     extern { fn work_with(data: *const c_ushort); }
///
///     unsafe { work_with(data.as_ptr()) }
/// }
///
/// fn main() {
///     let s = CWideString::from_str("data data data data").unwrap();
///     work(&s);
/// }
/// ```
///
/// Converting a foreign C string into a Rust `String`
///
/// ```no_run
/// extern crate cwstring;
///
/// use cwstring::ffi::CWideStr;
/// use std::os::raw::c_ushort;
///
/// extern { fn my_string() -> *const c_ushort; }
///
/// fn my_string_safe() -> String {
///     unsafe {
///         CWideStr::from_ptr(my_string()).to_string_lossy()
///     }
/// }
///
/// fn main() {
///     println!("string: {}", my_string_safe());
/// }
/// ```
#[derive(Hash)]
pub struct CWideStr {
    // FIXME: this should not be represented with a DST slice but rather with
    //        just a raw `c_ushort` along with some form of marker to make
    //        this an unsized type. Essentially `sizeof(&CWideStr)` should be the
    //        same as `sizeof(&c_ushort)` but `CWideStr` should be an unsized type.
    inner: [c_ushort]
}

/// An error returned from `CWideString::new` to indicate that a nul byte was found
/// in the vector provided.
#[derive(Clone, PartialEq, Debug)]
pub struct WideNulError(usize, Vec<u16>);

/// An error returned from `CWideString::into_string` to indicate that a UTF-8 error
/// was encountered during the conversion.
#[derive(Clone, PartialEq, Debug)]
pub struct IntoStringError {
    inner: CWideString,
    error: Utf8Error,
}

impl CWideString {
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
    /// use cwstring::ffi::CWideString;
    /// use std::os::raw::c_ushort;
    ///
    /// extern { fn puts(s: *const c_ushort); }
    ///
    /// fn main() {
    ///     let to_print = CWideString::new("Hello!".encode_utf16().collect::<Vec<u16>>()).unwrap();
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
    pub fn new<T: Into<Vec<u16>>>(t: T) -> Result<CWideString, WideNulError> {
        Self::_new(t.into())
    }

    fn _new(bytes: Vec<u16>) -> Result<CWideString, WideNulError> {
        match wmemchr(0, &bytes) {
            Some(i) => Err(WideNulError(i, bytes)),
            None => Ok(unsafe { CWideString::from_vec_unchecked(bytes) }),
        }
    }

    /// Creates a C-compatible string from a byte vector without checking for
    /// interior 0 bytes.
    ///
    /// This method is equivalent to `new` except that no runtime assertion
    /// is made that `v` contains no 0 bytes, and it requires an actual
    /// byte vector, not anything that can be converted to one with Into.
    pub unsafe fn from_vec_unchecked(mut v: Vec<u16>) -> CWideString {
        v.push(0);
        CWideString { inner: v.into_boxed_slice() }
    }

    /// Retakes ownership of a `CWideString` that was transferred to C.
    ///
    /// This should only ever be called with a pointer that was earlier
    /// obtained by calling `into_raw` on a `CWideString`. Additionally, the length
    /// of the string will be recalculated from the pointer.
    pub unsafe fn from_raw(ptr: *mut c_ushort) -> CWideString {
        let len = wstrlen(ptr) + 1; // Including the NUL byte
        let slice = slice::from_raw_parts(ptr, len as usize);
        CWideString { inner: mem::transmute(slice) }
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
    /// use cwstring::ffi::CWideString;
    /// use std::os::raw::c_ushort;
    ///
    /// extern { fn puts(s: *const c_ushort); }
    ///
    /// fn main() {
    ///     let to_print = CWideString::from_str("Hello!").unwrap();
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
    pub fn from_str(v: &str) -> Result<CWideString, WideNulError> {
        Self::_new(v.encode_utf16().collect())
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
    pub fn into_bytes(self) -> Vec<u16> {
        let mut vec = self.inner.into_vec();
        let _nul = vec.pop();
        debug_assert_eq!(_nul, Some(0u16));
        vec
    }

    /// Equivalent to the `into_bytes` function except that the returned vector
    /// includes the trailing nul byte.
    pub fn into_bytes_with_nul(self) -> Vec<u16> {
        self.inner.into_vec()
    }

    /// Returns the contents of this `CWideString` as a slice of bytes.
    ///
    /// The returned slice does **not** contain the trailing nul separator and
    /// it is guaranteed to not have any interior nul bytes.
    pub fn as_bytes(&self) -> &[u16] {
        &self.inner[..self.inner.len() - 1]
    }

    /// Equivalent to the `as_bytes` function except that the returned slice
    /// includes the trailing nul byte.
    pub fn as_bytes_with_nul(&self) -> &[u16] {
        &self.inner
    }
}

impl ops::Deref for CWideString {
    type Target = CWideStr;

    fn deref(&self) -> &CWideStr {
        unsafe { mem::transmute(self.as_bytes_with_nul()) }
    }
}

impl fmt::Debug for CWideString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl From<CWideString> for Vec<u16> {
    fn from(s: CWideString) -> Vec<u16> {
        s.into_bytes()
    }
}

impl fmt::Debug for CWideStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\"")?;
        /*todo: for byte in self.to_bytes().iter().flat_map(|&b| ascii::escape_default(b)) {
            f.write_char(byte as char)?;
        }*/
        write!(f, "\"")
    }
}

impl Borrow<CWideStr> for CWideString {
    fn borrow(&self) -> &CWideStr { self }
}

impl WideNulError {
    /// Returns the position of the nul byte in the slice that was provided to
    /// `CWideString::new`.
    pub fn nul_position(&self) -> usize { self.0 }

    /// Consumes this error, returning the underlying vector of bytes which
    /// generated the error in the first place.
    pub fn into_vec(self) -> Vec<u16> { self.1 }
}

impl Error for WideNulError {
    fn description(&self) -> &str { "nul byte found in data" }
}

impl fmt::Display for WideNulError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "nul byte found in provided data at position: {}", self.0)
    }
}

impl From<WideNulError> for io::Error {
    fn from(_: WideNulError) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidInput,
                       "data provided contains a nul byte")
    }
}

impl CWideStr {
    /// Casts a raw C string to a safe C string wrapper.
    ///
    /// This function will cast the provided `ptr` to the `CWideStr` wrapper which
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
    /// use cwstring::ffi::CWideStr;
    /// use std::os::raw::c_ushort;
    ///
    /// extern {
    ///     fn my_string() -> *const c_ushort;
    /// }
    ///
    /// unsafe {
    ///     let slice = CWideStr::from_ptr(my_string());
    ///     println!("string returned: {}", slice.to_string().unwrap());
    /// }
    /// # }
    /// ```
    pub unsafe fn from_ptr<'a>(ptr: *const c_ushort) -> &'a CWideStr {
        let len = wstrlen(ptr);
        mem::transmute(slice::from_raw_parts(ptr, len as usize + 1))
    }

    /// Creates a C string wrapper from a byte slice.
    ///
    /// This function will cast the provided `bytes` to a `CWideStr` wrapper after
    /// ensuring that it is null terminated and does not contain any interior
    /// nul bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(cstr_from_bytes)]
    /// extern crate cwstring;
    ///
    /// use cwstring::ffi::CWideStr;
    ///
    /// # fn main() {
    /// let wide = "hello\0".encode_utf16().collect::<Vec<u16>>();
    /// let cstr = CWideStr::from_bytes_with_nul(&wide);
    /// assert!(cstr.is_some());
    /// # }
    /// ```
    pub fn from_bytes_with_nul(bytes: &[u16]) -> Option<&CWideStr> {
        if bytes.is_empty() || wmemchr(0, &bytes) != Some(bytes.len() - 1) {
            None
        } else {
            Some(unsafe { Self::from_bytes_with_nul_unchecked(bytes) })
        }
    }

    /// Unsafely creates a C string wrapper from a byte slice.
    ///
    /// This function will cast the provided `bytes` to a `CWideStr` wrapper without
    /// performing any sanity checks. The provided slice must be null terminated
    /// and not contain any interior nul bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(cstr_from_bytes)]
    /// extern crate cwstring;
    ///
    /// use cwstring::ffi::{CWideStr, CWideString};
    ///
    /// # fn main() {
    /// unsafe {
    ///     let cstring = CWideString::from_str("hello").unwrap();
    ///     let cstr = CWideStr::from_bytes_with_nul_unchecked(cstring.to_bytes_with_nul());
    ///     assert_eq!(cstr, &*cstring);
    /// }
    /// # }
    /// ```
    pub unsafe fn from_bytes_with_nul_unchecked(bytes: &[u16]) -> &CWideStr {
        mem::transmute(bytes)
    }

    /// Returns the inner pointer to this C string.
    ///
    /// The returned pointer will be valid for as long as `self` is and points
    /// to a contiguous region of memory terminated with a 0 byte to represent
    /// the end of the string.
    pub fn as_ptr(&self) -> *const u16 {
        self.inner.as_ptr()
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
    pub fn to_bytes(&self) -> &[u16] {
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
    pub fn to_bytes_with_nul(&self) -> &[u16] {
        unsafe { mem::transmute(&self.inner) }
    }

    /// Yields a `&str` slice if the `CWideStr` contains valid UTF-8.
    ///
    /// This function will calculate the length of this string and check for
    /// UTF-8 validity, and then return the `&str` if it's valid.
    ///
    /// > **Note**: This method is currently implemented to check for validity
    /// > after a 0-cost cast, but it is planned to alter its definition in the
    /// > future to perform the length calculation in addition to the UTF-8
    /// > check whenever this method is called.
    pub fn to_string(&self) -> Result<String, FromUtf16Error> {
        // NB: When CWideStr is changed to perform the length check in .to_bytes()
        // instead of in from_ptr(), it may be worth considering if this should
        // be rewritten to do the UTF-8 check inline with the length calculation
        // instead of doing it afterwards.
        String::from_utf16(self.to_bytes())
    }

    /// Converts a `CWideStr` into a `Cow<str>`.
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
    pub fn to_string_lossy(&self) -> String {
        String::from_utf16_lossy(self.to_bytes())
    }
}

impl PartialEq for CWideStr {
    fn eq(&self, other: &CWideStr) -> bool {
        self.to_bytes().eq(other.to_bytes())
    }
}
impl Eq for CWideStr {}
impl PartialOrd for CWideStr {
    fn partial_cmp(&self, other: &CWideStr) -> Option<Ordering> {
        self.to_bytes().partial_cmp(&other.to_bytes())
    }
}
impl Ord for CWideStr {
    fn cmp(&self, other: &CWideStr) -> Ordering {
        self.to_bytes().cmp(&other.to_bytes())
    }
}

impl ToOwned for CWideStr {
    type Owned = CWideString;

    fn to_owned(&self) -> CWideString {
        unsafe { CWideString::from_vec_unchecked(self.to_bytes().to_vec()) }
    }
}

impl<'a> From<&'a CWideStr> for CWideString {
    fn from(s: &'a CWideStr) -> CWideString {
        s.to_owned()
    }
}

impl ops::Index<ops::RangeFull> for CWideString {
    type Output = CWideStr;

    #[inline]
    fn index(&self, _index: ops::RangeFull) -> &CWideStr {
        self
    }
}

impl AsRef<CWideStr> for CWideStr {
    fn as_ref(&self) -> &CWideStr {
        self
    }
}

impl AsRef<CWideStr> for CWideString {
    fn as_ref(&self) -> &CWideStr {
        self
    }
}

#[inline]
fn wmemchr(x: u16, text: &[u16]) -> Option<usize> {
    for i in 0..text.len() {
        if text[i] == x {
            return Some(i);
        }
    }
    None
}

#[inline]
unsafe fn wstrlen(cs: *const u16) -> libc::size_t {
    let mut len = 0;
    loop {
        if *cs.offset(len as isize) == 0u16 {
            break;
        }
        len += 1;
    }
    len
}

#[cfg(test)]
mod tests {
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
            assert_eq!(CWideStr::from_ptr(ptr).to_bytes(), &utf16("123")[..]);
            assert_eq!(CWideStr::from_ptr(ptr).to_bytes_with_nul(), &utf16("123\0")[..]);
        }
    }

    #[test]
    fn simple() {
        let s = CWideString::from_str("1234").unwrap();
        assert_eq!(s.as_bytes(), &utf16("1234")[..]);
        assert_eq!(s.as_bytes_with_nul(), &utf16("1234\0")[..]);
    }

    #[test]
    fn build_with_zero1() {
        assert!(CWideString::from_str("\0").is_err());
    }
    #[test]
    fn build_with_zero2() {
        assert!(CWideString::new(vec![0]).is_err());
    }

    #[test]
    fn build_with_zero3() {
        unsafe {
            let s = CWideString::from_vec_unchecked(vec![0]);
            assert_eq!(s.as_bytes(), &[0u16]);
        }
    }

    #[test]
    fn formatted() {
        /*todo: let s = CWideString::new(&"abc\x01\x02\n\xE2\x80\xA6\xFF"[..]).unwrap();
        assert_eq!(format!("{:?}", s), r#""abc\x01\x02\n\xe2\x80\xa6\xff""#);*/
    }

    #[test]
    fn borrowed() {
        unsafe {
            let s = CWideStr::from_ptr(utf16("12\0").as_ptr() as *const _);
            assert_eq!(s.to_bytes(), &utf16("12")[..]);
            assert_eq!(s.to_bytes_with_nul(), &utf16("12\0")[..]);
        }
    }

    #[test]
    fn to_string() {
        let data: Vec<u16> = "123".encode_utf16().chain(Some(0xD83Du16)).chain(Some(0xDE03u16)).chain(Some(0)).collect();
        let ptr = data.as_ptr();
        unsafe {
            assert!(CWideStr::from_ptr(ptr).to_string().is_ok());
            assert_eq!(CWideStr::from_ptr(ptr).to_string().unwrap(), "123😃");
            assert_eq!(CWideStr::from_ptr(ptr).to_string_lossy(), Borrowed("123😃"));
        }
        let data: Vec<u16> = "123".encode_utf16().chain(Some(0xD83Du16)).chain(Some(0)).collect();
        let ptr = data.as_ptr();
        unsafe {
            assert!(CWideStr::from_ptr(ptr).to_string().is_err());
            assert_eq!(CWideStr::from_ptr(ptr).to_string_lossy(), Owned::<str>(format!("123\u{FFFD}")));
        }
    }

    #[test]
    fn from_ptr() {
        let data = utf16("123\0");
        let ptr = data.as_ptr();

        let owned = unsafe { CWideStr::from_ptr(ptr).to_owned() };
        assert_eq!(owned.as_bytes_with_nul(), &data[..]);
    }

    #[test]
    fn equal_hash() {
        /*todo: let data = b"123\xE2\xFA\xA6\0";
        let ptr = data.as_ptr();
        let cstr: &'static CWideStr = unsafe { CWideStr::from_ptr(ptr) };

        let mut s = SipHasher::new_with_keys(0, 0);
        cstr.hash(&mut s);
        let cstr_hash = s.finish();
        let mut s = SipHasher::new_with_keys(0, 0);
        CWideString::new(&data[..data.len() - 1]).unwrap().hash(&mut s);
        let cstring_hash = s.finish();

        assert_eq!(cstr_hash, cstring_hash);*/
    }

    #[test]
    fn from_bytes_with_nul() {
        let data = utf16("123\0");
        let cstr = CWideStr::from_bytes_with_nul(&data);
        assert_eq!(cstr.map(CWideStr::to_bytes), Some(&utf16("123")[..]));
        assert_eq!(cstr.map(CWideStr::to_bytes_with_nul), Some(&utf16("123\0")[..]));

        unsafe {
            let cstr_unchecked = CWideStr::from_bytes_with_nul_unchecked(&data);
            assert_eq!(cstr, Some(cstr_unchecked));
        }
    }

    #[test]
    fn from_bytes_with_nul_unterminated() {
        let data = utf16("123");
        let cstr = CWideStr::from_bytes_with_nul(&data);
        assert!(cstr.is_none());
    }

    #[test]
    fn from_bytes_with_nul_interior() {
        let data = utf16("1\023\0");
        let cstr = CWideStr::from_bytes_with_nul(&data);
        assert!(cstr.is_none());
    }
}
