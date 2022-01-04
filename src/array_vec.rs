use bytemuck::{Pod, Zeroable};
use std::mem::{ManuallyDrop, MaybeUninit};

pub struct ArrayVec<T, const CAP: usize> {
    data: [MaybeUninit<T>; CAP],
    len: usize,
}

impl<T, const CAP: usize> ArrayVec<T, CAP> {
    pub fn new() -> Self {
        Self {
            data: MaybeUninit::uninit_array(),
            len: 0,
        }
    }

    pub fn push(&mut self, elem: T) -> Option<T> {
        if self.len < CAP {
            self.data[self.len] = MaybeUninit::new(elem);
            self.len += 1;
            None
        } else {
            Some(elem)
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len > 0 {
            self.len -= 1;
            let mut value = MaybeUninit::uninit();
            std::mem::swap(&mut self.data[self.len], &mut value);
            Some(unsafe { MaybeUninit::assume_init(value) })
        } else {
            None
        }
    }

    #[warn(unsafe_op_in_unsafe_fn)]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe { MaybeUninit::assume_init_ref(self.data.get_unchecked(index)) }
    }
    #[warn(unsafe_op_in_unsafe_fn)]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        unsafe { MaybeUninit::assume_init_mut(self.data.get_unchecked_mut(index)) }
    }
}

impl<T: Zeroable + Pod, const CAP: usize> ArrayVec<T, CAP> {
    pub fn as_byte_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self)
    }
}

impl<T, const CAP: usize> std::ops::Drop for ArrayVec<T, CAP> {
    fn drop(&mut self) {
        for d in &mut self.data[0..self.len] {
            unsafe { d.assume_init_drop() }
        }
    }
}

impl<T, const CAP: usize> std::ops::Deref for ArrayVec<T, CAP> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        // SAFETY:
        // there are `len` count elements stored from 0 to `len`, and they are valid
        unsafe { MaybeUninit::slice_assume_init_ref(&self.data[0..self.len]) }
    }
}

impl<T, const CAP: usize> std::ops::DerefMut for ArrayVec<T, CAP> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY:
        // there are `len` count elements stored from 0 to `len`, and they are valid
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.data[0..self.len]) }
    }
}

impl<T, const CAP: usize> Default for ArrayVec<T, CAP> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ArrayVecIter<T, const CAP: usize> {
    data: [MaybeUninit<T>; CAP],
    len: usize,
    idx: usize,
}

impl<T, const CAP: usize> Iterator for ArrayVecIter<T, CAP> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.len {
            let item = unsafe {
                MaybeUninit::assume_init(std::mem::replace(
                    self.data.get_unchecked_mut(self.idx),
                    MaybeUninit::uninit(),
                ))
            };
            self.idx += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len - self.idx, Some(self.len - self.idx))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len - self.idx
    }
}

impl<T, const CAP: usize> Drop for ArrayVecIter<T, CAP> {
    fn drop(&mut self) {
        for d in &mut self.data[self.idx..self.len] {
            unsafe { d.assume_init_drop() }
        }
    }
}

impl<T, const CAP: usize> IntoIterator for ArrayVec<T, CAP> {
    type Item = T;
    type IntoIter = ArrayVecIter<T, CAP>;

    fn into_iter(self) -> Self::IntoIter {
        let mut me = ManuallyDrop::new(self);
        let data = std::mem::replace(&mut me.data, MaybeUninit::uninit_array());
        ArrayVecIter {
            data,
            len: me.len,
            idx: 0,
        }
    }
}

impl<'i, T, const CAP: usize> IntoIterator for &'i ArrayVec<T, CAP> {
    type Item = <&'i [T] as IntoIterator>::Item;
    type IntoIter = <&'i [T] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'i, T, const CAP: usize> IntoIterator for &'i mut ArrayVec<T, CAP> {
    type Item = <&'i mut [T] as IntoIterator>::Item;
    type IntoIter = <&'i mut [T] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

pub fn optional_array<T, const S: usize>() -> [Option<T>; S] {
    let mut arr: [MaybeUninit<Option<T>>; S] = MaybeUninit::uninit_array();
    for e in arr.iter_mut() {
        e.write(None);
    }
    unsafe { MaybeUninit::array_assume_init(arr) }
}

pub fn default_array<T: Default, const S: usize>() -> [T; S] {
    let mut arr: [MaybeUninit<T>; S] = MaybeUninit::uninit_array();
    for e in arr.iter_mut() {
        e.write(<T as Default>::default());
    }
    unsafe { MaybeUninit::array_assume_init(arr) }
}

#[macro_export]
macro_rules! array_vec {
    ($($v:expr),*) => {{
        let mut array_vec = crate::array_vec::ArrayVec::new();
        $(
            array_vec.push($v);
        )*
        array_vec
    }};
}
