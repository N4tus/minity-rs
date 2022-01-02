use bytemuck::{Pod, Zeroable};
use std::mem::MaybeUninit;

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

    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe { MaybeUninit::assume_init_ref(self.data.get_unchecked(index)) }
    }

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
