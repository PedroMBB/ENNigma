use array_macro::array;
use std::{
    marker::PhantomData,
    ops::{Add, Mul},
    sync::Arc,
};

use crate::{DefaultWithContext, FromWithContext, Gen2DArray, NumberType, Sqrt};

pub type Gen1DArray<T, const SIZE: usize> = Gen2DArray<T, 1, SIZE>;

impl<T: NumberType, const SIZE: usize> FromWithContext<Vec<T>, T::ContextType>
    for Gen1DArray<T, SIZE>
{
    fn from_ctx(lst: Vec<T>, ctx: &Arc<T::ContextType>) -> Self {
        if lst.len() != SIZE {
            panic!("[Gen1DArray] From<Vec<_>> cannot handle vectors with length ({}) different from size ({})", lst.len(), SIZE);
        }

        Gen2DArray {
            contents: lst
                .try_into()
                .expect("Should convert array to vec (from_ctx 1D)"),
            context: Arc::clone(&ctx),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

impl<T: NumberType + DefaultWithContext<T::ContextType>, const COLS: usize> Gen1DArray<T, COLS>
where
    for<'a> &'a T: Add<&'a T, Output = T>,
    for<'a> &'a T: Mul<&'a T, Output = T>,
    T: Sqrt<Output = T>,
{
    pub fn length(self) -> T {
        let sum = self
            .contents
            .iter()
            .map(|v| v * v)
            .fold::<T, _>(DefaultWithContext::default_ctx(&self.context), |a, b| {
                &a + &b
            });

        sum.sqrt()
    }
}

impl<T: NumberType + Clone, const SIZE: usize> Into<Vec<T>> for &Gen1DArray<T, SIZE> {
    fn into(self) -> Vec<T> {
        self.contents.iter().cloned().collect()
    }
}
impl<'a, T: NumberType, const SIZE: usize> Into<Vec<&'a T>> for &'a Gen1DArray<T, SIZE> {
    fn into(self) -> Vec<&'a T> {
        self.contents.iter().collect()
    }
}
impl<T: NumberType + Clone, const SIZE: usize> Gen1DArray<T, SIZE> {
    pub fn as_1d_array(self) -> [T; SIZE] {
        let new_arr: [T; SIZE] = array![i => self.contents[i].clone(); SIZE];

        new_arr
    }
}
