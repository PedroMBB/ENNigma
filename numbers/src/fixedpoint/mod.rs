use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, sync::Arc};

use crate::{BooleanType, NumberType};

mod implementations;
mod operations;

pub use operations::karatsuba_adapt;

#[derive(Debug)]
pub struct FixedPointNumber<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C> {
    pub(crate) bits: Arc<[T; SIZE]>,
    precision: PhantomData<[usize; PRECISION]>,
    pub(crate) context: Arc<C>,
}

#[derive(Deserialize)]
pub struct FixedPointNumberNoContext<const SIZE: usize, const PRECISION: usize, T> {
    pub(crate) bits: Vec<T>,
    #[serde(default = "PhantomData::default")]
    size: PhantomData<[usize; SIZE]>,
    #[serde(default = "PhantomData::default")]
    precision: PhantomData<[usize; PRECISION]>,
}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C> NumberType
    for FixedPointNumber<SIZE, PRECISION, T, C>
{
    type ContextType = C;
}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C> + Serialize, C> Serialize
    for FixedPointNumber<SIZE, PRECISION, T, C>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        pub struct FixedPointNumberSer<'a, T> {
            bits: &'a [T],
        }

        let val = FixedPointNumberSer {
            bits: self.bits.as_slice(),
        };

        val.serialize(serializer)
    }
}

pub fn iterator_to_sized_arc<const SIZE: usize, T, I: Iterator<Item = T>>(
    iterator: I,
) -> Arc<[T; SIZE]> {
    let array: Arc<[T]> = iterator.take(SIZE).collect();
    let size = array.len();

    let new_arr: Arc<[T; SIZE]> = array
        .try_into()
        .map_err(|_| {
            format!(
                "Could not convert slice to array, expected size {}, but got size {}",
                SIZE, size
            )
        })
        .unwrap();
    new_arr
}
