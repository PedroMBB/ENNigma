use crate::{AddContext, Gen2DArray, NumberType};
use serde::{de::Visitor, Deserialize};
use std::{marker::PhantomData, sync::Arc};

#[derive(Debug)]
pub struct Gen2DArrayNoContext<T, const ROWS: usize, const COLS: usize> {
    contents: Vec<T>,
    rows: PhantomData<[T; ROWS]>,
    cols: PhantomData<[T; COLS]>,
}

impl<
        T: NumberType + AddContext<T::ContextType, FromType = T1>,
        T1,
        const ROWS: usize,
        const COLS: usize,
    > AddContext<T::ContextType> for Gen2DArray<T, ROWS, COLS>
{
    type FromType = Gen2DArrayNoContext<T1, ROWS, COLS>;
    fn add_context(t: &Self::FromType, ctx: &Arc<T::ContextType>) -> Self {
        Self {
            contents: t
                .contents
                .iter()
                .map(|v| T::add_context(v, ctx))
                .collect::<Vec<_>>()
                .try_into()
                .expect("Should be able to convert Vec to Array AddContext"),
            context: Arc::clone(ctx),
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        }
    }
}

impl<'de, T: Deserialize<'de>, const ROWS: usize, const COLS: usize> Deserialize<'de>
    for Gen2DArrayNoContext<T, ROWS, COLS>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MatrixVisitor<T, const PREV_N: usize, const CURR_N: usize>(
            PhantomData<[[T; PREV_N]; CURR_N]>,
        );
        impl<'de, T: Deserialize<'de>, const PREV_N: usize, const CURR_N: usize> Visitor<'de>
            for MatrixVisitor<T, PREV_N, CURR_N>
        {
            type Value = Vec<T>;
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let total = PREV_N * CURR_N;

                let weights: Vec<T> = (0..total)
                    .map(|_| seq.next_element())
                    .filter_map(|v| v.transpose())
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(weights)
            }

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "The matrix values")
            }
        }

        Ok(Self {
            contents: deserializer
                .deserialize_seq(MatrixVisitor::<T, ROWS, COLS>(PhantomData::default()))?,
            rows: PhantomData::default(),
            cols: PhantomData::default(),
        })
    }
}
