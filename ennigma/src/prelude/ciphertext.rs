use numbers::NumberType as NT;
use std::sync::Arc;

use crypto::{EncryptedBit, EncryptedContext, EncryptedFixedPointNumber};
use neuralnetworks::Model;
use numbers::{Gen1DArray, Gen2DArray, SwitchContext};

use crate::CryptoFFFFLayer;

pub type BitType = EncryptedBit;
pub type ContextType = EncryptedContext;

pub type NumberType<const SIZE: usize, const PRECISION: usize> =
    EncryptedFixedPointNumber<SIZE, PRECISION>;

pub type ModelType<
    const SIZE: usize,
    const PRECISION: usize,
    const INPUT_N: usize,
    const OUTPUT_N: usize,
> = Model<EncryptedFixedPointNumber<SIZE, PRECISION>, INPUT_N, OUTPUT_N>;

pub type Gen2DArrayType<
    const SIZE: usize,
    const PRECISION: usize,
    const ROWS: usize,
    const COLS: usize,
> = Gen2DArray<EncryptedFixedPointNumber<SIZE, PRECISION>, ROWS, COLS>;
pub type Gen1DArrayType<const SIZE: usize, const PRECISION: usize, const ROWS: usize> =
    Gen1DArray<EncryptedFixedPointNumber<SIZE, PRECISION>, ROWS>;

pub type FFFFLayerType<
    const BITS: usize,
    const CALC_BITS: usize,
    const PRECISION: usize,
    const PREV_N: usize,
    const CURR_N: usize,
    AF,
    LOSS,
> = CryptoFFFFLayer<BITS, CALC_BITS, PRECISION, PREV_N, CURR_N, AF, LOSS>;

pub fn generate_context() -> (Arc<ContextType>, Arc<ContextType>) {
    let ctx: ContextType = ContextType::default();
    let client_ctx = Arc::new(ctx.remove_server_key());
    let server_ctx = Arc::new(client_ctx.as_ref().get_server_context());

    (client_ctx, server_ctx)
}

pub fn to_plaintext<const SIZE: usize, const PRECISION: usize>(
    v: &NumberType<SIZE, PRECISION>,
    client_ctx: &Arc<ContextType>,
) -> f32 {
    v.clone().switch_context(client_ctx).into()
}

pub fn to_plaintext_array<
    T: NT + SwitchContext<T::ContextType> + Into<f32>,
    const ROWS: usize,
    const COLS: usize,
>(
    arr: &Gen2DArray<T, ROWS, COLS>,
    client_ctx: &Arc<T::ContextType>,
) -> Gen2DArray<f32, ROWS, COLS> {
    arr.apply_with_context(&Arc::new(()), |_, v| v.switch_context(client_ctx).into())
}
