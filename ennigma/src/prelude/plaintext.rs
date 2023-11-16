use neuralnetworks::{layers::BoolFFFFLayer, Model};
use numbers::SwitchContext;
use numbers::{BoolFixedPointNumber, Gen1DArray, Gen2DArray};
use std::sync::Arc;

pub type BitType = bool;
pub type ContextType = ();

pub type NumberType<const SIZE: usize, const PRECISION: usize> =
    BoolFixedPointNumber<SIZE, PRECISION>;

pub type ModelType<
    const SIZE: usize,
    const PRECISION: usize,
    const INPUT_N: usize,
    const OUTPUT_N: usize,
> = Model<BoolFixedPointNumber<SIZE, PRECISION>, INPUT_N, OUTPUT_N>;

pub type Gen2DArrayType<
    const SIZE: usize,
    const PRECISION: usize,
    const ROWS: usize,
    const COLS: usize,
> = Gen2DArray<BoolFixedPointNumber<SIZE, PRECISION>, ROWS, COLS>;
pub type Gen1DArrayType<const SIZE: usize, const PRECISION: usize, const ROWS: usize> =
    Gen1DArray<BoolFixedPointNumber<SIZE, PRECISION>, ROWS>;

pub type FFFFLayerType<
    const BITS: usize,
    const CALC_BITS: usize,
    const PRECISION: usize,
    const PREV_N: usize,
    const CURR_N: usize,
    AF,
    LOSS,
> = BoolFFFFLayer<BITS, CALC_BITS, PRECISION, PREV_N, CURR_N, AF, LOSS>;

pub fn generate_context() -> (Arc<ContextType>, Arc<ContextType>) {
    let client_ctx = Arc::new(());
    let server_ctx = Arc::new(());

    (client_ctx, server_ctx)
}

pub fn to_plaintext<const SIZE: usize, const PRECISION: usize>(
    v: &NumberType<SIZE, PRECISION>,
    client_ctx: &Arc<ContextType>,
) -> f32 {
    v.clone().switch_context(client_ctx).into()
}

pub fn to_plaintext_array<
    const SIZE: usize,
    const PRECISION: usize,
    const ROWS: usize,
    const COLS: usize,
>(
    arr: &Gen2DArrayType<SIZE, PRECISION, ROWS, COLS>,
    client_ctx: &Arc<ContextType>,
) -> Gen2DArray<f32, ROWS, COLS> {
    arr.apply_with_context(&Arc::new(()), |_, v| {
        v.clone().switch_context(client_ctx).into()
    })
}
