use crypto::{EncryptedBit, EncryptedContext};
use neuralnetworks::layers::FeedForwardFullyConnectedLayer;

pub type CryptoFFFFLayer<
    const SIZE: usize,
    const CALC_SIZE: usize,
    const PRECISION: usize,
    const PREV_N: usize,
    const CURR_N: usize,
    AF,
    LOSS,
> = FeedForwardFullyConnectedLayer<
    EncryptedBit,
    EncryptedContext,
    SIZE,
    CALC_SIZE,
    PRECISION,
    PREV_N,
    CURR_N,
    AF,
    LOSS,
>;
