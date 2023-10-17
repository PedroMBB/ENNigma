use crypto::{EncryptedBit, EncryptedBitNoContext, EncryptedContext};
use numbers::{FixedPointNumber, FixedPointNumberNoContext};

#[cfg(feature = "neuralnetwork")]
mod ffff_layer;
#[cfg(feature = "neuralnetwork")]
pub use ffff_layer::*;

#[cfg(feature = "neuralnetwork")]
pub type EncryptedModel<
    const SIZE: usize,
    const PRECISION: usize,
    const INPUT: usize,
    const OUTPUT: usize,
> = neuralnetworks::Model<EncryptedFixedPrecision<SIZE, PRECISION>, INPUT, OUTPUT>;

pub type EncryptedFixedPrecision<const SIZE: usize, const PRECISION: usize> =
    FixedPointNumber<SIZE, PRECISION, EncryptedBit, EncryptedContext>;
pub type EncryptedFixedPrecisionNoContext<const SIZE: usize, const PRECISION: usize> =
    FixedPointNumberNoContext<SIZE, PRECISION, EncryptedBitNoContext>;
