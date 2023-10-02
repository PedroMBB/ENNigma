use numbers::FixedPointNumber;

use crate::{EncryptedBit, EncryptedContext};

pub type EncryptedFixedPointNumber<const SIZE: usize, const PRECISION: usize> =
    FixedPointNumber<SIZE, PRECISION, EncryptedBit, EncryptedContext>;
