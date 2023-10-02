use std::sync::Arc;

use crate::{BooleanType, FixedPointNumber, FromWithContext};

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C>
    FixedPointNumber<SIZE, PRECISION, T, C>
{
    pub fn truncate(mut self) -> Self {
        let msb: T = FromWithContext::from_ctx(false, &self.context);
        let b = Arc::make_mut(&mut self.bits);

        for i in 0..PRECISION {
            b[i] = msb.clone();
        }

        self
    }
}
