use std::sync::Arc;

use numbers::AddContext;
use serde::{Deserialize, Serialize};
use tfhe::boolean::prelude::Ciphertext;

use crate::{EncryptedBit, EncryptedContext};

#[derive(Deserialize, Serialize)]
pub struct EncryptedBitNoContext {
    pub(super) bit: Ciphertext,
}

impl AddContext<EncryptedContext> for EncryptedBit {
    type FromType = EncryptedBitNoContext;
    fn add_context(t: &Self::FromType, ctx: &Arc<EncryptedContext>) -> Self {
        Self {
            bit: t.bit.clone(),
            context: Arc::clone(ctx),
        }
    }
}
