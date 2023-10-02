use numbers::{FromWithContext, SwitchContext};
use serde::Serialize;
use std::{fmt::Debug, sync::Arc};
use tfhe::boolean::prelude::Ciphertext;

use crate::EncryptedContext;

mod noctx;
mod operations;

pub use noctx::EncryptedBitNoContext;

#[derive(Clone)]
pub struct EncryptedBit {
    context: Arc<EncryptedContext>,
    bit: Ciphertext,
}

impl Serialize for EncryptedBit {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bit = self.bit.clone();
        // TODO: Handle trivial encrypted bits
        EncryptedBitNoContext { bit }.serialize(serializer)
    }
}
impl Debug for EncryptedBit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self.bit))
    }
}

impl SwitchContext<EncryptedContext> for EncryptedBit {
    fn switch_context(&self, ctx: &Arc<EncryptedContext>) -> Self {
        Self {
            context: ctx.clone(),
            bit: self.bit.clone(),
        }
    }
}
impl FromWithContext<bool, EncryptedContext> for EncryptedBit {
    fn from_ctx(v: bool, ctx: &Arc<EncryptedContext>) -> Self {
        Self {
            bit: match ctx.client_key.as_ref() {
                Some(key) => match ctx.server_key.is_completed() {
                    true => ctx.get_key().trivial_encrypt(v),
                    false => key.encrypt(v),
                },
                None => ctx.get_key().trivial_encrypt(v),
            },
            context: Arc::clone(ctx),
        }
    }
}
impl Into<bool> for EncryptedBit {
    fn into(self) -> bool {
        self.context
            .client_key
            .as_ref()
            .expect("Cannot decrypt a value without without the private key")
            .decrypt(&self.bit)
    }
}
