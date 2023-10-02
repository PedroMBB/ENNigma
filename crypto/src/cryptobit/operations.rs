use crate::EncryptedContext;

use super::EncryptedBit;
use numbers::{BooleanAssignOperations, BooleanRefOperations, BooleanType, Mux};
use std::{
    ops::{BitAnd, BitOr, BitXor, Not},
    sync::Arc,
};
use tfhe::boolean::{prelude::*, server_key::BinaryBooleanGatesAssign};

impl BitAnd<EncryptedBit> for EncryptedBit {
    type Output = EncryptedBit;
    fn bitand(mut self, rhs: EncryptedBit) -> Self::Output {
        self.context.get_key().and_assign(&mut self.bit, &rhs.bit);
        self
    }
}
impl BitAnd<&EncryptedBit> for &EncryptedBit {
    type Output = EncryptedBit;
    fn bitand(self, rhs: &EncryptedBit) -> Self::Output {
        Self::Output {
            context: Arc::clone(&self.context),
            bit: self.context.get_key().and(&self.bit, &rhs.bit),
        }
    }
}

impl BitOr<EncryptedBit> for EncryptedBit {
    type Output = EncryptedBit;
    fn bitor(mut self, rhs: EncryptedBit) -> Self::Output {
        self.context.get_key().or_assign(&mut self.bit, &rhs.bit);
        self
    }
}
impl BitOr<&EncryptedBit> for &EncryptedBit {
    type Output = EncryptedBit;
    fn bitor(self, rhs: &EncryptedBit) -> Self::Output {
        Self::Output {
            context: Arc::clone(&self.context),
            bit: self.context.get_key().or(&self.bit, &rhs.bit),
        }
    }
}

impl BitXor<EncryptedBit> for EncryptedBit {
    type Output = EncryptedBit;
    fn bitxor(mut self, rhs: EncryptedBit) -> Self::Output {
        self.context.get_key().xor_assign(&mut self.bit, &rhs.bit);
        self
    }
}
impl BitXor<&EncryptedBit> for &EncryptedBit {
    type Output = EncryptedBit;
    fn bitxor(self, rhs: &EncryptedBit) -> Self::Output {
        Self::Output {
            context: Arc::clone(&self.context),
            bit: self.context.get_key().xor(&self.bit, &rhs.bit),
        }
    }
}

impl Not for EncryptedBit {
    type Output = EncryptedBit;
    fn not(mut self) -> Self::Output {
        self.context.get_key().not_assign(&mut self.bit);
        self
    }
}
impl BooleanRefOperations for EncryptedBit {
    fn not_ref(&self) -> Self {
        Self {
            bit: self.context.get_key().not(&self.bit),
            context: Arc::clone(&self.context),
        }
    }
    fn and_ref(&self, rhs: &Self) -> Self {
        Self {
            bit: self.context.get_key().and(&self.bit, &rhs.bit),
            context: Arc::clone(&self.context),
        }
    }
    fn or_ref(&self, rhs: &Self) -> Self {
        Self {
            bit: self.context.get_key().or(&self.bit, &rhs.bit),
            context: Arc::clone(&self.context),
        }
    }
    fn xor_ref(&self, rhs: &Self) -> Self {
        Self {
            bit: self.context.get_key().xor(&self.bit, &rhs.bit),
            context: Arc::clone(&self.context),
        }
    }
    fn nand_ref(&self, rhs: &Self) -> Self {
        Self {
            bit: self.context.get_key().nand(&self.bit, &rhs.bit),
            context: Arc::clone(&self.context),
        }
    }
    fn nor_ref(&self, rhs: &Self) -> Self {
        Self {
            bit: self.context.get_key().nor(&self.bit, &rhs.bit),
            context: Arc::clone(&self.context),
        }
    }
    fn xnor_ref(&self, rhs: &Self) -> Self {
        Self {
            bit: self.context.get_key().xnor(&self.bit, &rhs.bit),
            context: Arc::clone(&self.context),
        }
    }
}

impl BooleanAssignOperations for EncryptedBit {
    fn not_assign(&mut self) {
        self.context.get_key().not_assign(&mut self.bit)
    }

    fn and_assign(&mut self, rhs: &Self) {
        self.context.get_key().and_assign(&mut self.bit, &rhs.bit);
    }
    fn or_assign(&mut self, rhs: &Self) {
        self.context.get_key().or_assign(&mut self.bit, &rhs.bit);
    }
    fn xor_assign(&mut self, rhs: &Self) {
        self.context.get_key().xor_assign(&mut self.bit, &rhs.bit);
    }

    fn nand_assign(&mut self, rhs: &Self) {
        self.context.get_key().nand_assign(&mut self.bit, &rhs.bit);
    }
    fn nor_assign(&mut self, rhs: &Self) {
        self.context.get_key().nor_assign(&mut self.bit, &rhs.bit);
    }
    fn xnor_assign(&mut self, rhs: &Self) {
        self.context.get_key().xnor_assign(&mut self.bit, &rhs.bit);
    }
}

impl Mux for EncryptedBit {
    type Output = EncryptedBit;
    fn mux(&self, yes: &Self::Output, no: &Self::Output) -> Self::Output {
        Self {
            context: Arc::clone(&self.context),
            bit: self.context.get_key().mux(&self.bit, &yes.bit, &no.bit),
        }
    }
}

impl BooleanType<EncryptedContext> for EncryptedBit {}
