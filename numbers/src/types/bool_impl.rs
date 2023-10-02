use std::{ops::Not, sync::Arc};

use crate::{
    AddContext, BooleanAssignOperations, BooleanRefOperations, BooleanType, FixedPointNumber,
    FromWithContext, Mux, SwitchContext,
};

pub type BoolFixedPointNumber<const SIZE: usize, const PRECISION: usize> =
    FixedPointNumber<SIZE, PRECISION, bool, ()>;

/* Default boolean impl */
impl Mux for bool {
    type Output = bool;
    fn mux(&self, yes: &Self::Output, no: &Self::Output) -> Self::Output {
        match self {
            true => *yes,
            false => *no,
        }
    }
}
impl FromWithContext<bool, ()> for bool {
    fn from_ctx(t: bool, _ctx: &Arc<()>) -> Self {
        t
    }
}
impl AddContext<()> for bool {
    type FromType = bool;
    fn add_context(t: &Self::FromType, _ctx: &Arc<()>) -> Self {
        *t
    }
}
impl BooleanRefOperations for bool {
    fn not_ref(&self) -> Self {
        self.not()
    }
    fn and_ref(&self, rhs: &Self) -> Self {
        *self & *rhs
    }
    fn or_ref(&self, rhs: &Self) -> Self {
        *self | *rhs
    }
    fn xor_ref(&self, rhs: &Self) -> Self {
        *self ^ *rhs
    }
    fn nand_ref(&self, rhs: &Self) -> Self {
        !(*self & *rhs)
    }
    fn nor_ref(&self, rhs: &Self) -> Self {
        if !*self & !*rhs {
            return true;
        } else {
            return false;
        }
    }
    fn xnor_ref(&self, rhs: &Self) -> Self {
        !(*self ^ *rhs)
    }
}
impl BooleanAssignOperations for bool {
    fn not_assign(&mut self) {
        *self = !*self;
    }

    fn and_assign(&mut self, rhs: &Self) {
        *self = *self & *rhs;
    }
    fn or_assign(&mut self, rhs: &Self) {
        *self = *self | *rhs;
    }
    fn xor_assign(&mut self, rhs: &Self) {
        *self = *self ^ *rhs;
    }

    fn nand_assign(&mut self, rhs: &Self) {
        *self = !(*self & *rhs)
    }
    fn nor_assign(&mut self, rhs: &Self) {
        if !*self & !*rhs {
            *self = true;
        } else {
            *self = false;
        }
    }
    fn xnor_assign(&mut self, rhs: &Self) {
        *self = !(*self ^ *rhs);
    }
}
impl SwitchContext<()> for bool {
    fn switch_context(&self, _ctx: &Arc<()>) -> Self {
        *self
    }
}
impl BooleanType<()> for bool {}
