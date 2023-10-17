use std::{
    ops::{BitAnd, BitOr, BitXor},
    sync::Arc,
};

mod bool_impl;
pub use bool_impl::*;

pub trait BooleanType<T>:
    Sized
    + RefBitAnd<Output = Self>
    + BitOr<Output = Self>
    + RefBitOr<Output = Self>
    + BitXor<Output = Self>
    + RefBitXor<Output = Self>
    + Mux<Output = Self>
    + FromWithContext<bool, T>
    + BooleanAssignOperations
    + BooleanRefOperations
    + SwitchContext<T>
    + Into<bool>
    + std::fmt::Debug
    + Clone
{
}

pub trait NumberType {
    type ContextType;
}

pub trait RefBitAnd<Rhs = Self> {
    type Output;
    fn and(&self, rhs: &Rhs) -> Self::Output;
}
impl<T, Rhs, Output> RefBitAnd<Rhs> for T
where
    for<'a> &'a T: BitAnd<&'a Rhs, Output = Output>,
{
    type Output = Output;

    fn and(&self, rhs: &Rhs) -> Self::Output {
        self & rhs
    }
}

pub trait Max {
    fn max(&self, rhs: &Self) -> Self;
}
pub trait Min {
    fn min(&self, rhs: &Self) -> Self;
}
pub trait IsGreater {
    fn is_greater(&self, rhs: &Self) -> Self;
}
pub trait Equal {
    fn equal(&self, rhs: &Self) -> Self;
}

/*
    Boolean Operations
*/

pub trait BooleanRefOperations {
    #[deprecated]
    fn not_ref(&self) -> Self;
    fn and_ref(&self, rhs: &Self) -> Self;
    fn or_ref(&self, rhs: &Self) -> Self;
    fn xor_ref(&self, rhs: &Self) -> Self;
    fn nand_ref(&self, rhs: &Self) -> Self;
    fn nor_ref(&self, rhs: &Self) -> Self;
    fn xnor_ref(&self, rhs: &Self) -> Self;
}

pub trait BooleanAssignOperations {
    fn not_assign(&mut self);

    fn and_assign(&mut self, rhs: &Self);
    fn or_assign(&mut self, rhs: &Self);
    fn xor_assign(&mut self, rhs: &Self);

    fn nand_assign(&mut self, rhs: &Self);
    fn xnor_assign(&mut self, rhs: &Self);
    fn nor_assign(&mut self, rhs: &Self);
}

pub trait RefBitOr<Rhs = Self> {
    type Output;
    fn or(&self, rhs: &Rhs) -> Self::Output;
}
impl<T, Rhs, Output> RefBitOr<Rhs> for T
where
    for<'a> &'a T: BitOr<&'a Rhs, Output = Output>,
{
    type Output = Output;

    fn or(&self, rhs: &Rhs) -> Self::Output {
        self | rhs
    }
}

pub trait RefBitXor<Rhs = Self> {
    type Output;
    fn xor(&self, rhs: &Rhs) -> Self::Output;
}
impl<T, Rhs, Output> RefBitXor<Rhs> for T
where
    for<'a> &'a T: BitXor<&'a Rhs, Output = Output>,
{
    type Output = Output;

    fn xor(&self, rhs: &Rhs) -> Self::Output {
        self ^ rhs
    }
}

pub trait Abs {
    type Output;
    fn abs(self) -> Self::Output;
    fn abs_assign(&mut self);
}

pub trait DivRemainder<T> {
    type Output;
    fn div_remainder(self, divisor: T) -> (Self::Output, Self::Output);
}
pub trait Sqrt {
    type Output;
    fn sqrt(self) -> Self::Output;
}
pub trait Mux {
    type Output;
    fn mux(&self, yes: &Self::Output, no: &Self::Output) -> Self::Output;
}

/*

Number Operations

*/

// TODO: Replace by FromWithContext and create different types for server and client contexts
pub trait SwitchContext<C> {
    fn switch_context(&self, ctx: &Arc<C>) -> Self;
}
pub trait FromWithContext<T, CTX> {
    fn from_ctx(t: T, ctx: &Arc<CTX>) -> Self;
}

pub trait DefaultWithContext<CTX> {
    fn default_ctx(ctx: &Arc<CTX>) -> Self;
}

pub trait OneWithContext<CTX> {
    fn one_ctx(ctx: &Arc<CTX>) -> Self;
}

impl Sqrt for f32 {
    type Output = f32;
    fn sqrt(self) -> Self::Output {
        f32::sqrt(self)
    }
}

impl<C> AddContext<C> for f32 {
    type FromType = f32;
    fn add_context(t: &Self::FromType, _ctx: &Arc<C>) -> Self {
        t.clone()
    }
}

impl<C> FromWithContext<f32, C> for f32 {
    fn from_ctx(t: f32, _ctx: &Arc<C>) -> Self {
        t
    }
}
impl<C> DefaultWithContext<C> for f32 {
    fn default_ctx(_: &Arc<C>) -> Self {
        0.0
    }
}
impl<C> OneWithContext<C> for f32 {
    fn one_ctx(_: &Arc<C>) -> Self {
        1.0
    }
}

impl NumberType for f32 {
    type ContextType = ();
}

impl NumberType for i32 {
    type ContextType = ();
}
pub trait AddContext<C> {
    type FromType;
    fn add_context(t: &Self::FromType, ctx: &Arc<C>) -> Self;
}
