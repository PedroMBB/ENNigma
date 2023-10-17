use crate::ActivationFn;
use numbers::{BooleanType, FixedPointNumber, FromWithContext, Gen1DArray, IsGreater, Max, Min};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Default)]
pub struct ReLUAF {}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C, const N: usize>
    ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N> for ReLUAF
where
    FixedPointNumber<SIZE, PRECISION, T, C>: FromWithContext<f32, C>,
{
    fn activate(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> FixedPointNumber<SIZE, PRECISION, T, C> {
        let min_v = FromWithContext::from_ctx(0.0, ctx);

        v.max(&min_v)
    }

    fn activate_and_derivative(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> (
        FixedPointNumber<SIZE, PRECISION, T, C>,
        FixedPointNumber<SIZE, PRECISION, T, C>,
    ) {
        let min_v = FromWithContext::from_ctx(0.0, ctx);
        let one_v = FromWithContext::from_ctx(1.0, ctx);

        let act = v.max(&min_v);
        let der = v.is_greater(&min_v).mux_number(&one_v, &min_v);

        (act, der)
    }

    fn activate_multiple(
        &self,
        ctx: &Arc<C>,
        lst: Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) -> Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N> {
        lst.apply(|v| {
            <ReLUAF as ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N>>::activate(
                self,
                ctx,
                v.clone(),
            )
        })
    }
    fn activate_and_derivative_multiple(
        &self,
        ctx: &Arc<C>,
        lst: Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) -> (
        Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
        Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) {
        lst.apply_two(|v| <ReLUAF as ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N>>::activate_and_derivative(self, ctx, v.clone()))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReLUTruncAF {}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C, const N: usize>
    ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N> for ReLUTruncAF
where
    FixedPointNumber<SIZE, PRECISION, T, C>: FromWithContext<f32, C>,
{
    fn activate(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> FixedPointNumber<SIZE, PRECISION, T, C> {
        let v = v + &FromWithContext::from_ctx(0.5, ctx);

        let min_v = FromWithContext::from_ctx(0.0, ctx);
        let max_v = FromWithContext::from_ctx(1.0, ctx);

        v.max(&min_v).min(&max_v)
    }

    fn activate_and_derivative(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> (
        FixedPointNumber<SIZE, PRECISION, T, C>,
        FixedPointNumber<SIZE, PRECISION, T, C>,
    ) {
        let v = v + &FromWithContext::from_ctx(0.5, ctx);

        let min_v = FromWithContext::from_ctx(0.0, ctx);
        let one_v = FromWithContext::from_ctx(1.0, ctx);

        let act = v.max(&min_v).min(&one_v);

        let der = v.is_greater(&min_v).mux_number(&v, &min_v);
        let der = one_v.is_greater(&v).mux_number(&der, &one_v);

        (act, der)
    }

    fn activate_multiple(
        &self,
        ctx: &Arc<C>,
        lst: Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) -> Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N> {
        lst.apply(|v| {
            <ReLUTruncAF as ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N>>::activate(
                self,
                ctx,
                v.clone(),
            )
        })
    }
    fn activate_and_derivative_multiple(
        &self,
        ctx: &Arc<C>,
        lst: Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) -> (
        Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
        Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) {
        lst.apply_two(|v| <ReLUTruncAF as ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N>>::activate_and_derivative(self, ctx, v.clone()))
    }
}
