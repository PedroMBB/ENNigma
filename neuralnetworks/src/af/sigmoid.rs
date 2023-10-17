use crate::ActivationFn;
use numbers::{
    Abs, BooleanType, FixedPointNumber, FromWithContext, Gen1DArray, IsGreater, Max, Min, Mux,
    NumberType,
};
use std::marker::PhantomData;
use std::ops::{Add, Not, Shr};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Default)]
pub struct SigmoidLSTruncAproxAF<const CALC_SIZE: usize, const CALC_PRECISION: usize>(
    PhantomData<[[bool; CALC_PRECISION]; CALC_SIZE]>,
);
impl SigmoidLSTruncAproxAF<16, 10> {
    pub fn default_size() -> Self {
        SigmoidLSTruncAproxAF(PhantomData::default())
    }
}

impl<
        const CALC_SIZE: usize,
        const CALC_PRECISION: usize,
        const SIZE: usize,
        const PRECISION: usize,
        T: BooleanType<C>,
        C,
        const N: usize,
    > ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N>
    for SigmoidLSTruncAproxAF<CALC_SIZE, CALC_PRECISION>
where
    FixedPointNumber<SIZE, PRECISION, T, C>: FromWithContext<f32, C>,
{
    fn activate_multiple(
        &self,
        ctx: &Arc<C>,
        lst: Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) -> Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N> {
        let min_v = FromWithContext::from_ctx(-7.5, ctx);
        let max_v = FromWithContext::from_ctx(7.5, ctx);

        let new_v = lst.apply(|v| v.max(&min_v));
        let new_v = new_v.apply(|v| v.min(&max_v));

        let mut new_v: Gen1DArray<FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C>, N> =
            new_v.apply(|v| v.update_size());

        let b0: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(0.5, ctx);
        let b1: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(1.73496, ctx);
        let b2: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(-4.19407, ctx);
        let b3: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(5.43402, ctx);
        let b4: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(-2.50739, ctx);

        let eight: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(8.0, ctx);
        new_v /= &eight;

        let v2 = new_v.clone().hadamard_prod(&new_v);
        let v3 = v2.clone().hadamard_prod(&new_v);
        let v5 = v2.clone().hadamard_prod(&v3);
        let v7 = v5.clone().hadamard_prod(&v2);

        let r0 = new_v.apply(|_| b0.clone());
        let r1: Gen1DArray<FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C>, N> = new_v * &b1;
        let r2: Gen1DArray<FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C>, N> = v3 * &b2;
        let r3: Gen1DArray<FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C>, N> = v5 * &b3;
        let r4: Gen1DArray<FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C>, N> = v7 * &b4;

        let res: Gen1DArray<FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C>, N> =
            ((r0 + &r1) + &(r2 + &r3)) + &r4;

        let r = res.apply(|v| v.update_size());
        r
    }
    fn activate_and_derivative_multiple(
        &self,
        ctx: &Arc<C>,
        lst: Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) -> (
        Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
        Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) {
        let sig = <SigmoidLSTruncAproxAF<CALC_SIZE, CALC_PRECISION> as ActivationFn<
            FixedPointNumber<SIZE, PRECISION, T, C>,
            N,
        >>::activate_multiple(&self, ctx, lst);

        let b0: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(1.0, ctx);
        let rhs = sig.apply(|v| b0.clone() - v);

        let deriv = rhs.hadamard_prod(&sig);
        (sig, deriv)
    }

    fn activate(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> FixedPointNumber<SIZE, PRECISION, T, C> {
        let min_v = FromWithContext::from_ctx(-7.5, ctx);
        let max_v = FromWithContext::from_ctx(7.5, ctx);

        let new_v = v.max(&min_v);
        let new_v = new_v.min(&max_v);

        let new_v: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> = new_v.update_size();

        let b0: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(0.5, ctx);
        let b1: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(1.73496, ctx);
        let b2: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(-4.19407, ctx);
        let b3: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(5.43402, ctx);
        let b4: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(-2.50739, ctx);

        let eight: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(8.0, ctx);
        let new_v_div = new_v / &eight;

        let v2 = new_v_div.clone() * &new_v_div;
        let v3 = v2.clone() * &new_v_div;
        let v5 = v2.clone() * &v3;
        let v7 = v5.clone() * &v2;

        let r1 = b1 * &new_v_div;
        let r2 = b2 * &v3;
        let r3 = b3 * &v5;
        let r4 = b4 * &v7;

        let res = ((b0 + &r1) + &(r2 + &r3)) + &r4;
        let r = res.update_size();
        r
    }

    fn activate_and_derivative(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> (
        FixedPointNumber<SIZE, PRECISION, T, C>,
        FixedPointNumber<SIZE, PRECISION, T, C>,
    ) {
        let sig = <SigmoidLSTruncAproxAF<CALC_SIZE, CALC_PRECISION> as ActivationFn<
            FixedPointNumber<SIZE, PRECISION, T, C>,
            N,
        >>::activate(&self, ctx, v);

        let b0: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(1.0, ctx);

        let deriv = (b0 - &sig) * &sig;
        (sig, deriv)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SigmoidPowAproxAF<const CALC_SIZE: usize, const CALC_PRECISION: usize>(
    PhantomData<[[bool; CALC_PRECISION]; CALC_SIZE]>,
);
impl SigmoidPowAproxAF<16, 10> {
    pub fn default_size() -> Self {
        SigmoidPowAproxAF(PhantomData::default())
    }
}

impl<
        const CALC_SIZE: usize,
        const CALC_PRECISION: usize,
        const SIZE: usize,
        const PRECISION: usize,
        T: BooleanType<C>,
        C,
        const N: usize,
    > ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N>
    for SigmoidPowAproxAF<CALC_SIZE, CALC_PRECISION>
where
    FixedPointNumber<SIZE, PRECISION, T, C>: FromWithContext<f32, C>,
{
    fn activate(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> FixedPointNumber<SIZE, PRECISION, T, C> {
        let val_0 = FromWithContext::from_ctx(0.0, ctx);
        let val_1_b: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(1.0, ctx);
        let val_1: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(1.0, ctx);

        let abs_v = v.clone().abs();
        let not_v: FixedPointNumber<SIZE, PRECISION, T, C> = !abs_v.clone();

        let exponent = not_v - &val_1_b;

        let truncated_b = v.clone().truncate().abs();
        let diff: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            (abs_v - &truncated_b).update_size();

        let diff = val_1.clone() - &diff;

        let exp_res: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            exponent.pow_2().update_size();

        let neg = exp_res * &diff;
        let pos = !neg.clone() + &val_1;

        let cond: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            v.is_greater(&val_0).update_size();

        cond.mux_number(&pos, &neg).update_size()
    }

    fn activate_and_derivative(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> (
        FixedPointNumber<SIZE, PRECISION, T, C>,
        FixedPointNumber<SIZE, PRECISION, T, C>,
    ) {
        let sig = <SigmoidPowAproxAF<CALC_SIZE, CALC_PRECISION> as ActivationFn<
            FixedPointNumber<SIZE, PRECISION, T, C>,
            N,
        >>::activate(&self, ctx, v);

        let sig_small: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> = sig.update_size();

        let b0: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(1.0, ctx);

        let deriv = (b0 - &sig_small) * &sig_small;

        (sig, deriv.update_size())
    }

    fn activate_multiple(
        &self,
        ctx: &Arc<C>,
        lst: Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) -> Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N> {
        lst.apply(|v| {
            <SigmoidPowAproxAF<CALC_SIZE, CALC_PRECISION> as ActivationFn<
                FixedPointNumber<SIZE, PRECISION, T, C>,
                N,
            >>::activate(self, ctx, v.clone())
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
        lst.apply_two(|v| {
            <SigmoidPowAproxAF<CALC_SIZE, CALC_PRECISION> as ActivationFn<
                FixedPointNumber<SIZE, PRECISION, T, C>,
                N,
            >>::activate_and_derivative(self, ctx, v.clone())
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SigmoidMyersAproxAF {}

impl<T: NumberType, const N: usize> ActivationFn<T, N> for SigmoidMyersAproxAF
where
    T: FromWithContext<f32, T::ContextType>,
    for<'a> &'a T: Shr<usize, Output = T>,
    for<'a> T: Add<&'a T, Output = T>,
    T: IsGreater,
    T: Mux<Output = T>,
{
    fn activate(&self, ctx: &Arc<T::ContextType>, v: T) -> T {
        let f1: T = FromWithContext::from_ctx(8.0, ctx);
        let f2: T = FromWithContext::from_ctx(4.0, ctx);
        let f3: T = FromWithContext::from_ctx(2.0, ctx);
        let f4: T = FromWithContext::from_ctx(1.0, ctx);
        let f5: T = FromWithContext::from_ctx(-1.0, ctx);
        let f6: T = FromWithContext::from_ctx(-2.0, ctx);
        let f7: T = FromWithContext::from_ctx(-4.0, ctx);
        let f8: T = FromWithContext::from_ctx(-8.0, ctx);

        let b1: T = FromWithContext::from_ctx(0.875, ctx);
        let b2: T = FromWithContext::from_ctx(0.8125, ctx);
        let b3: T = FromWithContext::from_ctx(0.625, ctx);
        let b4: T = FromWithContext::from_ctx(0.5, ctx);
        let b5: T = FromWithContext::from_ctx(0.375, ctx);
        let b6: T = FromWithContext::from_ctx(0.1875, ctx);
        let b7: T = FromWithContext::from_ctx(0.125, ctx);

        let v1: T = FromWithContext::from_ctx(1.0, ctx);
        let v2: T = (&v >> 6) + &b1;
        let v3: T = (&v >> 5) + &b2;
        let v4: T = (&v >> 3) + &b3;
        let v5: T = (&v >> 2) + &b4;
        let v6: T = (&v >> 3) + &b5;
        let v7: T = (&v >> 5) + &b6;
        let v8: T = (&v >> 6) + &b7;
        let v9: T = FromWithContext::from_ctx(0.0, ctx);

        let res = v.is_greater(&f8).mux(&v8, &v9);
        let res = v.is_greater(&f7).mux(&v7, &res);
        let res = v.is_greater(&f6).mux(&v6, &res);
        let res = v.is_greater(&f5).mux(&v5, &res);
        let res = v.is_greater(&f4).mux(&v4, &res);
        let res = v.is_greater(&f3).mux(&v3, &res);
        let res: T = v.is_greater(&f2).mux(&v2, &res);
        let res = v.is_greater(&f1).mux(&v1, &res);

        // res.optimize_integer_size(1, true)
        res
    }

    fn activate_and_derivative(&self, ctx: &Arc<T::ContextType>, v: T) -> (T, T) {
        /*
        let sig = <SigmoidMyersAproxAF as ActivationFn<
            FixedPointNumber<SIZE, PRECISION, T, C>,
            C,
            N,
        >>::activate(&self, ctx, v);

        let b0: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(1.0, ctx);

        let deriv: FixedPointNumber<SIZE, PRECISION, T, C> = (b0 - &sig) * &sig;

        (sig, deriv.optimize_integer_size(1, true))
        */

        // /*
        let f1: T = FromWithContext::from_ctx(8.0, ctx);
        let f2: T = FromWithContext::from_ctx(4.0, ctx);
        let f3: T = FromWithContext::from_ctx(2.0, ctx);
        let f4: T = FromWithContext::from_ctx(1.0, ctx);
        let f5: T = FromWithContext::from_ctx(-1.0, ctx);
        let f6: T = FromWithContext::from_ctx(-2.0, ctx);
        let f7: T = FromWithContext::from_ctx(-4.0, ctx);
        let f8: T = FromWithContext::from_ctx(-8.0, ctx);

        let b1: T = FromWithContext::from_ctx(0.875, ctx);
        let b2: T = FromWithContext::from_ctx(0.8125, ctx);
        let b3: T = FromWithContext::from_ctx(0.625, ctx);
        let b4: T = FromWithContext::from_ctx(0.5, ctx);
        let b5: T = FromWithContext::from_ctx(0.375, ctx);
        let b6: T = FromWithContext::from_ctx(0.1875, ctx);
        let b7: T = FromWithContext::from_ctx(0.125, ctx);

        let v1: T = FromWithContext::from_ctx(1.0, ctx);
        let v2: T = (&v >> 6) + &b1;
        let v3: T = (&v >> 5) + &b2;
        let v4: T = (&v >> 3) + &b3;
        let v5: T = (&v >> 2) + &b4;
        let v6: T = (&v >> 3) + &b5;
        let v7: T = (&v >> 5) + &b6;
        let v8: T = (&v >> 6) + &b7;
        let v9: T = FromWithContext::from_ctx(0.0, ctx);

        let comp1 = v.is_greater(&f8);
        let comp2 = v.is_greater(&f7);
        let comp3 = v.is_greater(&f6);
        let comp4 = v.is_greater(&f5);
        let comp5 = v.is_greater(&f4);
        let comp6 = v.is_greater(&f3);
        let comp7 = v.is_greater(&f2);
        let comp8 = v.is_greater(&f1);

        let res = comp1.mux(&v8, &v9);
        let res = comp2.mux(&v7, &res);
        let res = comp3.mux(&v6, &res);
        let res = comp4.mux(&v5, &res);
        let res = comp5.mux(&v4, &res);
        let res = comp6.mux(&v3, &res);
        let res = comp7.mux(&v2, &res);
        let res = comp8.mux(&v1, &res);

        let d9: T = FromWithContext::from_ctx(0.0, ctx);
        let d8: T = FromWithContext::from_ctx(0.015625, ctx);
        let d7: T = FromWithContext::from_ctx(0.03125, ctx);
        let d6: T = FromWithContext::from_ctx(0.125, ctx);
        let d5: T = FromWithContext::from_ctx(0.25, ctx);

        let der = comp1.mux(&d8, &d9);
        let der = comp2.mux(&d7, &der);
        let der = comp3.mux(&d6, &der);
        let der = comp4.mux(&d5, &der);
        let der = comp5.mux(&d6, &der);
        let der = comp6.mux(&d7, &der);
        let der = comp7.mux(&d8, &der);
        let der = comp8.mux(&d9, &der);

        (
            res, //.optimize_integer_size(1, true),
            der, //.optimize_integer_size(1, true),
        )
        // */
    }

    fn activate_multiple(
        &self,
        ctx: &Arc<T::ContextType>,
        lst: Gen1DArray<T, N>,
    ) -> Gen1DArray<T, N> {
        let f1: T = FromWithContext::from_ctx(8.0, ctx);
        let f2: T = FromWithContext::from_ctx(4.0, ctx);
        let f3: T = FromWithContext::from_ctx(2.0, ctx);
        let f4: T = FromWithContext::from_ctx(1.0, ctx);
        let f5: T = FromWithContext::from_ctx(-1.0, ctx);
        let f6: T = FromWithContext::from_ctx(-2.0, ctx);
        let f7: T = FromWithContext::from_ctx(-4.0, ctx);
        let f8: T = FromWithContext::from_ctx(-8.0, ctx);

        // let m1: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.015625, ctx);
        // let m2: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.03125, ctx);
        // let m3: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.125, ctx);
        // let m4: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.25, ctx);

        let b1: T = FromWithContext::from_ctx(0.875, ctx);
        let b2: T = FromWithContext::from_ctx(0.8125, ctx);
        let b3: T = FromWithContext::from_ctx(0.625, ctx);
        let b4: T = FromWithContext::from_ctx(0.5, ctx);
        let b5: T = FromWithContext::from_ctx(0.375, ctx);
        let b6: T = FromWithContext::from_ctx(0.1875, ctx);
        let b7: T = FromWithContext::from_ctx(0.125, ctx);

        let v1: T = FromWithContext::from_ctx(1.0, ctx);
        let v2 = lst.apply(|v| (v >> 6) + &b1);
        let v3 = lst.apply(|v| (v >> 5) + &b2);
        let v4 = lst.apply(|v| (v >> 3) + &b3);
        let v5 = lst.apply(|v| (v >> 2) + &b4);
        let v6 = lst.apply(|v| (v >> 3) + &b5);
        let v7 = lst.apply(|v| (v >> 5) + &b6);
        let v8 = lst.apply(|v| (v >> 6) + &b7);
        let v9: T = FromWithContext::from_ctx(0.0, ctx);

        let res = lst.apply_zip(&v8, |(v, v8)| v.is_greater(&f8).mux(&v8, &v9));
        let res = lst.apply_zip_2(&v7, &res, |(v, v7, res)| v.is_greater(&f7).mux(&v7, &res));
        let res = lst.apply_zip_2(&v6, &res, |(v, v6, res)| v.is_greater(&f6).mux(&v6, &res));
        let res = lst.apply_zip_2(&v5, &res, |(v, v5, res)| v.is_greater(&f5).mux(&v5, &res));
        let res = lst.apply_zip_2(&v4, &res, |(v, v4, res)| v.is_greater(&f4).mux(&v4, &res));
        let res = lst.apply_zip_2(&v3, &res, |(v, v3, res)| v.is_greater(&f3).mux(&v3, &res));
        let res = lst.apply_zip_2(&v2, &res, |(v, v2, res)| v.is_greater(&f2).mux(&v2, &res));
        let mut res = lst.apply_zip(&res, |(v, res)| v.is_greater(&f1).mux(&v1, &res));

        // res.apply_mut(|v| {
        //     v.optimize_integer_size_ref(1, true);
        // });
        res
    }

    fn activate_and_derivative_multiple(
        &self,
        ctx: &Arc<T::ContextType>,
        lst: Gen1DArray<T, N>,
    ) -> (Gen1DArray<T, N>, Gen1DArray<T, N>) {
        /*
        let sig = <SigmoidMyersAproxAF as ActivationFn<
            FixedPointNumber<SIZE, PRECISION, T, C>,
            C,
            N,
        >>::activate_multiple(&self, ctx, lst);

        let b0: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(1.0, ctx);

        let deriv = sig.apply(|sig_v| {
            let r = (b0.clone() - &sig_v) * sig_v;
            r.optimize_integer_size(1, true)
        });

        (sig, deriv)
        */

        // /*
        let f1: T = FromWithContext::from_ctx(8.0, ctx);
        let f2: T = FromWithContext::from_ctx(4.0, ctx);
        let f3: T = FromWithContext::from_ctx(2.0, ctx);
        let f4: T = FromWithContext::from_ctx(1.0, ctx);
        let f5: T = FromWithContext::from_ctx(-1.0, ctx);
        let f6: T = FromWithContext::from_ctx(-2.0, ctx);
        let f7: T = FromWithContext::from_ctx(-4.0, ctx);
        let f8: T = FromWithContext::from_ctx(-8.0, ctx);

        let b1: T = FromWithContext::from_ctx(0.875, ctx);
        let b2: T = FromWithContext::from_ctx(0.8125, ctx);
        let b3: T = FromWithContext::from_ctx(0.625, ctx);
        let b4: T = FromWithContext::from_ctx(0.5, ctx);
        let b5: T = FromWithContext::from_ctx(0.375, ctx);
        let b6: T = FromWithContext::from_ctx(0.1875, ctx);
        let b7: T = FromWithContext::from_ctx(0.125, ctx);

        let v1: T = FromWithContext::from_ctx(1.0, ctx);
        let v2 = lst.apply(|v| (v >> 6) + &b1);
        let v3 = lst.apply(|v| (v >> 5) + &b2);
        let v4 = lst.apply(|v| (v >> 3) + &b3);
        let v5 = lst.apply(|v| (v >> 2) + &b4);
        let v6 = lst.apply(|v| (v >> 3) + &b5);
        let v7 = lst.apply(|v| (v >> 5) + &b6);
        let v8 = lst.apply(|v| (v >> 6) + &b7);
        let v9: T = FromWithContext::from_ctx(0.0, ctx);

        let comp1 = lst.apply(|v| v.is_greater(&f8));
        let comp2 = lst.apply(|v| v.is_greater(&f7));
        let comp3 = lst.apply(|v| v.is_greater(&f6));
        let comp4 = lst.apply(|v| v.is_greater(&f5));
        let comp5 = lst.apply(|v| v.is_greater(&f4));
        let comp6 = lst.apply(|v| v.is_greater(&f3));
        let comp7 = lst.apply(|v| v.is_greater(&f2));
        let comp8 = lst.apply(|v| v.is_greater(&f1));

        let res = comp1.apply_zip(&v8, |(v, v8)| v.mux(&v8, &v9));
        let res = comp2.apply_zip_2(&v7, &res, |(v, v7, res)| v.mux(&v7, &res));
        let res = comp3.apply_zip_2(&v6, &res, |(v, v6, res)| v.mux(&v6, &res));
        let res = comp4.apply_zip_2(&v5, &res, |(v, v5, res)| v.mux(&v5, &res));
        let res = comp5.apply_zip_2(&v4, &res, |(v, v4, res)| v.mux(&v4, &res));
        let res = comp6.apply_zip_2(&v3, &res, |(v, v3, res)| v.mux(&v3, &res));
        let res = comp7.apply_zip_2(&v2, &res, |(v, v2, res)| v.mux(&v2, &res));
        let mut res = comp8.apply_zip(&res, |(v, res)| v.mux(&v1, &res));

        let d9: T = FromWithContext::from_ctx(0.0, ctx);
        let d8: T = FromWithContext::from_ctx(0.015625, ctx);
        let d7: T = FromWithContext::from_ctx(0.03125, ctx);
        let d6: T = FromWithContext::from_ctx(0.125, ctx);
        let d5: T = FromWithContext::from_ctx(0.25, ctx);

        let der = comp1.apply(|v| v.mux(&d8, &d9));
        let der = comp2.apply_zip(&der, |(v, der)| v.mux(&d7, &der));
        let der = comp3.apply_zip(&der, |(v, der)| v.mux(&d6, &der));
        let der = comp4.apply_zip(&der, |(v, der)| v.mux(&d5, &der));
        let der = comp5.apply_zip(&der, |(v, der)| v.mux(&d6, &der));
        let der = comp6.apply_zip(&der, |(v, der)| v.mux(&d7, &der));
        let der = comp7.apply_zip(&der, |(v, der)| v.mux(&d8, &der));
        let mut der = comp8.apply_zip(&der, |(v, der)| v.mux(&d9, &der));

        // res.apply_mut(|v| v.optimize_integer_size_ref(1, true));
        // der.apply_mut(|v| v.optimize_integer_size_ref(1, true));
        (res, der)
        // */
    }
}

#[derive(Debug, Clone, Copy, Default)]
#[deprecated]
pub struct SigmoidMyersMulAproxAF {}

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C, const N: usize>
    ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N> for SigmoidMyersMulAproxAF
where
    FixedPointNumber<SIZE, PRECISION, T, C>: FromWithContext<f32, C>,
{
    fn activate(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> FixedPointNumber<SIZE, PRECISION, T, C> {
        let f1: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(8.0, ctx);
        let f2: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(4.0, ctx);
        let f3: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(2.0, ctx);
        let f4: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(1.0, ctx);
        let f5: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(-1.0, ctx);
        let f6: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(-2.0, ctx);
        let f7: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(-4.0, ctx);
        let f8: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(-8.0, ctx);

        let m1: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.015625, ctx);
        let m2: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.03125, ctx);
        let m3: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.125, ctx);
        let m4: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.25, ctx);

        let b1: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.875, ctx);
        let b2: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.8125, ctx);
        let b3: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.625, ctx);
        let b4: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.5, ctx);
        let b5: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.375, ctx);
        let b6: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.1875, ctx);
        let b7: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.125, ctx);

        let v1: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(1.0, ctx);
        let v2: FixedPointNumber<SIZE, PRECISION, T, C> = (m1.clone() * &v) + &b1;
        let v3: FixedPointNumber<SIZE, PRECISION, T, C> = (m2.clone() * &v) + &b2;
        let v4: FixedPointNumber<SIZE, PRECISION, T, C> = (m3.clone() * &v) + &b3;
        let v5: FixedPointNumber<SIZE, PRECISION, T, C> = (m4 * &v) + &b4;
        let v6: FixedPointNumber<SIZE, PRECISION, T, C> = (m3 * &v) + &b5;
        let v7: FixedPointNumber<SIZE, PRECISION, T, C> = (m2 * &v) + &b6;
        let v8: FixedPointNumber<SIZE, PRECISION, T, C> = (m1 * &v) + &b7;
        let v9: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.0, ctx);

        let res = v.is_greater(&f8).mux_number(&v8, &v9);
        let res = v.is_greater(&f7).mux_number(&v7, &res);
        let res = v.is_greater(&f6).mux_number(&v6, &res);
        let res = v.is_greater(&f5).mux_number(&v5, &res);
        let res = v.is_greater(&f4).mux_number(&v4, &res);
        let res = v.is_greater(&f3).mux_number(&v3, &res);
        let res: FixedPointNumber<SIZE, PRECISION, T, C> = v.is_greater(&f2).mux_number(&v2, &res);
        let res = v.is_greater(&f1).mux_number(&v1, &res);

        res.optimize_integer_size(1, true)
    }

    fn activate_and_derivative(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> (
        FixedPointNumber<SIZE, PRECISION, T, C>,
        FixedPointNumber<SIZE, PRECISION, T, C>,
    ) {
        let sig = <SigmoidMyersMulAproxAF as ActivationFn<
            FixedPointNumber<SIZE, PRECISION, T, C>,
            N,
        >>::activate(&self, ctx, v);

        let b0: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(1.0, ctx);

        let deriv: FixedPointNumber<SIZE, PRECISION, T, C> = (b0 - &sig) * &sig;

        (sig, deriv.optimize_integer_size(1, true))
    }

    fn activate_multiple(
        &self,
        _ctx: &Arc<C>,
        _lst: Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) -> Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N> {
        todo!()
    }

    fn activate_and_derivative_multiple(
        &self,
        _ctx: &Arc<C>,
        _lst: Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) -> (
        Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
        Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy)]
#[deprecated]
pub struct SigmoidAF {}

#[allow(deprecated)]
impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C, const N: usize>
    ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N> for SigmoidAF
where
    FixedPointNumber<SIZE, PRECISION, T, C>: FromWithContext<f32, C>,
{
    fn activate(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> FixedPointNumber<SIZE, PRECISION, T, C> {
        type FP<T1, C1> = FixedPointNumber<32, 16, T1, C1>;
        let new_v: FP<T, C> = v.update_size();

        let b0: FP<T, C> = FromWithContext::from_ctx(1.0, ctx);
        let b1: FP<T, C> = FromWithContext::from_ctx(f64::log2(std::f64::consts::E) as f32, ctx);

        let exp = (new_v * &b1).not();
        let two_pow = exp.pow_2();
        let div = two_pow + &b0;
        let res = b0 / &div;

        let r = res.update_size();

        r
    }

    fn activate_and_derivative(
        &self,
        ctx: &Arc<C>,
        v: FixedPointNumber<SIZE, PRECISION, T, C>,
    ) -> (
        FixedPointNumber<SIZE, PRECISION, T, C>,
        FixedPointNumber<SIZE, PRECISION, T, C>,
    ) {
        let sig = <SigmoidAF as ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N>>::activate(
            &self, ctx, v,
        );

        let b0: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(1.0, ctx);

        let deriv = (b0 - &sig) * &sig;
        (sig, deriv)
    }

    fn activate_multiple(
        &self,
        ctx: &Arc<C>,
        lst: Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N>,
    ) -> Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, N> {
        lst.apply(|v| {
            <SigmoidAF as ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N>>::activate(
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
        lst.apply_two(|v| <SigmoidAF as ActivationFn<FixedPointNumber<SIZE, PRECISION, T, C>, N>>::activate_and_derivative(self, ctx, v.clone()))
    }
}

#[allow(deprecated)]
#[cfg(test)]
mod test {
    use super::{SigmoidAF, SigmoidLSTruncAproxAF, SigmoidMyersAproxAF, SigmoidPowAproxAF};
    use crate::ActivationFn;
    use numbers::{FixedPointNumber, FromWithContext, Gen1DArray};
    use std::{f32::consts::E, sync::Arc};

    #[test]
    fn test_mayersaproximation() {
        let ctx = Arc::new(());
        let af = SigmoidMyersAproxAF::default();

        let threathold = 0.1;

        let test = |x: f32| {
            let v: FixedPointNumber<64, 32, bool, ()> = FromWithContext::from_ctx(x, &ctx);
            let res = <SigmoidMyersAproxAF as ActivationFn<
                FixedPointNumber<64, 32, bool, ()>,
                1,
            >>::activate_and_derivative(&af, &ctx, v);

            let rec_y: f32 = res.0.into();
            let rec_der_y: f32 = res.1.into();
            let exp = 1.0 / (1.0 + (-x).exp());
            let exp_der = exp * (1.0 - exp);

            assert!(
                (exp - rec_y).abs() <= threathold,
                "x: {}, y: {} (expected: {}), diff = {} > {}",
                x,
                rec_y,
                exp,
                (exp - rec_y).abs(),
                threathold
            );
            assert!(
                (exp_der - rec_der_y).abs() <= threathold,
                "Der x: {}, y: {} (expected: {}), diff = {} > {}",
                x,
                rec_der_y,
                exp_der,
                (exp_der - rec_der_y).abs(),
                threathold
            );
        };

        for v in -10..10 {
            let i: f32 = v as f32;
            test(i + 0.0);
            test(i + 0.25);
            test(i + 0.5);
            test(i + 0.75);
        }
    }

    #[test]
    fn test_mayersaproximation_multiple() {
        let ctx = Arc::new(());
        let af = SigmoidMyersAproxAF::default();

        let threathold = 0.1;

        let test = |rec_y: (f32, f32), x: f32| {
            let exp = 1.0 / (1.0 + (-x).exp());
            let exp_der = exp * (1.0 - exp);

            assert!(
                (exp - rec_y.0).abs() <= threathold,
                "x: {}, y: {} (expected: {}), diff = {} > {}",
                x,
                rec_y.0,
                exp,
                (exp - rec_y.0).abs(),
                threathold
            );
            assert!(
                (exp_der - rec_y.1).abs() <= threathold,
                "Der x: {}, y: {} (expected: {}), diff = {} > {}",
                x,
                rec_y.1,
                exp_der,
                (exp_der - rec_y.1).abs(),
                threathold
            );
        };

        let mut lst: Vec<f32> = Vec::with_capacity(20 * 4);
        for v in -10..10 {
            let i: f32 = v as f32;
            lst.push(i + 0.0);
            lst.push(i + 0.25);
            lst.push(i + 0.5);
            lst.push(i + 0.75);
        }

        let vals: Gen1DArray<f32, 80> = Gen1DArray::from_ctx(lst.clone(), &ctx);
        let vals: Gen1DArray<FixedPointNumber<64, 32, bool, ()>, 80> =
            vals.apply(|x| FromWithContext::from_ctx(*x, &ctx));
        let (act, deriv) = <SigmoidMyersAproxAF as ActivationFn<
            FixedPointNumber<64, 32, bool, ()>,
            80,
        >>::activate_and_derivative_multiple(&af, &ctx, vals);

        let act_slice = act.as_slice();
        let act_lst: Vec<_> = act_slice.iter().map(|v| v.iter()).flatten().collect();

        let deriv_slice = deriv.as_slice();
        let deriv_lst: Vec<_> = deriv_slice.iter().map(|v| v.iter()).flatten().collect();
        for (y, exp) in act_lst.iter().zip(deriv_lst.iter()).zip(lst.iter()) {
            let rec_y: (f32, f32) = ((**y.0).clone().into(), (**y.1).clone().into());
            test(rec_y, *exp);
        }
    }

    #[test]
    fn test_aproximation() {
        let ctx = Arc::new(());
        let af = SigmoidLSTruncAproxAF::default_size();

        let test = |x: f32, y: f32, threathold: f32| {
            let v: FixedPointNumber<64, 32, bool, ()> = FromWithContext::from_ctx(x, &ctx);
            let res = <SigmoidLSTruncAproxAF<16, 10> as ActivationFn<
                FixedPointNumber<64, 32, bool, ()>,
                1,
            >>::activate(&af, &ctx, v);

            let rec_y: f32 = res.into();

            // println!("{}", rec_y);
            assert!(
                (y - rec_y).abs() <= threathold,
                "x: {}, y: {} (expected: {}), diff = {} > {}",
                x,
                rec_y,
                y,
                (y - rec_y).abs(),
                threathold
            );
        };

        test(00.0, 0.500, 0.04);
        test(01.0, 0.709, 0.04);
        test(-1.0, 0.269, 0.04);
        test(02.0, 0.873, 0.04);
        test(-2.0, 0.119, 0.04);
        test(03.0, 0.953, 0.04);
        test(-3.0, 0.047, 0.04);
        test(04.0, 0.982, 0.04);
        test(-4.0, 0.018, 0.04);
        test(05.0, 0.993, 0.04);
        test(-5.0, 0.007, 0.04);
    }

    #[test]
    fn test_aproximationv2() {
        let ctx = Arc::new(());
        let af = SigmoidPowAproxAF::default_size();

        let test = |x: f32, y: f32, threathold: f32| {
            let v: FixedPointNumber<64, 32, bool, ()> = FromWithContext::from_ctx(x, &ctx);
            let res = <SigmoidPowAproxAF<16, 10> as ActivationFn<
                FixedPointNumber<64, 32, bool, ()>,
                1,
            >>::activate(&af, &ctx, v);

            let rec_y: f32 = res.into();

            let y = 1.0 / (1.0 + E.powf(x * -1.0));

            println!("{:02}: Rec {:.04} Exp {:.04}", x, rec_y, y);
            // println!("{}", rec_y);
            // assert!(
            //     (y - rec_y).abs() <= threathold,
            //     "x: {}, y: {} (expected: {}), diff = {} > {}",
            //     x,
            //     rec_y,
            //     y,
            //     (y - rec_y).abs(),
            //     threathold
            // );
        };

        let mut v = -5.0;
        while v < 5.1 {
            test(v, 0.007, 0.04);
            v += 0.25;
        }
    }

    #[test]
    fn test_real() {
        let ctx = Arc::new(());
        let af = SigmoidAF {};

        let test = |x: f32, _y: f32, _threathold: f32| {
            let v: FixedPointNumber<64, 32, bool, ()> = FromWithContext::from_ctx(x, &ctx);
            let res = <SigmoidAF as ActivationFn<FixedPointNumber<64, 32, bool, ()>, 1>>::activate(
                &af, &ctx, v,
            );

            let rec_y: f32 = res.into();

            println!("{}", rec_y);

            // assert!(
            //     (y - rec_y).abs() <= threathold,
            //     "x: {}, y: {} (expected: {}), diff = {} > {}",
            //     x,
            //     rec_y,
            //     y,
            //     (y - rec_y).abs(),
            //     threathold
            // );
        };

        test(-5.0, 0.007, 0.05);
        test(-4.0, 0.018, 0.05);
        test(-3.0, 0.047, 0.05);
        test(-2.0, 0.119, 0.05);
        test(-1.0, 0.269, 0.05);
        test(00.0, 0.500, 0.05);
        test(01.0, 0.709, 0.05);
        test(02.0, 0.873, 0.05);
        test(03.0, 0.953, 0.05);
        test(04.0, 0.982, 0.05);
        test(05.0, 0.993, 0.05);
    }
}
