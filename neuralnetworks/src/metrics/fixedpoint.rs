use std::{marker::PhantomData, sync::Arc};

use super::{Metric, TrainMetric};
use numbers::{
    Abs, BooleanType, DefaultWithContext, FixedPointNumber, FromWithContext, Gen1DArray, IsGreater,
};

pub struct FixedPrecisionAccuracyMetric<const SIZE: usize, const PRECISION: usize>(
    PhantomData<[[usize; SIZE]; PRECISION]>,
);
impl<const SIZE: usize, const PRECISION: usize> FixedPrecisionAccuracyMetric<SIZE, PRECISION> {
    pub fn new() -> Self {
        Self(PhantomData::default())
    }
}
impl<
        const SIZE: usize,
        const PRECISION: usize,
        T: BooleanType<C>,
        C,
        const CALC_SIZE: usize,
        const CALC_PRECISION: usize,
    > Metric<FixedPointNumber<SIZE, PRECISION, T, C>, 1>
    for FixedPrecisionAccuracyMetric<CALC_SIZE, CALC_PRECISION>
{
    fn calc_metric(
        &self,
        ctx: &Arc<C>,
        lst: &[(
            &Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, 1>,
            &Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, 1>,
        )],
    ) -> TrainMetric<FixedPointNumber<SIZE, PRECISION, T, C>> {
        let point_five: FixedPointNumber<SIZE, PRECISION, T, C> =
            FromWithContext::from_ctx(0.5, ctx);

        let zero: FixedPointNumber<SIZE, PRECISION, T, C> = FromWithContext::from_ctx(0.0, ctx);
        let total_size: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            FromWithContext::from_ctx(lst.len() as f32, ctx);

        let zero: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> = zero.update_size();
        let total_size: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> =
            total_size.update_size();

        let correct_count: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> = lst
            .into_iter()
            .map(|(exp, reac)| {
                let mut diff: Gen1DArray<_, 1> = (*exp).clone();
                diff -= *reac;

                let res: Vec<&FixedPointNumber<SIZE, PRECISION, T, C>> = (&diff).into();
                let res_val: &FixedPointNumber<SIZE, PRECISION, T, C> = res.get(0).unwrap();
                let res_val: FixedPointNumber<SIZE, PRECISION, T, C> = res_val.clone().abs();

                point_five.is_greater(&res_val).update_size()
            })
            .fold(zero, |a, b| a + &b);

        let res: FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C> = &correct_count / &total_size;

        TrainMetric::new("Accuracy", res.update_size())
    }
}

pub struct FixedPrecisionMeanSquareErrorMetric<const SIZE: usize, const PRECISION: usize>(
    PhantomData<[[usize; SIZE]; PRECISION]>,
);
impl<const SIZE: usize, const PRECISION: usize>
    FixedPrecisionMeanSquareErrorMetric<SIZE, PRECISION>
{
    pub fn new() -> Self {
        Self(PhantomData::default())
    }
}
impl<
        const SIZE: usize,
        const PRECISION: usize,
        T: BooleanType<C>,
        C,
        const CALC_SIZE: usize,
        const CALC_PRECISION: usize,
    > Metric<FixedPointNumber<SIZE, PRECISION, T, C>, 1>
    for FixedPrecisionMeanSquareErrorMetric<CALC_SIZE, CALC_PRECISION>
{
    fn calc_metric(
        &self,
        ctx: &Arc<C>,
        lst: &[(
            &Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, 1>,
            &Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, 1>,
        )],
    ) -> TrainMetric<FixedPointNumber<SIZE, PRECISION, T, C>> {
        let total_size: FixedPointNumber<22, 0, T, C> =
            FromWithContext::from_ctx(lst.len() as f32, ctx);

        let r = lst
            .into_iter()
            .map(|(exp, reac)| {
                let mut res: Gen1DArray<FixedPointNumber<SIZE, PRECISION, T, C>, 1> =
                    (*exp).clone();
                res -= *reac;

                let lst: Vec<&_> = (&res).into();
                let r = *lst.get(0).unwrap() * *lst.get(0).unwrap();
                r.update_size()
            })
            .fold::<FixedPointNumber<CALC_SIZE, CALC_PRECISION, T, C>, _>(
                DefaultWithContext::default_ctx(ctx),
                |a, b| a + &b,
            );

        let r = &r / &total_size.update_size();

        TrainMetric::new("MSE", r.update_size())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use super::super::{AccuracyMetric, Metric, TrainMetric};
    use numbers::{BoolFixedPointNumber, FromWithContext, Gen1DArray};

    #[test]
    fn test_accuracy() {
        type Number = BoolFixedPointNumber<16, 8>;

        let ctx = Arc::new(());
        let acc: Box<dyn Metric<Number, 1>> = Box::new(AccuracyMetric {});

        let assert_acc = |input: &[f32], output: &[f32], accuracy: f32| {
            let lst: Vec<(Gen1DArray<Number, 1>, Gen1DArray<Number, 1>)> = input
                .iter()
                .zip(output.iter())
                .map(|(i, o)| {
                    let v1 = FromWithContext::from_ctx(*i, &Arc::new(()));
                    let v2 = FromWithContext::from_ctx(*o, &Arc::new(()));

                    (
                        Gen1DArray::<Number, 1>::from_array([[v1]], &ctx),
                        Gen1DArray::<Number, 1>::from_array([[v2]], &ctx),
                    )
                })
                .collect();
            let lst1: Vec<(&Gen1DArray<Number, 1>, &Gen1DArray<Number, 1>)> =
                lst.iter().map(|v| (&v.0, &v.1)).collect();

            let acc: TrainMetric<Number> = acc.calc_metric(&ctx, lst1.as_slice());
            let acc_plaintext: f32 = acc.value.into();

            assert_eq!(acc_plaintext, accuracy);
        };

        let expected = [1.0, 1.0, 0.0, 0.0];

        let received_100 = [1.0, 1.0, 0.0, 0.0];
        let received_50 = [1.0, 0.0, 1.0, 0.0];
        let received_0 = [0.0, 0.0, 1.0, 1.0];
        let received_100_frac = [0.51, 1.49, -0.49, 0.49];

        assert_acc(&expected, &received_100, 1.0);
        assert_acc(&expected, &received_50, 0.5);
        assert_acc(&expected, &received_0, 0.0);
        assert_acc(&expected, &received_100_frac, 1.0);
    }
}
