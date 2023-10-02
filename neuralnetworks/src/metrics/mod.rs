use numbers::Gen1DArray;
use std::{ops::SubAssign, sync::Arc};

mod fixedpoint;
pub use fixedpoint::*;

pub trait Metric<T, C, const N: usize>: Sync + Send {
    fn calc_metric(
        &self,
        ctx: &Arc<C>,
        lst: &[(&Gen1DArray<T, C, N>, &Gen1DArray<T, C, N>)],
    ) -> TrainMetric<T>;
}

pub trait LossFunction<T, C, const SIZE: usize>: Sync + Send {
    fn cost_derivative(
        &self,
        received: Gen1DArray<T, C, SIZE>,
        expected: &Gen1DArray<T, C, SIZE>,
    ) -> Gen1DArray<T, C, SIZE>;
}

#[derive(Clone, Copy)]
pub struct QuadLossFunction {}

impl<T, C, const SIZE: usize> LossFunction<T, C, SIZE> for QuadLossFunction
where
    for<'a> T: SubAssign<&'a T>,
{
    fn cost_derivative(
        &self,
        mut received: Gen1DArray<T, C, SIZE>,
        expected: &Gen1DArray<T, C, SIZE>,
    ) -> Gen1DArray<T, C, SIZE> {
        received -= expected;
        received
    }
}

pub use metrics::*;

use crate::TrainMetric;
mod metrics {
    use std::{
        ops::{Add, Div, Mul, SubAssign},
        sync::Arc,
    };

    use numbers::{Abs, DefaultWithContext, FromWithContext, Gen1DArray, IsGreater};

    use crate::TrainMetric;

    use super::Metric;

    #[derive(Default)]
    pub struct MeanSquareErrorMetric {}
    impl<T, C> Metric<T, C, 1> for MeanSquareErrorMetric
    where
        T: Clone,
        T: FromWithContext<f32, C>,
        T: DefaultWithContext<C>,
        for<'a> T: SubAssign<&'a T>,
        for<'a> T: Add<&'a T, Output = T>,
        for<'a> &'a T: Div<&'a T, Output = T>,
        for<'a> &'a T: Mul<&'a T, Output = T>,
    {
        fn calc_metric(
            &self,
            ctx: &Arc<C>,
            lst: &[(&Gen1DArray<T, C, 1>, &Gen1DArray<T, C, 1>)],
        ) -> TrainMetric<T> {
            let first_size: f32 = (lst.len() as u16).try_into().expect("Should convert");
            let first_size: T = FromWithContext::from_ctx(first_size, &ctx);

            let r = lst
                .into_iter()
                .map(|(exp, reac)| {
                    let mut res: Gen1DArray<T, C, 1> = (*exp).clone();
                    res -= *reac;

                    let lst: Vec<&T> = (&res).into();
                    &(*lst.get(0).unwrap() * *lst.get(0).unwrap()) / &first_size
                })
                .fold(T::default_ctx(ctx), |a, b| a + &b);

            TrainMetric::new("MSE", r)
        }
    }

    #[derive(Default)]
    pub struct AccuracyMetric {}

    impl<T: Clone, C> Metric<T, C, 1> for AccuracyMetric
    where
        T: FromWithContext<f32, C> + IsGreater,
        for<'a> T: Add<&'a T, Output = T>,
        for<'a> T: SubAssign<&'a T>,
        for<'a> &'a T: Div<&'a T, Output = T>,
        for<'a> T: Abs<Output = T>,
    {
        fn calc_metric(
            &self,
            ctx: &Arc<C>,
            lst: &[(&Gen1DArray<T, C, 1>, &Gen1DArray<T, C, 1>)],
        ) -> TrainMetric<T> {
            let zero: T = FromWithContext::from_ctx(0.0, ctx);
            let point_five: T = FromWithContext::from_ctx(0.5, ctx);
            let total_size: T = FromWithContext::from_ctx(lst.len() as f32, ctx);

            let correct_count: T = lst
                .into_iter()
                .map(|(exp, reac)| {
                    let mut diff: Gen1DArray<T, C, 1> = (**exp).clone();
                    diff -= *reac;

                    let res: Vec<&T> = (&diff).into();
                    let res_val: &T = res.get(0).unwrap();
                    let res_val: T = res_val.clone().abs();

                    point_five.is_greater(&res_val)
                })
                .fold::<T, _>(zero, |a: T, b: T| a + &b);

            let res: T = &correct_count / &total_size;

            TrainMetric::new("Accuracy", res)
        }
    }

    // pub struct MultiClassAccuracyMetric {}
    // impl<T: DefaultWithContext<C>, C, const SIZE: usize> Metric<T, C, SIZE>
    //     for MultiClassAccuracyMetric
    // {
    //     fn calc_metric(
    //         &self,
    //         ctx: &Arc<C>,
    //         lst: &[(&Gen1DArray<T, C, SIZE>, &Gen1DArray<T, C, SIZE>)],
    //     ) -> TrainMetric<T> {
    //         let mut tp_count: T = DefaultWithContext::default_ctx(ctx);
    //         let mut fn_count: T = DefaultWithContext::default_ctx(ctx);
    //         let mut fp_count: T = DefaultWithContext::default_ctx(ctx);
    //         let mut tn_count: T = DefaultWithContext::default_ctx(ctx);

    //         todo!()
    //     }
    // }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use numbers::Gen1DArray;

    use crate::{
        metrics::{AccuracyMetric, Metric},
        TrainMetric,
    };

    #[test]
    fn test_accuracy() {
        let ctx = Arc::new(());
        let acc: Box<dyn Metric<f32, (), 1>> = Box::new(AccuracyMetric {});

        let assert_acc = |input: &[f32], output: &[f32], accuracy: f32| {
            let lst: Vec<(Gen1DArray<f32, (), 1>, Gen1DArray<f32, (), 1>)> = input
                .iter()
                .zip(output.iter())
                .map(|(i, o)| {
                    (
                        Gen1DArray::<f32, (), 1>::from_array([[*i]], &ctx),
                        Gen1DArray::<f32, (), 1>::from_array([[*o]], &ctx),
                    )
                })
                .collect();
            let lst1: Vec<(&Gen1DArray<f32, (), 1>, &Gen1DArray<f32, (), 1>)> =
                lst.iter().map(|v| (&v.0, &v.1)).collect();

            let acc: TrainMetric<f32> = acc.calc_metric(&ctx, lst1.as_slice());

            assert_eq!(acc.value, accuracy);
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
