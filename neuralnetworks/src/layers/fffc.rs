use numbers::{
    AddContext, BooleanType, FixedPointNumber, FixedPointNumberNoContext, FromWithContext,
    Gen1DArray, Gen2DArray, Gen2DArrayNoContext,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::LinkedList,
    sync::{Arc, Mutex},
};

use super::{ErrorBackpropagation, Layer, LayerOutput, LayerUpdate};
use crate::{metrics::LossFunction, profiling::evaluate, ActivationFn};

pub type BoolFFFFLayer<
    const BITS: usize,
    const CALC_BITS: usize,
    const PRECISION: usize,
    const PREV_N: usize,
    const CURR_N: usize,
    AF,
    LOSS,
> = FeedForwardFullyConnectedLayer<bool, (), BITS, CALC_BITS, PRECISION, PREV_N, CURR_N, AF, LOSS>;

pub struct FeedForwardFullyConnectedLayer<
    T: BooleanType<C>,
    C,
    const BITS: usize,
    const CALC_BITS: usize,
    const PRECISION: usize,
    const PREV_N: usize,
    const CURR_N: usize,
    AF: ActivationFn<FixedPointNumber<CALC_BITS, PRECISION, T, C>, C, CURR_N>,
    LOSS: LossFunction<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
> {
    name: String,
    b: Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
    weights: Gen2DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, PREV_N, CURR_N>,
    activation_fn: AF,
    loss_fn: LOSS,
    weights_updates: Arc<
        Mutex<
            LinkedList<FFWeightUpdate<FixedPointNumber<BITS, PRECISION, T, C>, C, PREV_N, CURR_N>>,
        >,
    >,
}

impl<
        T: BooleanType<C>,
        C,
        const PREV_N: usize,
        const CURR_N: usize,
        const BITS: usize,
        const CALC_BITS: usize,
        const PRECISION: usize,
        AF: ActivationFn<FixedPointNumber<CALC_BITS, PRECISION, T, C>, C, CURR_N>,
        LOSS: LossFunction<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
    > FeedForwardFullyConnectedLayer<T, C, BITS, CALC_BITS, PRECISION, PREV_N, CURR_N, AF, LOSS>
{
    pub fn with_weights(
        weights: [[FixedPointNumber<BITS, PRECISION, T, C>; CURR_N]; PREV_N],
        b: [FixedPointNumber<BITS, PRECISION, T, C>; CURR_N],
        activation_fn: AF,
        loss_fn: LOSS,
        ctx: &Arc<C>,
    ) -> Self {
        Self {
            name: "unnamed".to_owned(),
            weights: Gen2DArray::from_array(weights, ctx),
            b: Gen2DArray::from_array([b], ctx),
            activation_fn,
            loss_fn,
            weights_updates: Arc::new(Mutex::new(LinkedList::default())),
        }
    }
}

#[cfg(feature = "rand")]
impl<
        T: BooleanType<C>,
        C,
        const PREV_N: usize,
        const CURR_N: usize,
        const BITS: usize,
        const CALC_BITS: usize,
        const PRECISION: usize,
        AF: ActivationFn<FixedPointNumber<CALC_BITS, PRECISION, T, C>, C, CURR_N>,
        LOSS: LossFunction<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
    > FeedForwardFullyConnectedLayer<T, C, BITS, CALC_BITS, PRECISION, PREV_N, CURR_N, AF, LOSS>
{
    pub fn new_random(
        rng: &mut dyn rand::RngCore,
        activation_fn: AF,
        loss_fn: LOSS,
        ctx: &Arc<C>,
    ) -> Self {
        use rand::distributions::Distribution;
        let dist = rand_distr::Normal::new(0.0, 1.0).expect("Should create distribution");

        let w = Gen2DArray::random(
            rng,
            |rng| {
                let v: f32 = dist.sample(rng);
                FixedPointNumber::from_ctx(v, ctx)
            },
            ctx,
        );
        let b = Gen1DArray::random(
            rng,
            |rng| {
                let v: f32 = dist.sample(rng);
                FixedPointNumber::from_ctx(v, ctx)
            },
            ctx,
        );

        Self::with_weights(w.as_array(), b.as_1d_array(), activation_fn, loss_fn, ctx)
    }
}

impl<
        T: 'static + BooleanType<C> + Send + Sync + AddContext<C>,
        const PREV_N: usize,
        const CURR_N: usize,
        const BITS: usize,
        const CALC_BITS: usize,
        const PRECISION: usize,
        AF: ActivationFn<FixedPointNumber<CALC_BITS, PRECISION, T, C>, C, CURR_N>,
        LOSS: LossFunction<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
        C: 'static + Send + Sync,
    > Layer<FixedPointNumber<BITS, PRECISION, T, C>, C, PREV_N, CURR_N>
    for FeedForwardFullyConnectedLayer<T, C, BITS, CALC_BITS, PRECISION, PREV_N, CURR_N, AF, LOSS>
where
    T: Serialize,
    T::FromType: DeserializeOwned,
{
    fn update_name(&mut self, name: &str) {
        self.name = name.to_owned();
    }
    fn execute_layer(
        &self,
        ctx: &Arc<C>,
        arr: &Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, PREV_N>,
    ) -> Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N> {
        evaluate!(
            &self.name,
            "ExecuteLayer",
            arr.multiply_add_execute(&self.weights, &self.b, |v| {
                evaluate!(
                    &self.name,
                    "ActivateMultiple",
                    self.activation_fn
                        .activate_multiple(ctx, v)
                        .apply(|v| v.update_size())
                )
            })
        )
    }

    fn pre_train(
        &self,
        ctx: &Arc<C>,
        input: &Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, PREV_N>,
    ) -> (
        Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
        Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
    ) {
        let r = evaluate!(
            &self.name,
            "PreTrain",
            input.multiply_add_execute(&self.weights, &self.b, |v| {
                evaluate!(
                    &self.name,
                    "ActivateDerivMultiple",
                    self.activation_fn.activate_and_derivative_multiple(ctx, v)
                )
            })
        );

        (
            r.0.apply(|v| v.update_size()),
            r.1.apply(|v| v.update_size()),
        )
    }

    fn train(
        &self,
        _ctx: &Arc<C>,
        input: &Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, PREV_N>,
        received: (
            Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
            Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
        ),
        expected: super::LayerOutput<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
        learning_rate: &isize,
    ) -> ErrorBackpropagation<FixedPointNumber<BITS, PRECISION, T, C>, C, PREV_N> {
        let (apply_af, derivated_af): (
            Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
            Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N>,
        ) = received;

        let delta = match expected {
            LayerOutput::FinalOutput(exp) => {
                let loss = evaluate!(
                    &self.name,
                    "CostDerivFinal",
                    self.loss_fn.cost_derivative(apply_af, &exp)
                );
                let error = evaluate!(
                    &self.name,
                    "CostDerivProdFinal",
                    loss.hadamard_prod(&derivated_af)
                );
                error
            }
            LayerOutput::Intermediate(error) => {
                /* Section: A */
                let error = evaluate!(
                    &self.name,
                    "CostDerivProdIntermediate",
                    error.hadamard_prod(&derivated_af)
                );
                error
            }
        };

        /* Calculate learning rate */
        let lr = learning_rate.abs() as usize;
        let lr_delta = evaluate!(
            &self.name,
            "DeltaLr",
            match learning_rate > &0 {
                true => delta.apply(|v| v << lr),
                false => delta.apply(|v| v >> lr),
            }
        );
        let input_trans = evaluate!(&self.name, "NeurChangeTrans", input.transpose());
        let neurons_change: Gen2DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, PREV_N, CURR_N> =
            evaluate!(&self.name, "NeurChange", &input_trans * &lr_delta);

        // self.b = &self.b - &lr_delta;
        // self.weights = &self.weights - &neurons_change;

        let update = FFWeightUpdate {
            b: lr_delta,
            weights: neurons_change,
        };

        evaluate!(&self.name, "WeightUpdate", {
            let mut updates = self
                .weights_updates
                .lock()
                .expect("Should be able to lock the mutex");
            updates.push_back(update);
        });

        /* START: Should be in section 'A', but due to generics restrictions must be calculated before being sent */
        let curr_layer_error: Box<
            dyn FnOnce() -> Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, PREV_N>,
        > = {
            let weights_trans = self.weights.transpose();
            let _name = self.name.to_string();

            Box::new(move || evaluate!(&_name, "DeltaMult", delta.mul_opt_rhs(weights_trans)))
        };
        /* END */

        ErrorBackpropagation {
            error: curr_layer_error,
        }
    }

    fn update_weights(&mut self, _ctx: &Arc<C>) {
        let mut lst = self
            .weights_updates
            .lock()
            .expect("Should be able to lock mutex");

        // let len: FixedPointNumber<BITS, PRECISION, T, C> = FromWithContext::from_ctx(lst.len() as f32, ctx);

        let avg_weights: Gen2DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, PREV_N, CURR_N> = evaluate!(
            &self.name,
            "WeightUpdateWAvg",
            Gen2DArray::fold_add(lst.iter().map(|v| &v.weights))
        );
        let avg_b: Gen1DArray<FixedPointNumber<BITS, PRECISION, T, C>, C, CURR_N> = evaluate!(
            &self.name,
            "WeightUpdateBAvg",
            Gen2DArray::fold_add(lst.iter().map(|v| &v.b))
        );

        lst.clear();

        // let change = learning_rate / &len;

        // avg_weights /= &len;
        // avg_b /= &len;

        evaluate!(&self.name, "WeightUpdateBSub", self.b -= &avg_b);
        evaluate!(&self.name, "WeightUpdateWSub", self.weights -= &avg_weights);
    }

    fn get_weights(&self) -> Vec<u8> {
        let mut b_lst: Vec<u8> = vec![];
        let mut weights_lst: Vec<u8> = vec![];

        ciborium::into_writer(&self.b, &mut b_lst).expect("Should be able to serialize b");
        ciborium::into_writer(&self.weights, &mut weights_lst)
            .expect("Should be able to serialize weights");

        let v = FFFFWeights {
            b: b_lst,
            weights: weights_lst,
        };
        let val = serde_json::to_vec(&v).expect("Should serialize FFFFWeights");
        val
    }
    fn set_weights(&mut self, ctx: &Arc<C>, weights: Vec<u8>) {
        let v: FFFFWeights =
            serde_json::from_slice(weights.as_slice()).expect("Should deserialize FFFFWeights");

        let b: Gen2DArrayNoContext<
            FixedPointNumberNoContext<BITS, PRECISION, T::FromType>,
            1,
            CURR_N,
        > = ciborium::from_reader(v.b.as_slice()).expect("Should deserialize b");
        let weights: Gen2DArrayNoContext<
            FixedPointNumberNoContext<BITS, PRECISION, T::FromType>,
            PREV_N,
            CURR_N,
        > = ciborium::from_reader(v.weights.as_slice()).expect("Should deserialize weights");

        self.b = AddContext::add_context(&b, &ctx);
        self.weights = AddContext::add_context(&weights, &ctx);
    }
}

#[derive(Serialize, Deserialize)]
struct FFFFWeights {
    b: Vec<u8>,
    weights: Vec<u8>,
}

struct FFWeightUpdate<T, C, const PREV_N: usize, const CURR_N: usize> {
    b: Gen1DArray<T, C, CURR_N>,
    weights: Gen2DArray<T, C, PREV_N, CURR_N>,
}
impl<T, C, const PREV_N: usize, const CURR_N: usize> LayerUpdate
    for FFWeightUpdate<T, C, PREV_N, CURR_N>
{
}

/*
#[cfg(test)]
mod test {
    use numbers::Gen1DArray;
    use std::sync::Arc;

    use crate::{
        layers::{FeedForwardFullyConnectedLayer, Layer, LayerOutput},
        metrics::QuadLossFunction,
        ActivationFn,
    };

    #[derive(Debug)]
    struct CustomAF {}
    impl<const N: usize> ActivationFn<f32, (), N> for CustomAF {
        fn activate(&self, _: &Arc<()>, v: &f32) -> f32 {
            *v
        }

        fn activate_and_derivative(&self, _: &Arc<()>, v: &f32) -> (f32, f32) {
            (*v, 1.0)
        }
    }

    #[test]
    fn test_train_correct() {
        let ctx = Arc::new(());
        let af = CustomAF {};
        let qloss = QuadLossFunction {};

        let layer =
            FeedForwardFullyConnectedLayer::with_weights([[1.0], [0.0]], [-1.0], af, qloss, &ctx);

        let input = Gen1DArray::from_array([[4.0, 2.0]], &ctx);
        let output = Gen1DArray::from_array([[3.0]], &ctx);

        let v = layer.pre_train(&ctx, &input);
        let error = layer.train(&ctx, &input, v, LayerOutput::FinalOutput(output), &1.0);

        assert_eq!(error.error.length(), 0.0);
    }

    #[test]
    fn test_train_incorrect_output() {
        let ctx = Arc::new(());
        let af = CustomAF {};
        let qloss = QuadLossFunction {};

        let layer =
            FeedForwardFullyConnectedLayer::with_weights([[0.5], [0.0]], [0.2], af, qloss, &ctx);

        let input = Gen1DArray::from_array([[1.0, -1.0]], &ctx);
        let output = Gen1DArray::from_array([[1.0]], &ctx);

        let v = layer.pre_train(&ctx, &input);
        let error1 = layer.train(&ctx, &input, v, LayerOutput::FinalOutput(output), &0.1);
        let error1 = error1.error.length();
        println!(
            "Weights {:?}, b {:?}, error: {}",
            layer.weights, layer.b, error1
        );

        let v = layer.pre_train(&ctx, &input);
        let output = Gen1DArray::from_array([[1.0]], &ctx);
        let error2 = layer.train(&ctx, &input, v, LayerOutput::FinalOutput(output), &0.1);
        let error2 = error2.error.length();
        println!(
            "Weights {:?}, b {:?}, error: {}",
            layer.weights, layer.b, error2
        );

        let v = layer.pre_train(&ctx, &input);
        let output = Gen1DArray::from_array([[1.0]], &ctx);
        let error3 = layer.train(&ctx, &input, v, LayerOutput::FinalOutput(output), &0.1);
        let error3 = error3.error.length();
        println!(
            "Weights {:?}, b {:?}, error: {}",
            layer.weights, layer.b, error3
        );

        assert!(error1 > error2);
        assert!(error2 > error3);
    }

    #[test]
    fn test_train_incorrect_intermidiate() {
        let ctx = Arc::new(());
        let af = CustomAF {};
        let qloss = QuadLossFunction {};

        let layer =
            FeedForwardFullyConnectedLayer::with_weights([[0.5], [0.0]], [0.2], af, qloss, &ctx);

        let input = Gen1DArray::from_array([[1.0, -1.0]], &ctx);

        let v = layer.pre_train(&ctx, &input);
        let error1 = layer.train(
            &ctx,
            &input,
            v,
            LayerOutput::Intermidiate(Gen1DArray::from_array([[0.5]], &ctx)),
            &0.1,
        );
        let error1 = error1.error.length();
        println!(
            "Weights {:?}, b {:?}, error: {}",
            layer.weights, layer.b, error1
        );

        let v = layer.pre_train(&ctx, &input);
        let error2 = layer.train(
            &ctx,
            &input,
            v,
            LayerOutput::Intermidiate(Gen1DArray::from_array([[-0.5]], &ctx)),
            &0.1,
        );
        let error2 = error2.error.length();
        println!(
            "Weights {:?}, b {:?}, error: {}",
            layer.weights, layer.b, error2
        );

        let v = layer.pre_train(&ctx, &input);
        let error3 = layer.train(
            &ctx,
            &input,
            v,
            LayerOutput::Intermidiate(Gen1DArray::from_array([[-0.4]], &ctx)),
            &0.1,
        );
        let error3 = error3.error.length();
        println!(
            "Weights {:?}, b {:?}, error: {}",
            layer.weights, layer.b, error3
        );

        assert!(error1 > error2);
        assert!(error2 > error3);
    }
}
 */
