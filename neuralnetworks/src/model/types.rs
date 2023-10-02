use std::sync::Arc;

use crate::{
    layers::{Layer, LayerOutput},
    ModelWeights,
};
use base64::{engine::general_purpose, Engine as _};
use numbers::{FromWithContext, Gen1DArray};

pub(super) struct BackpropagationData<T> {
    pub output: Vec<T>,
    pub error: Box<dyn FnOnce() -> Vec<T>>,
}

pub(super) enum NextLayer<T, C, const INPUT_N: usize, const OUTPUT_N: usize> {
    Layer(Arc<dyn CurrentLayer<T, C, INPUT_N, OUTPUT_N>>),
    Finish(Box<fn(Gen1DArray<T, C, INPUT_N>) -> Gen1DArray<T, C, OUTPUT_N>>),
}
pub(super) trait CurrentLayer<T: 'static + Clone, C, const N1: usize, const N3: usize>:
    Sync + Send
{
    fn execute(&self, ctx: &Arc<C>, input: &Gen1DArray<T, C, N1>) -> Gen1DArray<T, C, N3>;
    fn fit(
        &self,
        ctx: &Arc<C>,
        input: &Gen1DArray<T, C, N1>,
        expected: &Gen1DArray<T, C, N3>,
        learning_rate: &isize,
    ) -> BackpropagationData<T>;
    fn update_weights(&mut self, ctx: &Arc<C>);
    fn get_weights(&self, weights: &mut ModelWeights);
    fn set_weights(&mut self, ctx: &Arc<C>, weights: &mut ModelWeights);
}

pub(super) struct CurrentLayerImpl<
    T: 'static + Clone,
    C: 'static,
    const N1: usize,
    const N2: usize,
    const N3: usize,
> {
    pub layer: Box<dyn Layer<T, C, N1, N2>>,
    pub next_layer: NextLayer<T, C, N2, N3>,
}

impl<
        T: 'static + Clone + Sync + Send,
        C: Sync + Send,
        const N1: usize,
        const N2: usize,
        const N3: usize,
    > CurrentLayer<T, C, N1, N3> for CurrentLayerImpl<T, C, N1, N2, N3>
{
    fn execute(&self, ctx: &Arc<C>, input: &Gen1DArray<T, C, N1>) -> Gen1DArray<T, C, N3> {
        let inter = self.layer.execute_layer(ctx, input);

        match &self.next_layer {
            NextLayer::Layer(v) => v.execute(ctx, &inter),
            NextLayer::Finish(v) => v(inter),
        }
    }

    fn fit(
        &self,
        ctx: &Arc<C>,
        input: &Gen1DArray<T, C, N1>,
        expected: &Gen1DArray<T, C, N3>,
        learning_rate: &isize,
    ) -> BackpropagationData<T> {
        let (received, deriv): (Gen1DArray<T, C, N2>, Gen1DArray<T, C, N2>) =
            self.layer.pre_train(ctx, input);

        let (output, real_expected) = match &self.next_layer {
            NextLayer::Layer(v) => {
                let data = v.fit(ctx, &received, expected, learning_rate);
                let (output, error) = (data.output, data.error);
                let arr: Gen1DArray<T, C, N2> = FromWithContext::from_ctx(error(), ctx);

                (output, LayerOutput::Intermediate(arr))
            }
            NextLayer::Finish(map_fn) => {
                let expect_vec: Vec<_> = expected.into();
                let output_vec: Vec<_> = (&map_fn(received.clone())).into();
                let exp_arr: Gen1DArray<T, C, N2> = FromWithContext::from_ctx(expect_vec, ctx);

                (output_vec, LayerOutput::FinalOutput(exp_arr))
            }
        };

        let res = self
            .layer
            .train(ctx, input, (received, deriv), real_expected, learning_rate);

        BackpropagationData {
            output,
            error: Box::new(move || {
                let v: Gen1DArray<T, C, N1> = (res.error)();
                (&v).into()
            }),
        }
    }

    fn update_weights(&mut self, ctx: &Arc<C>) {
        self.layer.update_weights(ctx);
        match &mut self.next_layer {
            NextLayer::Layer(v) => {
                Arc::get_mut(v)
                    .expect("Should be able to get a mutable reference")
                    .update_weights(ctx);
            }
            _ => {}
        };
    }
    fn get_weights(&self, weights: &mut ModelWeights) {
        let lst = self.layer.get_weights();

        let val = general_purpose::STANDARD.encode(&lst);
        weights.layers.push(val);

        match &self.next_layer {
            NextLayer::Layer(v) => {
                v.get_weights(weights);
            }
            _ => {}
        };
    }
    fn set_weights(&mut self, ctx: &Arc<C>, weights: &mut ModelWeights) {
        match &mut self.next_layer {
            NextLayer::Layer(v) => {
                Arc::get_mut(v)
                    .expect("Should be able to get a mutable reference")
                    .set_weights(ctx, weights);
            }
            _ => {}
        };
        let w = weights
            .layers
            .pop()
            .expect("Should have weights for every layer");
        let mut lst: Vec<u8> = Vec::with_capacity(w.len());

        general_purpose::STANDARD
            .decode_vec(w, &mut lst)
            .expect("Should be valid base64 string");

        self.layer.set_weights(ctx, lst);
    }
}
