use numbers::Gen1DArray;
use std::sync::Arc;

mod fffc;
pub use fffc::*;

pub enum LayerOutput<T, C, const CURR_N: usize> {
    FinalOutput(Gen1DArray<T, C, CURR_N>),
    Intermediate(Gen1DArray<T, C, CURR_N>),
}

pub trait LayerUpdate {}

pub struct ErrorBackpropagation<T, C, const PREV_N: usize> {
    pub error: Box<dyn FnOnce() -> Gen1DArray<T, C, PREV_N>>,
}

pub trait Layer<T: 'static, C: 'static, const PREV_N: usize, const CURR_N: usize>:
    Send + Sync
{
    fn update_name(&mut self, name: &str);
    fn execute_layer(
        &self,
        ctx: &Arc<C>,
        arr: &Gen1DArray<T, C, PREV_N>,
    ) -> Gen1DArray<T, C, CURR_N>;
    fn pre_train(
        &self,
        ctx: &Arc<C>,
        input: &Gen1DArray<T, C, PREV_N>,
    ) -> (Gen1DArray<T, C, CURR_N>, Gen1DArray<T, C, CURR_N>);
    fn train(
        &self,
        ctx: &Arc<C>,
        input: &Gen1DArray<T, C, PREV_N>,
        received: (Gen1DArray<T, C, CURR_N>, Gen1DArray<T, C, CURR_N>),
        expected: LayerOutput<T, C, CURR_N>,
        learning_rate: &isize,
    ) -> ErrorBackpropagation<T, C, PREV_N>;
    fn update_weights(&mut self, ctx: &Arc<C>);

    fn get_weights(&self) -> Vec<u8>;
    fn set_weights(&mut self, ctx: &Arc<C>, weights: Vec<u8>);
}
