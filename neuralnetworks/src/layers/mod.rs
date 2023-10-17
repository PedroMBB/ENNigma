use numbers::{Gen1DArray, NumberType};
use std::sync::Arc;

mod fffc;
pub use fffc::*;

pub enum LayerOutput<T: NumberType, const CURR_N: usize> {
    FinalOutput(Gen1DArray<T, CURR_N>),
    Intermediate(Gen1DArray<T, CURR_N>),
}

pub trait LayerUpdate {}

pub struct ErrorBackpropagation<T: NumberType, const PREV_N: usize> {
    pub error: Box<dyn FnOnce() -> Gen1DArray<T, PREV_N>>,
}

pub trait Layer<T: 'static + NumberType, const PREV_N: usize, const CURR_N: usize>:
    Send + Sync
where
    T::ContextType: 'static,
{
    fn update_name(&mut self, name: &str);
    fn execute_layer(
        &self,
        ctx: &Arc<T::ContextType>,
        arr: &Gen1DArray<T, PREV_N>,
    ) -> Gen1DArray<T, CURR_N>;
    fn pre_train(
        &self,
        ctx: &Arc<T::ContextType>,
        input: &Gen1DArray<T, PREV_N>,
    ) -> (Gen1DArray<T, CURR_N>, Gen1DArray<T, CURR_N>);
    fn train(
        &self,
        ctx: &Arc<T::ContextType>,
        input: &Gen1DArray<T, PREV_N>,
        received: (Gen1DArray<T, CURR_N>, Gen1DArray<T, CURR_N>),
        expected: LayerOutput<T, CURR_N>,
        learning_rate: &isize,
    ) -> ErrorBackpropagation<T, PREV_N>;
    fn update_weights(&mut self, ctx: &Arc<T::ContextType>);

    fn get_weights(&self) -> Vec<u8>;
    fn set_weights(&mut self, ctx: &Arc<T::ContextType>, weights: Vec<u8>);
}
