use std::sync::{Arc, RwLock};

use numbers::NumberType;

use crate::layers::Layer;

use super::{
    types::{CurrentLayer, CurrentLayerImpl, NextLayer},
    Model,
};

pub struct ModelBuilder<T: 'static + Clone + NumberType, const N1: usize, const N2: usize> {
    current_i: usize,
    current_layer: Arc<dyn CurrentLayer<T, N1, N2>>,
}

impl<T: 'static + NumberType + Clone + Sync + Send, const N2: usize, const N3: usize>
    ModelBuilder<T, N2, N3>
where
    T::ContextType: Sync + Send,
{
    pub fn new(mut layer: impl Layer<T, N2, N3> + 'static) -> ModelBuilder<T, N2, N3> {
        layer.update_name("layer_1");

        ModelBuilder {
            current_i: 2,
            current_layer: Arc::new(CurrentLayerImpl {
                layer: Box::new(layer),
                next_layer: NextLayer::Finish(Box::new(|v| v)),
            }),
        }
    }

    pub fn add_layer<const N1: usize>(
        self,
        mut layer: impl Layer<T, N1, N2> + 'static,
    ) -> ModelBuilder<T, N1, N3> {
        layer.update_name(&format!("layer_{}", self.current_i));

        ModelBuilder {
            current_i: self.current_i + 1,
            current_layer: Arc::new(CurrentLayerImpl {
                layer: Box::new(layer),
                next_layer: NextLayer::Layer(self.current_layer),
            }),
        }
    }

    pub fn build(self, ctx: &Arc<T::ContextType>) -> Model<T, N2, N3> {
        Model {
            first_layer: RwLock::new(self.current_layer),
            context: Arc::clone(ctx),
        }
    }
}
