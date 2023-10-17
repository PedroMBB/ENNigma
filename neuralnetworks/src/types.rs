use std::sync::Arc;

pub trait SerializableModel {
    fn get_weights(&self) -> ModelWeights;
    fn set_weights(&mut self, weights: ModelWeights);
}

pub trait ActivationFn<T: NumberType, const N: usize>: std::fmt::Debug + Sync + Send {
    fn activate(&self, ctx: &Arc<T::ContextType>, v: T) -> T;
    fn activate_and_derivative(&self, ctx: &Arc<T::ContextType>, v: T) -> (T, T);

    fn activate_multiple(
        &self,
        ctx: &Arc<T::ContextType>,
        lst: Gen1DArray<T, N>,
    ) -> Gen1DArray<T, N>;
    fn activate_and_derivative_multiple(
        &self,
        ctx: &Arc<T::ContextType>,
        lst: Gen1DArray<T, N>,
    ) -> (Gen1DArray<T, N>, Gen1DArray<T, N>);
}
use chrono::NaiveDateTime;
use numbers::{Gen1DArray, NumberType};
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize, PartialEq, Clone)]
pub struct ModelWeights {
    pub layers: Vec<String>,
}

pub trait ModelTrainer<T> {
    fn send_status(&self, status: &TrainStatus<T>);
    fn pull_command(&self) -> TrainCommand;
}

pub enum TrainStatus<'a, T> {
    TrainSuspended {
        since: NaiveDateTime,
    },
    TrainStart {
        at: NaiveDateTime,
    },
    EpochStart {
        epoch: usize,
        at: NaiveDateTime,
    },
    EpochUpdate {
        current_epoch: usize,
        total_epochs: usize,
        current_batch: usize,
        total_batches: usize,
        batch_start: NaiveDateTime,
        batch_end: NaiveDateTime,
    },
    EpochEnd {
        epoch: usize,
        total_epochs: usize,
        start: NaiveDateTime,
        end: NaiveDateTime,
        metrics: Vec<TrainMetric<T>>,
        validation_metrics: Option<Vec<TrainMetric<T>>>,
        get_weights: &'a dyn Fn() -> ModelWeights,
    },
    TrainEnd {
        start: NaiveDateTime,
        end: NaiveDateTime,
    },
    Metrics {
        metrics: Vec<TrainMetric<T>>,
        validation_metrics: Option<Vec<TrainMetric<T>>>,
        learning_rate: isize,
    },
}

#[derive(Serialize, Deserialize)]
pub enum TrainCommand {
    None,
    Pause,
    Resume,
    UpdateLearningRate { new_learning_rate: isize },
    CalculateMetrics,
    StopTraining,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TrainMetric<T> {
    pub name: String,
    pub value: T,
}

impl<T> TrainMetric<T> {
    pub fn new(name: &'static str, value: T) -> Self {
        Self {
            name: name.to_string(),
            value,
        }
    }
}
