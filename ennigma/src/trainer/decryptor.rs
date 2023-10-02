use std::sync::Arc;

use neuralnetworks::{ModelTrainer, TrainCommand, TrainMetric, TrainStatus};
use numbers::{BooleanType, FixedPointNumber};

pub struct DecryptionTrainer<C> {
    next_trainer: Box<dyn ModelTrainer<f32>>,
    client_context: Arc<C>,
}

impl<C> DecryptionTrainer<C> {
    pub fn new<T: ModelTrainer<f32> + 'static>(new_trainer: T, client_context: &Arc<C>) -> Self {
        Self {
            next_trainer: Box::new(new_trainer),
            client_context: Arc::clone(client_context),
        }
    }
}

impl<T: BooleanType<C>, C, const SIZE: usize, const PRECISION: usize>
    ModelTrainer<FixedPointNumber<SIZE, PRECISION, T, C>> for DecryptionTrainer<C>
{
    fn send_status(&self, status: &TrainStatus<FixedPointNumber<SIZE, PRECISION, T, C>>) {
        self.next_trainer.send_status(&match status {
            TrainStatus::TrainStart { at } => TrainStatus::TrainStart { at: at.clone() },
            TrainStatus::TrainEnd { start, end } => TrainStatus::TrainEnd {
                start: start.clone(),
                end: end.clone(),
            },
            TrainStatus::TrainSuspended { since } => TrainStatus::TrainSuspended {
                since: since.clone(),
            },
            TrainStatus::EpochStart { epoch, at } => TrainStatus::EpochStart {
                epoch: *epoch,
                at: at.clone(),
            },
            TrainStatus::EpochUpdate {
                current_epoch,
                total_epochs,
                current_batch,
                total_batches,
                batch_start,
                batch_end,
            } => TrainStatus::EpochUpdate {
                current_epoch: *current_epoch,
                total_epochs: *total_epochs,
                current_batch: *current_batch,
                total_batches: *total_batches,
                batch_start: batch_start.clone(),
                batch_end: batch_end.clone(),
            },
            TrainStatus::EpochEnd {
                epoch,
                total_epochs,
                start,
                end,
                metrics,
                validation_metrics,
                get_weights,
            } => TrainStatus::EpochEnd {
                epoch: *epoch,
                total_epochs: *total_epochs,
                start: start.clone(),
                end: end.clone(),
                metrics: metrics
                    .iter()
                    .map(|v| TrainMetric {
                        name: v.name.to_owned(),
                        value: v.value.clone().switch_context(&self.client_context).into(),
                    })
                    .collect(),
                validation_metrics: validation_metrics.as_ref().map(|opt| {
                    opt.iter()
                        .map(|v| TrainMetric {
                            name: v.name.to_owned(),
                            value: v.value.clone().switch_context(&self.client_context).into(),
                        })
                        .collect()
                }),
                get_weights: *get_weights,
            },
            TrainStatus::Metrics {
                metrics,
                validation_metrics,
                learning_rate,
            } => TrainStatus::Metrics {
                learning_rate: *learning_rate,
                metrics: metrics
                    .iter()
                    .map(|v| TrainMetric {
                        name: v.name.to_owned(),
                        value: v.value.clone().switch_context(&self.client_context).into(),
                    })
                    .collect(),
                validation_metrics: validation_metrics.as_ref().map(|opt| {
                    opt.iter()
                        .map(|v| TrainMetric {
                            name: v.name.to_owned(),
                            value: v.value.clone().switch_context(&self.client_context).into(),
                        })
                        .collect()
                }),
            },
        });
    }
    fn pull_command(&self) -> TrainCommand {
        self.next_trainer.pull_command()
    }
}
