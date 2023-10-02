use crate::{
    ModelTrainer, ModelWeights, SerializableModel, TrainCommand, TrainMetric, TrainStatus,
};

use self::types::CurrentLayer;

use super::metrics::Metric;
use chrono::Utc;
use numbers::{BoolFixedPointNumber, FromWithContext, Gen1DArray};
use rand::seq::SliceRandom;
use serde::Serialize;
use std::{
    ops::{Add, DerefMut, Div, Mul},
    sync::{Arc, RwLock},
};

mod builder;
mod types;

pub use builder::ModelBuilder;

pub type BoolModel<
    const SIZE: usize,
    const PRECISION: usize,
    const INPUT: usize,
    const OUTPUT: usize,
> = Model<BoolFixedPointNumber<SIZE, PRECISION>, (), INPUT, OUTPUT>;

pub struct Model<T: 'static + Clone, C, const INPUT_N: usize, const OUTPUT_N: usize> {
    first_layer: RwLock<Arc<dyn CurrentLayer<T, C, INPUT_N, OUTPUT_N>>>,
    context: Arc<C>,
}

impl<
        T: 'static + Clone + Sync + Send + Serialize,
        C: 'static + Sync + Send,
        const INPUT_N: usize,
        const OUTPUT_N: usize,
    > Model<T, C, INPUT_N, OUTPUT_N>
where
    T: FromWithContext<f32, C>,
    for<'a> T: Add<&'a T, Output = T>,
    for<'a> T: Mul<&'a T, Output = T>,
    for<'a> T: Div<&'a T, Output = T>,
{
    pub fn execute(&self, input: &Gen1DArray<T, C, INPUT_N>) -> Gen1DArray<T, C, OUTPUT_N> {
        self.first_layer
            .read()
            .expect("Should be able to read lock mutex")
            .execute(&self.context, input)
    }
    pub fn execute_multiple(
        &self,
        input: &[&Gen1DArray<T, C, INPUT_N>],
    ) -> Vec<Gen1DArray<T, C, OUTPUT_N>> {
        #[cfg(feature = "profiling")]
        println!("Profiling: ON");

        let num_cores = std::env::var("NUM_THREADS")
            .map(|v| v.parse().ok())
            .ok()
            .flatten()
            .unwrap_or(16_usize);

        #[cfg(feature = "tokio")]
        let _handle = match tokio::runtime::Handle::try_current() {
            Ok(_rt) => {
                info!("[Model::execute_multiple] Using already created tokio runtime");
                // Runtime exists
                None
            }
            Err(_) => {
                // Runtime does not exist
                info!("[Model::execute_multiple] Creating tokio runtime");
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .worker_threads(num_cores)
                    .build()
                    .expect("Should build tokio runtime");

                Some(rt)
            }
        };
        #[cfg(feature = "tokio")]
        let _handle1 = match _handle.as_ref() {
            Some(rt) => Some(rt.enter()),
            None => None,
        };
        #[cfg(feature = "rayon")]
        info!("[Model::execute_multiple] Using rayon runtime");

        info!("[Model::execute_multiple] Using {} threads", num_cores);

        let val: f32 = (input.len() as f32 / num_cores as f32).ceil();

        let v = input.chunks(val as usize).enumerate().collect::<Vec<_>>();
        let v1: Vec<_> = v.iter().map(|v| (&v.0, v.1)).collect();

        {
            let layer = self
                .first_layer
                .read()
                .expect("Should be able to read first layer");

            execute_multiple(&layer, &self.context, v1)
        }
    }
    pub fn fit_single(
        &self,
        input: &Gen1DArray<T, C, INPUT_N>,
        expected: &Gen1DArray<T, C, OUTPUT_N>,
        learning_rate: &isize,
    ) -> Gen1DArray<T, C, OUTPUT_N> {
        let data = self
            .first_layer
            .read()
            .expect("Should be able to read lock mutex")
            .fit(&self.context, input, expected, learning_rate);
        let output: Gen1DArray<T, C, OUTPUT_N> =
            FromWithContext::from_ctx(data.output, &self.context);
        output
    }
    pub fn fit<'a, Trainer: ModelTrainer<T>>(
        &'a mut self,
        epochs: usize,
        chunk_size: usize,
        mut learning_rate: isize,
        train: (
            &'a [Gen1DArray<T, C, INPUT_N>],
            &'a [Gen1DArray<T, C, OUTPUT_N>],
        ),
        test: Option<&[(Gen1DArray<T, C, INPUT_N>, Gen1DArray<T, C, OUTPUT_N>)]>,
        metrics: &[Box<dyn Metric<T, C, OUTPUT_N>>],
        trainer: &Trainer,
        mut rng: &mut dyn rand::RngCore,
    ) -> Option<Vec<TrainMetric<T>>> {
        let mut stop = false;

        let train_start = Utc::now().naive_utc();
        trainer.send_status(&TrainStatus::TrainStart {
            at: Utc::now().naive_utc(),
        });
        let mut training: Vec<_> = train.0.into_iter().zip(train.1.into_iter()).collect();

        let num_cores = std::env::var("NUM_THREADS")
            .map(|v| v.parse().ok())
            .ok()
            .flatten()
            .unwrap_or(16_usize);

        #[cfg(feature = "tokio")]
        let _handle = match tokio::runtime::Handle::try_current() {
            Ok(_rt) => {
                info!("[Model::fit] Using already created tokio runtime");
                // Runtime exists
                None
            }
            Err(_) => {
                // Runtime does not exist
                info!("[Model::fit] Creating tokio runtime");
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .worker_threads(num_cores)
                    .build()
                    .expect("Should build tokio runtime");

                Some(rt)
            }
        };
        #[cfg(feature = "tokio")]
        let _handle1 = match _handle.as_ref() {
            Some(rt) => Some(rt.enter()),
            None => None,
        };
        #[cfg(feature = "rayon")]
        info!("[Model::fit] Using rayon runtime");

        info!("[Model::fit] Using {} threads", num_cores);

        #[cfg(feature = "profiling")]
        println!("Profiling: ON");

        // Calc val metrics
        let val_metrics = || match &test {
            Some(test_data) => {
                let val: f32 = (test_data.len() as f32 / num_cores as f32).ceil();
                let v: Vec<_> = test_data.chunks(val as usize).collect();

                let test_results: Vec<FitResult<T, C, OUTPUT_N>> = {
                    let layer = self
                        .first_layer
                        .read()
                        .expect("Should be able to read first layer");

                    execute_test_data(&layer, &self.context, v)
                };

                let metrics_test_results = test_results
                    .iter()
                    .map(|v| (&v.expected, &v.received))
                    .collect::<Vec<_>>();
                let metrics_val_test = metrics
                    .iter()
                    .map(|v| v.calc_metric(&self.context, metrics_test_results.as_slice()))
                    .collect::<Vec<_>>();
                Some(metrics_val_test)
            }
            None => None,
        };

        // Command handling
        let command_fn = |stop: &mut bool,
                          learning_rate_ref: &mut isize,
                          results: &Vec<FitResult<T, C, OUTPUT_N>>| {
            let mut leave = true;
            let since = Utc::now().naive_utc();
            loop {
                match trainer.pull_command() {
                    TrainCommand::None => {}
                    TrainCommand::Pause => {
                        leave = false;
                    }
                    TrainCommand::Resume => {
                        leave = true;
                    }
                    TrainCommand::UpdateLearningRate { new_learning_rate } => {
                        *learning_rate_ref = new_learning_rate;
                    }
                    TrainCommand::StopTraining => {
                        *stop = true;
                    }
                    TrainCommand::CalculateMetrics => {
                        let metrics_results = results
                            .iter()
                            .map(|v| (&v.expected, &v.received))
                            .collect::<Vec<_>>();
                        let metrics_val = metrics
                            .iter()
                            .map(|v| v.calc_metric(&self.context, metrics_results.as_slice()))
                            .collect::<Vec<_>>();

                        let validation_metrics = val_metrics();

                        trainer.send_status(&TrainStatus::Metrics {
                            metrics: metrics_val,
                            validation_metrics,
                            learning_rate: learning_rate_ref.clone(),
                        });
                    }
                }

                if leave {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
                trainer.send_status(&TrainStatus::TrainSuspended {
                    since: since.clone(),
                });
            }
        };

        /*
           Epochs
        */
        'outer: for epoch in 0..epochs {
            trace!("[Model::fit] Starting epoch {} of {}", epoch, epochs);
            command_fn(&mut stop, &mut learning_rate, &vec![]);
            if stop {
                break 'outer;
            }

            let epoch_number = epoch + 1;
            let epoch_start = Utc::now().naive_utc();
            trainer.send_status(&TrainStatus::EpochStart {
                epoch: epoch_number,
                at: epoch_start.clone(),
            });
            training.shuffle(&mut rng);

            let num_chunks: usize = ((training.len() as f64 / chunk_size as f64).ceil() as i64)
                .try_into()
                .expect("Should give a positive result");

            let mut results: Vec<FitResult<T, C, OUTPUT_N>> = Vec::with_capacity(training.len());

            for (chunk_n, chunk) in training.as_slice().chunks(chunk_size).enumerate() {
                trace!("[Model::fit] Starting chunk {} of {}", chunk_n, num_chunks);
                command_fn(&mut stop, &mut learning_rate, &results);
                if stop {
                    break 'outer;
                }

                let chunk_start = Utc::now().naive_utc();
                let val: f32 = (chunk.len() as f32 / num_cores as f32).ceil();

                let v: Vec<_> = chunk.chunks(val as usize).collect();

                // let size: T = FromWithContext::from_ctx(chunk_size as f32, &self.context);
                // let lr: T = &learning_rate / &size;
                let lr = learning_rate.clone();

                let mut local_results: Vec<FitResult<T, C, OUTPUT_N>> = {
                    let layer = self
                        .first_layer
                        .read()
                        .expect("Should be able to read first layer");

                    execute_chunk(&layer, &self.context, v, &lr)
                };

                {
                    let mut rw_ref = self
                        .first_layer
                        .write()
                        .expect("Should be able to write lock the mutex");
                    let mu_ref: &mut Arc<_> = rw_ref.deref_mut();
                    Arc::get_mut(mu_ref)
                        .expect("Should be able to get the mutable reference")
                        .update_weights(&self.context);
                }

                #[cfg(all(feature = "profiling", not(feature = "profiling_soft")))]
                return None;

                // let metrics_results = local_results
                //     .iter()
                //     .map(|v| (&v.expected, &v.received))
                //     .collect::<Vec<_>>();
                // let metrics = metrics
                //     .iter()
                //     .map(|v| v.calc_metric(&self.context, metrics_results.as_slice()))
                //     .collect::<Vec<_>>();

                let chunk_end = Utc::now().naive_utc();
                trainer.send_status(&TrainStatus::EpochUpdate {
                    current_epoch: epoch_number,
                    total_epochs: epochs,
                    current_batch: chunk_n + 1,
                    total_batches: num_chunks,
                    batch_start: chunk_start,
                    batch_end: chunk_end,
                });

                results.append(&mut local_results);

                trace!("[Model::fit] Finishing chunk {} of {}", chunk_n, num_chunks);
                command_fn(&mut stop, &mut learning_rate, &results);
                if stop {
                    break 'outer;
                }
            }

            let epoch_end = Utc::now().naive_utc();

            let metrics_results = results
                .iter()
                .map(|v| (&v.expected, &v.received))
                .collect::<Vec<_>>();
            let metrics_val = metrics
                .iter()
                .map(|v| v.calc_metric(&self.context, metrics_results.as_slice()))
                .collect::<Vec<_>>();

            let validation_metrics = val_metrics();
            trainer.send_status(&TrainStatus::EpochEnd {
                total_epochs: epochs,
                epoch: epoch_number,
                start: epoch_start,
                end: epoch_end,
                metrics: metrics_val,
                validation_metrics,
                get_weights: &|| self.get_weights(),
            });

            trace!("[Model::fit] Finishing epoch {} of {}", epoch, epochs);
            command_fn(&mut stop, &mut learning_rate, &results);
            if stop {
                break 'outer;
            }
        }

        /*
        Finish
        */
        let training_end = Utc::now().naive_utc();
        trainer.send_status(&TrainStatus::TrainEnd {
            start: train_start,
            end: training_end,
        });

        val_metrics()
    }
}

impl<
        T: 'static + Clone + Sync + Send + Serialize,
        C: 'static + Sync + Send,
        const INPUT_N: usize,
        const OUTPUT_N: usize,
    > SerializableModel for Model<T, C, INPUT_N, OUTPUT_N>
{
    fn get_weights(&self) -> ModelWeights {
        let mut weights = ModelWeights::default();
        let lock = self
            .first_layer
            .read()
            .expect("Should be able to lock model");
        lock.get_weights(&mut weights);
        weights
    }
    fn set_weights(&mut self, mut weights: ModelWeights) {
        let mut rw_ref = self
            .first_layer
            .write()
            .expect("Should be able to write lock the mutex");
        let mu_ref: &mut Arc<_> = rw_ref.deref_mut();
        Arc::get_mut(mu_ref)
            .expect("Should be able to get the mutable reference")
            .set_weights(&self.context, &mut weights);
    }
}

struct FitResult<T, C, const S: usize> {
    expected: Gen1DArray<T, C, S>,
    received: Gen1DArray<T, C, S>,
}

// impl<T: Clone, C, const S: usize> Clone for FitResult<T, C, S> {
//     fn clone(&self) -> Self {
//         Self {
//             expected: self.expected.clone(),
//             received: self.expected.clone(),
//         }
//     }
// }

#[cfg(feature = "rayon")]
fn execute_multiple<
    T: 'static + Clone + Sync + Send,
    C: 'static + Sync + Send,
    const INPUT_N: usize,
    const OUTPUT_N: usize,
>(
    layer: &Arc<dyn CurrentLayer<T, C, INPUT_N, OUTPUT_N>>,
    ctx: &Arc<C>,
    chunk: Vec<(&usize, &[&Gen1DArray<T, C, INPUT_N>])>,
) -> Vec<Gen1DArray<T, C, OUTPUT_N>> {
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    chunk
        .into_par_iter()
        .map(|(_, lst)| {
            lst.into_iter()
                .map(|inp| layer.execute(ctx, *inp))
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect()
}

#[cfg(feature = "rayon")]
fn execute_chunk<
    T: 'static + Clone + Sync + Send,
    C: 'static + Sync + Send,
    const INPUT_N: usize,
    const OUTPUT_N: usize,
>(
    layer: &Arc<dyn CurrentLayer<T, C, INPUT_N, OUTPUT_N>>,
    ctx: &Arc<C>,
    chunk: Vec<&[(&Gen1DArray<T, C, INPUT_N>, &Gen1DArray<T, C, OUTPUT_N>)]>,
    learning_rate: &isize,
) -> Vec<FitResult<T, C, OUTPUT_N>> {
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    chunk
        .into_par_iter()
        .map(|lst| {
            lst.into_iter()
                .map(|(inp, exp)| {
                    let data = layer.fit(ctx, *inp, *exp, learning_rate);

                    FitResult {
                        expected: (**exp).clone(),
                        received: FromWithContext::from_ctx(data.output, ctx),
                    }
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect()
}

#[cfg(feature = "rayon")]
fn execute_test_data<
    T: 'static + Clone + Sync + Send,
    C: 'static + Sync + Send,
    const INPUT_N: usize,
    const OUTPUT_N: usize,
>(
    layer: &Arc<dyn CurrentLayer<T, C, INPUT_N, OUTPUT_N>>,
    ctx: &Arc<C>,
    chunk: Vec<&[(Gen1DArray<T, C, INPUT_N>, Gen1DArray<T, C, OUTPUT_N>)]>,
) -> Vec<FitResult<T, C, OUTPUT_N>> {
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    chunk
        .into_par_iter()
        .map(|lst| {
            lst.into_iter()
                .map(|(inp, exp)| {
                    let data = layer.execute(ctx, inp);

                    FitResult {
                        expected: (*exp).clone(),
                        received: data,
                    }
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect()
}

#[cfg(feature = "tokio")]
fn execute_multiple<
    T: 'static + Clone + Sync + Send,
    C: 'static + Sync + Send,
    const INPUT_N: usize,
    const OUTPUT_N: usize,
>(
    layer: &Arc<dyn CurrentLayer<T, C, INPUT_N, OUTPUT_N>>,
    ctx: &Arc<C>,
    chunk: Vec<(&usize, &[&Gen1DArray<T, C, INPUT_N>])>,
) -> Vec<Gen1DArray<T, C, OUTPUT_N>> {
    use tokio::sync::Mutex;

    let lst: Vec<_> = chunk.iter().map(|_| None).collect();
    let ret_lst = Mutex::new(lst);
    tokio_scoped::scope(|scope| {
        for lst in chunk {
            scope.spawn(async {
                let l = lst
                    .1
                    .into_iter()
                    .map(|inp| layer.execute(ctx, *inp))
                    .collect::<Vec<_>>();
                ret_lst.lock().await[*lst.0] = Some(l);
            });
        }
    });

    let lst = ret_lst
        .into_inner()
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .expect("Should all be filled");

    lst.into_iter().flatten().collect()
}

#[cfg(feature = "tokio")]
fn execute_chunk<
    T: 'static + Clone + Sync + Send,
    C: 'static + Sync + Send,
    const INPUT_N: usize,
    const OUTPUT_N: usize,
>(
    layer: &Arc<dyn CurrentLayer<T, C, INPUT_N, OUTPUT_N>>,
    ctx: &Arc<C>,
    chunk: Vec<&[(&Gen1DArray<T, C, INPUT_N>, &Gen1DArray<T, C, OUTPUT_N>)]>,
    learning_rate: &isize,
) -> Vec<FitResult<T, C, OUTPUT_N>> {
    use tokio::sync::Mutex;

    let ret_lst = Mutex::new(vec![]);
    tokio_scoped::scope(|scope| {
        for lst in chunk {
            scope.spawn(async {
                let mut l = lst
                    .into_iter()
                    .map(|(inp, exp)| {
                        let data = layer.fit(ctx, *inp, *exp, learning_rate);

                        FitResult {
                            expected: (**exp).clone(),
                            received: FromWithContext::from_ctx(data.output, ctx),
                        }
                    })
                    .collect::<Vec<_>>();
                ret_lst.lock().await.append(&mut l);
            });
        }
    });
    ret_lst.into_inner()
}

#[cfg(feature = "tokio")]
fn execute_test_data<
    T: 'static + Clone + Sync + Send,
    C: 'static + Sync + Send,
    const INPUT_N: usize,
    const OUTPUT_N: usize,
>(
    layer: &Arc<dyn CurrentLayer<T, C, INPUT_N, OUTPUT_N>>,
    ctx: &Arc<C>,
    chunk: Vec<&[(Gen1DArray<T, C, INPUT_N>, Gen1DArray<T, C, OUTPUT_N>)]>,
) -> Vec<FitResult<T, C, OUTPUT_N>> {
    use tokio::sync::Mutex;

    let ret_lst = Mutex::new(vec![]);
    tokio_scoped::scope(|scope| {
        for lst in chunk {
            scope.spawn(async {
                let mut l = lst
                    .into_iter()
                    .map(|(inp, exp)| {
                        let data = layer.execute(ctx, inp);

                        FitResult {
                            expected: (*exp).clone(),
                            received: data,
                        }
                    })
                    .collect::<Vec<_>>();
                ret_lst.lock().await.append(&mut l);
            });
        }
    });
    ret_lst.into_inner()
}
