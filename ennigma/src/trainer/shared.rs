use std::io::Write;
use std::{
    io::{stdout, Stdout},
    sync::{Arc, Mutex},
};

use crossterm::{cursor, terminal, ExecutableCommand};
use crypto::EncryptedContext;
use neuralnetworks::{ModelTrainer, TrainCommand, TrainStatus};

use crate::EncryptedFixedPrecision;

pub struct SharedCryptoTrainer {
    context: Arc<EncryptedContext>,
    ioout: Mutex<Stdout>,
}
impl SharedCryptoTrainer {
    pub fn new(context: &Arc<EncryptedContext>) -> Self {
        let ioout = stdout();
        Self {
            context: Arc::clone(context),
            ioout: Mutex::new(ioout),
        }
    }
}

impl<const SIZE: usize, const PRECISION: usize>
    ModelTrainer<EncryptedFixedPrecision<SIZE, PRECISION>> for SharedCryptoTrainer
{
    fn send_status(&self, status: &TrainStatus<EncryptedFixedPrecision<SIZE, PRECISION>>) {
        let mut ioout = self.ioout.lock().expect("Should unlock mutex");
        match status {
            TrainStatus::TrainStart { at: _ } => {
                ioout.execute(cursor::Hide).unwrap();
                writeln!(
                    &mut ioout,
                    "========================================================"
                )
                .unwrap();
                writeln!(&mut ioout, "Training starting").unwrap();
                writeln!(
                    &mut ioout,
                    "========================================================"
                )
                .unwrap();
            }
            TrainStatus::EpochUpdate {
                current_epoch,
                total_epochs,
                current_batch,
                total_batches,
                // metrics,
                batch_start,
                batch_end,
            } => {
                ioout
                    .execute(terminal::Clear(terminal::ClearType::CurrentLine))
                    .unwrap();
                let duration = batch_end.signed_duration_since(batch_start.clone());
                ioout.execute(cursor::SavePosition).unwrap();
                ioout
                    .write_all(
                        format!(
                            "Epoch {}/{}: {} of {} ({}s, remaining {}s)",
                            current_epoch,
                            total_epochs,
                            current_batch,
                            total_batches,
                            duration.num_seconds(),
                            (duration * (total_batches - current_batch) as i32).to_string(),
                            // metrics
                            //     .iter()
                            //     .map(|m| {
                            //         let v: f32 =
                            //             m.value.clone().switch_context(&self.context).into();
                            //         format!("{}:{}", m.name, v)
                            //     })
                            //     .collect::<Vec<_>>()
                            //     .join(", ")
                        )
                        .as_bytes(),
                    )
                    .unwrap();
                ioout.execute(cursor::RestorePosition).unwrap();
                ioout.flush().unwrap();
            }
            TrainStatus::EpochEnd {
                epoch,
                total_epochs,
                start,
                end,
                metrics,
                validation_metrics,
                get_weights: _,
            } => {
                ioout
                    .execute(terminal::Clear(terminal::ClearType::CurrentLine))
                    .unwrap();
                println!(
                    "Epoch {}/{} - {}s: {} val:{}",
                    epoch,
                    total_epochs,
                    end.signed_duration_since(start.clone()).num_seconds(),
                    metrics
                        .iter()
                        .map(|m| {
                            let v: f32 = m.value.clone().switch_context(&self.context).into();
                            format!("{}:{}", m.name, v)
                        })
                        .collect::<Vec<_>>()
                        .join(", "),
                    match validation_metrics {
                        None => "".to_owned(),
                        Some(v) => v
                            .iter()
                            .map(|m| {
                                let v: f32 = m.value.clone().switch_context(&self.context).into();
                                format!("{}:{}", m.name, v)
                            })
                            .collect::<Vec<_>>()
                            .join(", "),
                    },
                );
            }
            TrainStatus::TrainEnd { start, end } => {
                writeln!(
                    &mut ioout,
                    "========================================================"
                )
                .unwrap();
                writeln!(
                    &mut ioout,
                    "Training finished in {}s",
                    end.signed_duration_since(start.clone()).num_seconds()
                )
                .unwrap();
                writeln!(
                    &mut ioout,
                    "========================================================"
                )
                .unwrap();
                ioout.execute(cursor::Show).unwrap();
            }
            _ => {}
        }
    }
    fn pull_command(&self) -> TrainCommand {
        TrainCommand::None
    }
}
