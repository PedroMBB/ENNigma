use std::{fmt::Display, io::Write, sync::Mutex};

use crate::{ModelTrainer, TrainCommand, TrainStatus};

pub struct SinkTrainer {
    writer: Mutex<(bool, Box<dyn Write + Send + Sync>)>,
}
impl SinkTrainer {
    pub fn new<W: Write + Send + Sync + 'static>(writer: W) -> Self {
        Self {
            writer: Mutex::new((false, Box::new(writer))),
        }
    }
}

impl<T: Display> ModelTrainer<T> for SinkTrainer {
    fn send_status(&self, status: &TrainStatus<T>) {
        let mut writer_lock = self.writer.lock().expect("Should unlock mutex");
        match status {
            TrainStatus::EpochEnd {
                epoch,
                start,
                end,
                metrics,
                validation_metrics,
                ..
            } => {
                if !writer_lock.0 {
                    writeln!(
                        &mut writer_lock.1,
                        "Epoch;Duration;{};{}",
                        metrics
                            .iter()
                            .map(|m| m.name.to_owned())
                            .collect::<Vec<_>>()
                            .join(";"),
                        match validation_metrics.as_ref() {
                            Some(v) => v
                                .iter()
                                .map(|m| format!("val_{}", m.name))
                                .collect::<Vec<_>>()
                                .join(";"),
                            None => "".to_owned(),
                        }
                    )
                    .unwrap();

                    writer_lock.0 = true;
                }

                writeln!(
                    &mut writer_lock.1,
                    "{};{};{};{}",
                    epoch,
                    end.signed_duration_since(start.clone()).num_seconds(),
                    metrics
                        .iter()
                        .map(|m| format!("{}", m.value))
                        .collect::<Vec<_>>()
                        .join(";"),
                    match validation_metrics.as_ref() {
                        Some(v) => v
                            .iter()
                            .map(|m| format!("{}", m.value))
                            .collect::<Vec<_>>()
                            .join(";"),
                        None => "".to_owned(),
                    }
                )
                .unwrap();
            }
            TrainStatus::TrainEnd { .. } => {
                writer_lock.1.flush().unwrap();
            }
            _ => {}
        }
    }
    fn pull_command(&self) -> TrainCommand {
        TrainCommand::None
    }
}
