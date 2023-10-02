use std::{
    fmt::Display,
    io::{stdout, Stdout, Write},
    sync::Mutex,
};

use crossterm::{cursor, terminal, ExecutableCommand};

use crate::{ModelTrainer, TrainCommand, TrainStatus};

pub struct ConsoleTrainer {
    ioout: Mutex<Stdout>,
}
impl ConsoleTrainer {
    pub fn new() -> Self {
        let ioout = stdout();
        Self {
            ioout: Mutex::new(ioout),
        }
    }
}

impl<T: Display> ModelTrainer<T> for ConsoleTrainer {
    fn send_status(&self, status: &TrainStatus<T>) {
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
                batch_start,
                batch_end,
            } => {
                let duration = batch_end.signed_duration_since(batch_start.clone());
                ioout
                    .execute(terminal::Clear(terminal::ClearType::CurrentLine))
                    .unwrap();
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
                            (duration * (total_batches - current_batch) as i32).num_seconds(),
                            // metrics
                            //     .iter()
                            //     .map(|m| format!("{}:{}", m.name, m.value))
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
                ..
            } => {
                ioout
                    .execute(terminal::Clear(terminal::ClearType::CurrentLine))
                    .unwrap();
                println!(
                    "Epoch {}/{} - {}s: {} val: {}",
                    epoch,
                    total_epochs,
                    end.signed_duration_since(start.clone()).num_seconds(),
                    metrics
                        .iter()
                        .map(|m| format!("{}:{}", m.name, m.value))
                        .collect::<Vec<_>>()
                        .join(", "),
                    match validation_metrics.as_ref() {
                        None => "".to_owned(),
                        Some(v) => v
                            .iter()
                            .map(|m| format!("{}:{}", m.name, m.value))
                            .collect::<Vec<_>>()
                            .join(", "),
                    }
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
