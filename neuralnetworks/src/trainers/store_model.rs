use crate::{ModelTrainer, TrainCommand, TrainStatus};
use std::fs;
use std::{fmt::Display, fs::File, path::MAIN_SEPARATOR_STR};

pub struct StoreModelTrainer {
    dir: String,
    base_name: String,
}
impl StoreModelTrainer {
    pub fn new(dir: &str, base_name: &str) -> Self {
        fs::create_dir_all(&dir).expect("Should be able to create model storage dir");

        Self {
            dir: dir.to_string(),
            base_name: base_name.to_string(),
        }
    }
}

impl<T: Display> ModelTrainer<T> for StoreModelTrainer {
    fn send_status(&self, status: &TrainStatus<T>) {
        match status {
            TrainStatus::EpochEnd {
                epoch, get_weights, ..
            } => {
                let file_name = format!("{}_{}.json", self.base_name, epoch);
                let file_path = format!("{}{}{}", self.dir, MAIN_SEPARATOR_STR, file_name);

                let Ok(mut file) = File::create(&file_path) else {
                    error!("[StoreModelTrainer::send_status] Could not create epoch file '{}'", file_path);
                    return
                };

                let weights = get_weights();

                let Ok(_) = serde_json::to_writer(&mut file, &weights) else {
                    error!("[StoreModelTrainer::send_status] Could not serialize model on epoch {}", epoch);
                    return
                };
            }
            _ => {}
        }
    }
    fn pull_command(&self) -> TrainCommand {
        TrainCommand::None
    }
}
