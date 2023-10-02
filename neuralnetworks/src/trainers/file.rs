use super::SinkTrainer;
use crate::{ModelTrainer, TrainCommand, TrainStatus};
use std::{fmt::Display, fs::File, path::MAIN_SEPARATOR_STR};

pub struct FileTrainer {
    trainer: SinkTrainer,
}
impl FileTrainer {
    pub fn new(dir: &str, name: &str) -> Self {
        std::fs::create_dir_all(&dir).expect("Should be able to create training dir");
        let ioout = File::create(&format!("{}{}{}", dir, MAIN_SEPARATOR_STR, name))
            .expect("Should be able to open the file");
        Self {
            trainer: SinkTrainer::new(ioout),
        }
    }
}

impl<T: Display> ModelTrainer<T> for FileTrainer {
    fn send_status(&self, status: &TrainStatus<T>) {
        self.trainer.send_status(status)
    }
    fn pull_command(&self) -> TrainCommand {
        <SinkTrainer as ModelTrainer<T>>::pull_command(&self.trainer)
    }
}
