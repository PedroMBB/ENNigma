use crate::{ModelTrainer, TrainCommand, TrainStatus};

pub struct ForwardTrainer<T> {
    command_trainer: Option<Box<dyn ModelTrainer<T>>>,
    trainers: Vec<Box<dyn ModelTrainer<T>>>,
}
impl<T> ModelTrainer<T> for ForwardTrainer<T> {
    fn send_status(&self, status: &TrainStatus<T>) {
        self.trainers.iter().for_each(|v| v.send_status(status))
    }
    fn pull_command(&self) -> TrainCommand {
        if let Some(t) = self.command_trainer.as_ref() {
            t.pull_command()
        } else {
            TrainCommand::None
        }
    }
}

pub struct ForwardTrainerBuilder<T>(
    Vec<Box<dyn ModelTrainer<T>>>,
    Option<Box<dyn ModelTrainer<T>>>,
);
impl<T> ForwardTrainerBuilder<T> {
    pub fn new() -> Self {
        ForwardTrainerBuilder(Vec::default(), None)
    }
    pub fn add_trainer<TR: ModelTrainer<T> + 'static>(mut self, trainer: TR) -> Self {
        self.0.push(Box::new(trainer));
        self
    }
    pub fn add_command_handler<TR: ModelTrainer<T> + 'static>(mut self, trainer: TR) -> Self {
        self.1 = Some(Box::new(trainer));
        self
    }
    pub fn build(self) -> ForwardTrainer<T> {
        ForwardTrainer {
            trainers: self.0,
            command_trainer: self.1,
        }
    }
}
