use std::fs::File;
use std::time::Instant;

#[derive(Default)]
pub struct ExecutionStorage {
    lines: Vec<String>,
}
impl ExecutionStorage {
    pub fn add_executions<'a, T, O>(&mut self, execs: Executions<'a, T, O>) {
        for exec in execs.executions {
            self.lines.push(format!(
                "{}\t{}\t{}\t{}\t{}",
                exec.name,
                exec.operation,
                exec.variant,
                exec.end.duration_since(exec.start).as_secs_f64(),
                exec.correct
            ));
        }
    }
}

pub struct Executions<'a, T, O> {
    pub executions: Vec<Execution>,
    pub mapper: &'a dyn Fn(&O) -> T,
}
impl<'a, T, O> Executions<'a, T, O> {
    pub fn execute<F>(
        &mut self,
        name: &str,
        operation: &'static str,
        variant: &'static str,
        expected: T,
        func: F,
    ) -> O
    where
        F: FnOnce() -> O,
        T: PartialEq,
    {
        let (exec, v) = Execution::execute(name, operation, variant, expected, func, &self.mapper);
        self.executions.push(exec);
        v
    }
    pub fn execute_nomap<F>(
        &mut self,
        name: &str,
        operation: &'static str,
        variant: &'static str,
        expected: T,
        func: F,
    ) -> T
    where
        F: Fn() -> T,
        T: PartialEq + Clone,
    {
        let (exec, v) =
            Execution::execute(name, operation, variant, expected, func, &|v| v.clone());
        self.executions.push(exec);
        v
    }
}

pub struct Execution {
    name: String,
    operation: &'static str,
    variant: &'static str,
    start: Instant,
    end: Instant,
    correct: bool,
}
impl Execution {
    fn execute<T, F, O>(
        name: &str,
        operation: &'static str,
        variant: &'static str,
        expected: T,
        func: F,
        map: &dyn Fn(&O) -> T,
    ) -> (Self, O)
    where
        F: FnOnce() -> O,
        T: PartialEq,
    {
        let start = Instant::now();
        let res = func();
        let end = Instant::now();

        let res_t = map(&res);

        (
            Self {
                name: name.to_owned(),
                operation,
                variant,
                start,
                end,
                correct: expected == res_t,
            },
            res,
        )
    }
}

pub fn write_to_file(file: &mut File, lst: Vec<Execution>) {
    use std::io::Write;
    writeln!(file, "Name\tOperation\tVariant\tDuration\tCorrect").unwrap();
    for exec in lst {
        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{}",
            exec.name,
            exec.operation,
            exec.variant,
            exec.end.duration_since(exec.start).as_secs_f64(),
            exec.correct
        )
        .unwrap();
    }
}
pub fn write_to_file_storage(file: &mut File, lst: ExecutionStorage) {
    use std::io::Write;
    writeln!(file, "Name\tOperation\tVariant\tDuration\tCorrect").unwrap();
    for exec in lst.lines {
        writeln!(file, "{}", exec).unwrap();
    }
}
