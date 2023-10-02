#[cfg(feature = "profiling")]
use std::{fs::File, io::Write, sync::Mutex, time::Instant};

#[cfg(feature = "profiling")]
use lazy_static::lazy_static;

#[cfg(feature = "profiling")]
lazy_static! {
    static ref PROFILING_FILE: Mutex<File> = {
        let mut file = File::create("durations.csv").unwrap();
        writeln!(&mut file, "place,operation,duration").expect("Should write header");
        Mutex::new(file)
    };
}

#[cfg(feature = "profiling")]
pub fn evaluate_fn<T>(place: &str, operation: &str, func: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let res = func();
    let duration = start.elapsed().as_secs_f64();
    {
        let mut lock = PROFILING_FILE.lock().unwrap();
        writeln!(&mut lock, "{},{},{}", place, operation, duration).unwrap();
    }

    res
}

#[cfg(feature = "profiling")]
macro_rules! evaluate {
    ($place:expr, $operation:expr, $func:expr ) => {
        crate::profiling::evaluate_fn($place, $operation, || $func)
    };
}

#[cfg(not(feature = "profiling"))]
macro_rules! evaluate {
    ($place:expr, $operation:expr, $func:expr ) => {
        $func
    };
}

pub(crate) use evaluate;
