use std::{fs::File, path::MAIN_SEPARATOR_STR, sync::Arc};

use ennigma::{
    numbers::{Gen1DArray, NumberType, SwitchContext},
    prelude::*,
};

pub fn dataset_from_csv<T: NumberType, const INPUT: usize, const OUTPUT: usize>(
    base_path: &str,
    train_file: &str,
    test_file: &str,
    client_ctx: &Arc<T::ContextType>,
    server_ctx: &Arc<T::ContextType>,
) -> (
    (Vec<Gen1DArray<T, INPUT>>, Vec<Gen1DArray<T, OUTPUT>>),
    Vec<(Gen1DArray<T, INPUT>, Gen1DArray<T, OUTPUT>)>,
)
where
    T: SwitchContext<T::ContextType>,
    T: FromWithContext<f32, T::ContextType>,
{
    println!("[CSV Loader] Loading training data");

    let train_file_path = format!("{}{}{}", base_path, MAIN_SEPARATOR_STR, train_file);
    let test_file_path = format!("{}{}{}", base_path, MAIN_SEPARATOR_STR, test_file);

    let mut input: Vec<Gen1DArray<T, INPUT>> = vec![];
    let mut output: Vec<Gen1DArray<T, OUTPUT>> = vec![];

    let file = File::open(&train_file_path).expect("Should open train file");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);
    for result in rdr.records() {
        let record = result.expect("Should read train record");

        let m = |r: &csv::StringRecord, i: usize| -> T {
            let v = r
                .get(i)
                .map(|v| v.parse::<f32>().ok())
                .flatten()
                .expect(&format!("Should have field {}", i));
            let v: T = FromWithContext::from_ctx(v, client_ctx);
            v.switch_context(server_ctx)
        };

        input.push(FromWithContext::from_ctx(
            (0..INPUT)
                .into_iter()
                .map(|v| m(&record, v))
                .collect::<Vec<_>>(),
            &client_ctx,
        ));
        output.push(FromWithContext::from_ctx(
            (0..OUTPUT)
                .into_iter()
                .map(|v| m(&record, INPUT + v))
                .collect::<Vec<_>>(),
            &client_ctx,
        ));
    }

    println!("[CSV Loader] Loading testing data");

    let mut val: Vec<(Gen1DArray<T, INPUT>, Gen1DArray<T, OUTPUT>)> = vec![];

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&test_file_path)
        .expect("Should open test file");
    for result in rdr.records() {
        let record = result.expect("Should read test record");

        let m = |r: &csv::StringRecord, i: usize| -> T {
            let v = r
                .get(i)
                .map(|v| v.parse::<f32>().ok())
                .flatten()
                .expect(&format!("Should have field {}", i));
            let v: T = FromWithContext::from_ctx(v, client_ctx);
            v.switch_context(server_ctx)
        };

        val.push((
            FromWithContext::from_ctx(
                (0..INPUT)
                    .into_iter()
                    .map(|v| m(&record, v))
                    .collect::<Vec<_>>(),
                &client_ctx,
            ),
            FromWithContext::from_ctx(
                (0..OUTPUT)
                    .into_iter()
                    .map(|v| m(&record, INPUT + v))
                    .collect::<Vec<_>>(),
                &client_ctx,
            ),
        ));
    }

    println!(
        "[CSV Loader] Data loaded (Train size: {}, Test size: {})",
        input.len(),
        val.len(),
    );

    ((input, output), val)
}
