use std::{fs::File, path::MAIN_SEPARATOR_STR, sync::Arc};

use ennigma::prelude::*;

pub fn dataset_from_csv<
    const SIZE: usize,
    const PRECISION: usize,
    const INPUT: usize,
    const OUTPUT: usize,
>(
    base_path: &str,
    train_file: &str,
    test_file: &str,
    client_ctx: &Arc<ContextType>,
    server_ctx: &Arc<ContextType>,
) -> (
    (
        Vec<Gen1DArrayType<SIZE, PRECISION, INPUT>>,
        Vec<Gen1DArrayType<SIZE, PRECISION, OUTPUT>>,
    ),
    Vec<(
        Gen1DArrayType<SIZE, PRECISION, INPUT>,
        Gen1DArrayType<SIZE, PRECISION, OUTPUT>,
    )>,
) {
    println!("[CSV Loader] Loading training data");

    let train_file_path = format!("{}{}{}", base_path, MAIN_SEPARATOR_STR, train_file);
    let test_file_path = format!("{}{}{}", base_path, MAIN_SEPARATOR_STR, test_file);

    let mut input: Vec<Gen1DArrayType<SIZE, PRECISION, INPUT>> = vec![];
    let mut output: Vec<Gen1DArrayType<SIZE, PRECISION, OUTPUT>> = vec![];

    let file = File::open(&train_file_path).expect("Should open train file");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);
    for result in rdr.records() {
        let record = result.expect("Should read train record");

        let m = |r: &csv::StringRecord, i: usize| -> NumberType<SIZE, PRECISION> {
            let v = r
                .get(i)
                .map(|v| v.parse::<f32>().ok())
                .flatten()
                .expect(&format!("Should have field {}", i));
            let v: NumberType<SIZE, PRECISION> = FromWithContext::from_ctx(v, client_ctx);
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

    let mut val: Vec<(
        Gen1DArrayType<SIZE, PRECISION, INPUT>,
        Gen1DArrayType<SIZE, PRECISION, OUTPUT>,
    )> = vec![];

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&test_file_path)
        .expect("Should open test file");
    for result in rdr.records() {
        let record = result.expect("Should read test record");

        let m = |r: &csv::StringRecord, i: usize| -> NumberType<SIZE, PRECISION> {
            let v = r
                .get(i)
                .map(|v| v.parse::<f32>().ok())
                .flatten()
                .expect(&format!("Should have field {}", i));
            let v: NumberType<SIZE, PRECISION> = FromWithContext::from_ctx(v, client_ctx);
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
