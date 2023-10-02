use std::{fs::File, io::Write, sync::Arc};

use crypto::{EncryptedContext, EncryptedFixedPointNumber};
use numbers::{Abs, FromWithContext, IsGreater};

use crate::types::{write_to_file_storage, ExecutionStorage, Executions};

pub fn test_encrypted_results(repetitions: usize) {
    let ctx = Arc::new(EncryptedContext::new());
    let client_ctx = Arc::new(ctx.remove_server_key());
    let server_ctx = Arc::new(client_ctx.get_server_context());

    let v1 = -1.5;
    let v2 = 2.25;

    let enc1: EncryptedFixedPointNumber<8, 4> =
        EncryptedFixedPointNumber::from_ctx(v1, &client_ctx).switch_context(&server_ctx);
    let enc2: EncryptedFixedPointNumber<8, 4> =
        EncryptedFixedPointNumber::from_ctx(v2, &client_ctx).switch_context(&server_ctx);

    fn test_op(
        client_ctx: &Arc<EncryptedContext>,
        expected: f32,
        operation: impl FnOnce() -> EncryptedFixedPointNumber<8, 4>,
    ) {
        let results: f32 = (operation)().switch_context(client_ctx).into();
        println!("[Result] Expected: {}, Obtained: {}", expected, results);
        assert!((results - expected).abs() < 0.125);
    }

    for i in 0..repetitions {
        println!("Running result test {}", i);
        println!("Sum");
        test_op(&client_ctx, v1 + v2, || enc1.clone() + &enc2);
        println!("Sub");
        test_op(&client_ctx, v1 - v2, || enc1.clone() - &enc2);
        println!("Mul");
        test_op(&client_ctx, v1 * v2, || enc1.clone() * &enc2);
        println!("Div");
        test_op(&client_ctx, v1 / v2, || enc1.clone() / &enc2);
    }
}

pub fn test_encrypted_operations(repetitions: usize) {
    let ctx = Arc::new(EncryptedContext::new());
    let client_ctx = Arc::new(ctx.remove_server_key());
    let server_ctx = Arc::new(client_ctx.get_server_context());

    let mut storage = ExecutionStorage::default();

    execute_functions::<8, 4>(&mut storage, &client_ctx, &server_ctx, repetitions);
    execute_functions::<16, 8>(&mut storage, &client_ctx, &server_ctx, repetitions);
    execute_functions::<32, 16>(&mut storage, &client_ctx, &server_ctx, repetitions);
    execute_functions::<64, 32>(&mut storage, &client_ctx, &server_ctx, repetitions);

    let mut file = File::create("performance.tsv").expect("Should create file");
    write_to_file_storage(&mut file, storage);
    file.flush().expect("Should flush");
}

fn execute_functions<const S: usize, const P: usize>(
    storage: &mut ExecutionStorage,
    client_ctx: &Arc<EncryptedContext>,
    server_ctx: &Arc<EncryptedContext>,
    repetitions: usize,
) {
    let to_fixed_point =
        move |v: &EncryptedFixedPointNumber<S, P>| v.clone().switch_context(client_ctx).into();

    let mut exec = Executions {
        executions: vec![],
        mapper: &to_fixed_point,
    };

    let name = format!("FP<{},{}>", S, P);

    // Sum
    for i in 0..repetitions {
        println!("[{}] Repetition {}/{}", name, i + 1, repetitions);

        let pv1 = 2.0_f32;
        let pv2 = 1.5_f32;

        let v1 = EncryptedFixedPointNumber::from_ctx(pv1, client_ctx).switch_context(server_ctx);
        let v2 = EncryptedFixedPointNumber::from_ctx(pv2, client_ctx).switch_context(server_ctx);

        exec.execute(&name, "SUM", "AIO", pv1 + pv2, || v1.clone() + &v2);
        exec.execute(&name, "SUB", "AIO", pv1 - pv2, || v1.clone() - &v2);
        exec.execute(&name, "MUL", "AIO", pv1 * pv2, || v1.clone() * &v2);
        // exec.execute(&name, "DIV", "AIO", pv1 / pv2, || v1.clone() / &v2);
        exec.execute(&name, "ABS", "AIO", pv1.abs(), || v1.clone().abs());
        exec.execute(
            &name,
            "GRE",
            "AIO",
            match (pv1 - pv2) > 0.0 {
                true => 1.0,
                false => 0.0,
            },
            || v1.is_greater(&v2),
        );
        exec.execute(&name, "SHR", "AIO", pv1 * 2.0_f32.powi(2), || {
            &v1 >> 2_usize
        });
        exec.execute(&name, "SHL", "AIO", pv1 / 2.0_f32.powi(2), || {
            &v1 << 2_usize
        });
        exec.execute(&name, "TRU", "AIO", pv1.trunc(), || v1.clone().truncate());
        exec.execute(&name, "NEG", "AIO", pv1 * -1.0, || !v1.clone());

        exec.execute(&name, "GRT", "2>1.5", 1.0, || v1.clone().is_greater(&v2));
        exec.execute(&name, "GRT", "1.5>2", 1.0, || v2.clone().is_greater(&v1));
    }

    storage.add_executions(exec);
}
