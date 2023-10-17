use std::{fs::File, io::Write, sync::Arc};

use crypto::{EncryptedContext, EncryptedFixedPointNumber};
use neuralnetworks::{
    af::{SigmoidMyersAproxAF, SigmoidMyersMulAproxAF},
    ActivationFn,
};
use numbers::FromWithContext;

use crate::types::{write_to_file_storage, ExecutionStorage, Executions};

pub fn test_mayer_af(repetitions: usize) {
    let ctx = Arc::new(EncryptedContext::new());
    let client_ctx = Arc::new(ctx.remove_server_key());
    let server_ctx = Arc::new(client_ctx.get_server_context());

    let mut storage = ExecutionStorage::default();

    execute_functions::<8, 4>(&mut storage, &client_ctx, &server_ctx, repetitions);
    execute_functions::<16, 8>(&mut storage, &client_ctx, &server_ctx, repetitions);
    execute_functions::<32, 16>(&mut storage, &client_ctx, &server_ctx, repetitions);
    execute_functions::<64, 32>(&mut storage, &client_ctx, &server_ctx, repetitions);

    let mut file = File::create("mayer.tsv").expect("Should create file");
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

    let name = format!("MAYER<{},{}>", S, P);

    let mayer_shft = SigmoidMyersAproxAF::default();
    let mayer_mul = SigmoidMyersMulAproxAF::default();

    // Sum
    for i in 0..repetitions {
        println!("[{}] Repetition {}/{}", name, i + 1, repetitions);

        let pv1 = 2.0_f32;
        let pv2 = 1.5_f32;

        let v1 = EncryptedFixedPointNumber::from_ctx(pv1, client_ctx).switch_context(server_ctx);
        let v2 = EncryptedFixedPointNumber::from_ctx(pv2, client_ctx).switch_context(server_ctx);

        exec.execute(&name, "SHIFT_ACT", "AIO", 0.0, || {
            <SigmoidMyersAproxAF as ActivationFn<EncryptedFixedPointNumber<S, P>, 1>>::activate(
                &mayer_shft,
                server_ctx,
                v1.clone(),
            );
            v2.clone()
        });
        exec.execute(&name, "SHIFT_DER", "AIO", 0.0, || {
            <SigmoidMyersAproxAF as ActivationFn<
                EncryptedFixedPointNumber<S, P>,
                1,
            >>::activate_and_derivative(&mayer_shft, server_ctx, v2.clone());
            v1.clone()
        });

        exec.execute(&name, "MUL_ACT", "AIO", 0.0, || {
            <SigmoidMyersMulAproxAF as ActivationFn<EncryptedFixedPointNumber<S, P>, 1>>::activate(
                &mayer_mul,
                server_ctx,
                v1.clone(),
            );
            v2.clone()
        });
        exec.execute(&name, "MUL_DER", "AIO", 0.0, || {
            <SigmoidMyersMulAproxAF as ActivationFn<
                EncryptedFixedPointNumber<S, P>,
                1,
            >>::activate_and_derivative(&mayer_mul, server_ctx, v2.clone());
            v1.clone()
        });
    }

    storage.add_executions(exec);
}
