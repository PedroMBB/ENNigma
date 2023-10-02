use std::{fs::File, io::Write, sync::Arc};

use crypto::{EncryptedBit, EncryptedContext};
use numbers::BooleanType;

use crate::types::{write_to_file_storage, ExecutionStorage, Executions};

pub fn test_encrypted_bits(repetitions: usize) {
    let ctx = Arc::new(EncryptedContext::new());
    let client_ctx = Arc::new(ctx.remove_server_key());
    let server_ctx = Arc::new(client_ctx.get_server_context());

    let mut storage = ExecutionStorage::default();

    execute_functions::<EncryptedBit, EncryptedContext>(
        &mut storage,
        &client_ctx,
        &server_ctx,
        repetitions,
    );
    execute_functions::<bool, ()>(&mut storage, &Arc::new(()), &Arc::new(()), repetitions);

    let mut file = File::create("bits.tsv").expect("Should create file");
    write_to_file_storage(&mut file, storage);
    file.flush().expect("Should flush");
}

fn execute_functions<T: BooleanType<C> + 'static, C>(
    storage: &mut ExecutionStorage,
    client_ctx: &Arc<C>,
    server_ctx: &Arc<C>,
    repetitions: usize,
) {
    let to_bool = move |v: &T| -> bool { v.clone().switch_context(client_ctx).into() };

    let mut exec = Executions {
        executions: vec![],
        mapper: &to_bool,
    };

    let name = format!("{}", std::any::type_name::<T>());
    let name_ass = format!("{}_ASSIGN", std::any::type_name::<T>());
    let name_clone_ass = format!("{}_CLONE_ASSIGN", std::any::type_name::<T>());

    // Sum
    for i in 0..repetitions {
        println!("[{}] Repetition {}/{}", name, i + 1, repetitions);

        let bit_0 = T::from_ctx(false, &client_ctx).switch_context(&server_ctx);
        let bit_1 = T::from_ctx(true, &client_ctx).switch_context(&server_ctx);

        exec.execute(&name, "NOT", "FAL", true, || bit_0.not_ref());
        exec.execute(&name, "NOT", "TRU", false, || bit_1.not_ref());

        exec.execute(&name, "AND", "T&T", true, || bit_1.and_ref(&bit_1));
        exec.execute(&name, "AND", "T&F", false, || bit_1.and_ref(&bit_0));
        exec.execute(&name, "AND", "F&T", false, || bit_0.and_ref(&bit_1));
        exec.execute(&name, "AND", "F&F", false, || bit_0.and_ref(&bit_0));

        exec.execute(&name, "OR", "T|T", true, || bit_1.or_ref(&bit_1));
        exec.execute(&name, "OR", "T|F", true, || bit_1.or_ref(&bit_0));
        exec.execute(&name, "OR", "F|T", true, || bit_0.or_ref(&bit_1));
        exec.execute(&name, "OR", "F|F", false, || bit_0.or_ref(&bit_0));

        exec.execute(&name, "XOR", "T^T", false, || bit_1.xor_ref(&bit_1));
        exec.execute(&name, "XOR", "T^F", true, || bit_1.xor_ref(&bit_0));
        exec.execute(&name, "XOR", "F^T", true, || bit_0.xor_ref(&bit_1));
        exec.execute(&name, "XOR", "F^F", false, || bit_0.xor_ref(&bit_0));

        exec.execute(&name, "NAND", "T?T", false, || bit_1.nand_ref(&bit_1));
        exec.execute(&name, "NAND", "T?F", true, || bit_1.nand_ref(&bit_0));
        exec.execute(&name, "NAND", "F?T", true, || bit_0.nand_ref(&bit_1));
        exec.execute(&name, "NAND", "F?F", true, || bit_0.nand_ref(&bit_0));

        exec.execute(&name, "NOR", "T?T", false, || bit_1.nor_ref(&bit_1));
        exec.execute(&name, "NOR", "T?F", false, || bit_1.nor_ref(&bit_0));
        exec.execute(&name, "NOR", "F?T", false, || bit_0.nor_ref(&bit_1));
        exec.execute(&name, "NOR", "F?F", true, || bit_0.nor_ref(&bit_0));

        exec.execute(&name, "XNOR", "T?T", true, || bit_1.xnor_ref(&bit_1));
        exec.execute(&name, "XNOR", "T?F", false, || bit_1.xnor_ref(&bit_0));
        exec.execute(&name, "XNOR", "F?T", false, || bit_0.xnor_ref(&bit_1));
        exec.execute(&name, "XNOR", "F?F", false, || bit_0.xnor_ref(&bit_0));

        exec.execute(&name, "MUX", "TTT", false, || bit_1.mux(&bit_1, &bit_1));
        exec.execute(&name, "MUX", "TTF", false, || bit_1.mux(&bit_1, &bit_0));
        exec.execute(&name, "MUX", "TFT", false, || bit_1.mux(&bit_0, &bit_1));
        exec.execute(&name, "MUX", "TFF", false, || bit_1.mux(&bit_0, &bit_0));
        exec.execute(&name, "MUX", "FTT", false, || bit_0.mux(&bit_1, &bit_1));
        exec.execute(&name, "MUX", "FTF", false, || bit_0.mux(&bit_1, &bit_0));
        exec.execute(&name, "MUX", "FFT", false, || bit_0.mux(&bit_0, &bit_1));
        exec.execute(&name, "MUX", "FFF", false, || bit_0.mux(&bit_0, &bit_0));

        let mut b = bit_0.clone();
        exec.execute(&name_ass, "NOT", "FAL", true, || {
            b.not_assign();
            b
        });
        let mut b = bit_1.clone();
        exec.execute(&name_ass, "NOT", "TRU", false, || {
            b.not_assign();
            b
        });

        let mut b = bit_1.clone();
        exec.execute(&name_ass, "AND", "T&T", true, || {
            b.and_assign(&bit_1);
            b
        });
        let mut b = bit_1.clone();
        exec.execute(&name_ass, "AND", "T&F", false, || {
            b.and_assign(&bit_0);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "AND", "F&T", false, || {
            b.and_assign(&bit_1);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "AND", "F&F", false, || {
            b.and_assign(&bit_0);
            b
        });

        let mut b = bit_1.clone();
        exec.execute(&name_ass, "OR", "T|T", true, || {
            b.or_assign(&bit_1);
            b
        });
        let mut b = bit_1.clone();
        exec.execute(&name_ass, "OR", "T|F", true, || {
            b.or_assign(&bit_0);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "OR", "F|T", true, || {
            b.or_assign(&bit_1);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "OR", "F|F", false, || {
            b.or_assign(&bit_0);
            b
        });

        let mut b = bit_1.clone();
        exec.execute(&name_ass, "XOR", "T^T", false, || {
            b.xor_assign(&bit_1);
            b
        });
        let mut b = bit_1.clone();
        exec.execute(&name_ass, "XOR", "T^F", true, || {
            b.xor_assign(&bit_0);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "XOR", "F^T", true, || {
            b.xor_assign(&bit_1);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "XOR", "F^F", false, || {
            b.xor_assign(&bit_0);
            b
        });

        let mut b = bit_1.clone();
        exec.execute(&name_ass, "NAND", "T?T", false, || {
            b.nand_assign(&bit_1);
            b
        });
        let mut b = bit_1.clone();
        exec.execute(&name_ass, "NAND", "T?F", true, || {
            b.nand_assign(&bit_0);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "NAND", "F?T", true, || {
            b.nand_assign(&bit_1);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "NAND", "F?F", true, || {
            b.nand_assign(&bit_0);
            b
        });

        let mut b = bit_1.clone();
        exec.execute(&name_ass, "NOR", "T?T", false, || {
            b.nor_assign(&bit_1);
            b
        });
        let mut b = bit_1.clone();
        exec.execute(&name_ass, "NOR", "T?F", false, || {
            b.nor_assign(&bit_0);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "NOR", "F?T", false, || {
            b.nor_assign(&bit_1);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "NOR", "F?F", true, || {
            b.nor_assign(&bit_0);
            b
        });

        let mut b = bit_1.clone();
        exec.execute(&name_ass, "XNOR", "T?T", true, || {
            b.xnor_assign(&bit_1);
            b
        });
        let mut b = bit_1.clone();
        exec.execute(&name_ass, "XNOR", "T?F", false, || {
            b.xnor_assign(&bit_0);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "XNOR", "F?T", false, || {
            b.xnor_assign(&bit_1);
            b
        });
        let mut b = bit_0.clone();
        exec.execute(&name_ass, "XNOR", "F?F", false, || {
            b.xnor_assign(&bit_0);
            b
        });

        // With Clone

        exec.execute(&name_clone_ass, "NOT", "FAL", true, || {
            let mut b = bit_0.clone();
            b.not_assign();
            b
        });
        exec.execute(&name_clone_ass, "NOT", "TRU", false, || {
            let mut b = bit_1.clone();
            b.not_assign();
            b
        });

        exec.execute(&name_clone_ass, "AND", "T&T", true, || {
            let mut b = bit_1.clone();
            b.and_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "AND", "T&F", false, || {
            let mut b = bit_1.clone();
            b.and_assign(&bit_0);
            b
        });
        exec.execute(&name_clone_ass, "AND", "F&T", false, || {
            let mut b = bit_0.clone();
            b.and_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "AND", "F&F", false, || {
            let mut b = bit_0.clone();
            b.and_assign(&bit_0);
            b
        });

        exec.execute(&name_clone_ass, "OR", "T|T", true, || {
            let mut b = bit_1.clone();
            b.or_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "OR", "T|F", true, || {
            let mut b = bit_1.clone();
            b.or_assign(&bit_0);
            b
        });
        exec.execute(&name_clone_ass, "OR", "F|T", true, || {
            let mut b = bit_0.clone();
            b.or_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "OR", "F|F", false, || {
            let mut b = bit_0.clone();
            b.or_assign(&bit_0);
            b
        });

        exec.execute(&name_clone_ass, "XOR", "T^T", false, || {
            let mut b = bit_1.clone();
            b.xor_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "XOR", "T^F", true, || {
            let mut b = bit_1.clone();
            b.xor_assign(&bit_0);
            b
        });
        exec.execute(&name_clone_ass, "XOR", "F^T", true, || {
            let mut b = bit_0.clone();
            b.xor_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "XOR", "F^F", false, || {
            let mut b = bit_0.clone();
            b.xor_assign(&bit_0);
            b
        });

        exec.execute(&name_clone_ass, "NAND", "T?T", false, || {
            let mut b = bit_1.clone();
            b.nand_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "NAND", "T?F", true, || {
            let mut b = bit_1.clone();
            b.nand_assign(&bit_0);
            b
        });
        exec.execute(&name_clone_ass, "NAND", "F?T", true, || {
            let mut b = bit_0.clone();
            b.nand_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "NAND", "F?F", true, || {
            let mut b = bit_0.clone();
            b.nand_assign(&bit_0);
            b
        });

        exec.execute(&name_clone_ass, "NOR", "T?T", false, || {
            let mut b = bit_1.clone();
            b.nor_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "NOR", "T?F", false, || {
            let mut b = bit_1.clone();
            b.nor_assign(&bit_0);
            b
        });
        exec.execute(&name_clone_ass, "NOR", "F?T", false, || {
            let mut b = bit_0.clone();
            b.nor_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "NOR", "F?F", true, || {
            let mut b = bit_0.clone();
            b.nor_assign(&bit_0);
            b
        });

        exec.execute(&name_clone_ass, "XNOR", "T?T", true, || {
            let mut b = bit_1.clone();
            b.xnor_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "XNOR", "T?F", false, || {
            let mut b = bit_1.clone();
            b.xnor_assign(&bit_0);
            b
        });
        exec.execute(&name_clone_ass, "XNOR", "F?T", false, || {
            let mut b = bit_0.clone();
            b.xnor_assign(&bit_1);
            b
        });
        exec.execute(&name_clone_ass, "XNOR", "F?F", false, || {
            let mut b = bit_0.clone();
            b.xnor_assign(&bit_0);
            b
        });
    }

    storage.add_executions(exec);
}
