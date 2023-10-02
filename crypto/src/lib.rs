mod cryptobit;
pub use cryptobit::*;

mod cryptofixedpoint;
pub use cryptofixedpoint::*;

mod cryptocontext;
pub use cryptocontext::*;

pub mod numbers {
    pub use numbers::*;
}

// TODO: Remove from here to a dedicated evaluation package
#[cfg(test)]
mod test {
    use std::{sync::Arc, time::Instant};

    use crate::{EncryptedBit, EncryptedContext};
    use numbers::{FixedPointNumber, FromWithContext, Gen2DArray};
    use tfhe::boolean::{
        gen_keys,
        prelude::Ciphertext,
        server_key::{BinaryBooleanGates, ServerKey},
    };

    pub type EncryptedFixedPrecision<const SIZE: usize, const PRECISION: usize> =
        FixedPointNumber<SIZE, PRECISION, EncryptedBit, EncryptedContext>;

    #[test]
    fn mult_enc() {
        type Num = EncryptedFixedPrecision<12, 2>;

        let ctx = Arc::new(EncryptedContext::new().remove_server_key());

        let arr1: Gen2DArray<Num, EncryptedContext, 7, 6> = FromWithContext::from_ctx(
            Gen2DArray::from_array(
                [
                    [12.0, 7.0, 10.0, 4.0, 5.0, 8.0],
                    [6.0, 11.0, 7.0, 10.0, 9.0, 5.0],
                    [10.0, 9.0, 8.0, 7.0, 6.0, 11.0],
                    [11.0, 5.0, 7.0, 8.0, 6.0, 10.0],
                    [7.0, 6.0, 11.0, 9.0, 8.0, 7.0],
                    [9.0, 7.0, 8.0, 14.0, 11.0, 6.0],
                    [8.0, 14.0, 10.0, 9.0, 7.0, 8.0],
                ],
                &ctx,
            ),
            &ctx,
        );

        let arr2: Gen2DArray<Num, EncryptedContext, 6, 7> = FromWithContext::from_ctx(
            Gen2DArray::from_array(
                [
                    [1.0, 5.0, 7.0, 9.0, 5.0, 6.0, 0.0],
                    [7.0, 2.0, 6.0, 2.0, 2.0, 9.0, 7.0],
                    [4.0, 5.0, 4.0, 5.0, 2.0, 3.0, 1.0],
                    [4.0, 2.0, 9.0, 4.0, 9.0, 7.0, 7.0],
                    [4.0, 7.0, 2.0, 7.0, 4.0, 0.0, 3.0],
                    [3.0, 5.0, 8.0, 0.0, 0.0, 7.0, 0.0],
                ],
                &ctx,
            ),
            &ctx,
        );

        let start = Instant::now();
        // let emc_received = &arr1 * &arr2; // 1520.1826776s
        let emc_received = arr1.mul_strassen(&arr2); // -
        println!(
            "Duration {}s",
            Instant::now().duration_since(start).as_secs_f64()
        );

        let received: Gen2DArray<f32, EncryptedContext, 7, 7> =
            emc_received.apply(|v| v.clone().into());

        received.apply(|v| {
            assert!(v - v == 0.0);
            ()
        });
    }

    #[test]
    fn mat_mult_enc() {
        let ctx = Arc::new(EncryptedContext::new().remove_server_key());

        let v1: EncryptedFixedPrecision<16, 4> = FromWithContext::from_ctx(8.0, &ctx);
        let v2: EncryptedFixedPrecision<16, 4> = FromWithContext::from_ctx(8.0, &ctx);

        let start = Instant::now();
        let r: FixedPointNumber<16, 4, EncryptedBit, EncryptedContext> = &v1 * &v2;
        println!(
            "Normal Mult: {}s, result {}",
            Instant::now().duration_since(start).as_secs_f64(),
            Into::<f32>::into(r)
        );

        let start = Instant::now();
        let r: FixedPointNumber<16, 4, EncryptedBit, EncryptedContext> =
            numbers::karatsuba_adapt(&v1, &v2);
        println!(
            "Normal Mult: {}s, result {}",
            Instant::now().duration_since(start).as_secs_f64(),
            Into::<f32>::into(r)
        );
    }

    #[test]
    fn mat_mult_enc_small() {
        let ctx = Arc::new(EncryptedContext::new().remove_server_key());

        let arr1: Gen2DArray<EncryptedFixedPrecision<10, 2>, EncryptedContext, 2, 3> =
            FromWithContext::from_ctx(
                Gen2DArray::from_array([[5.0, 3.0, 7.0], [4.0, 7.0, 3.0]], &ctx),
                &ctx,
            );

        let arr2: Gen2DArray<EncryptedFixedPrecision<10, 2>, EncryptedContext, 3, 1> =
            FromWithContext::from_ctx(Gen2DArray::from_array([[4.0], [0.0], [9.0]], &ctx), &ctx);

        let exp: Gen2DArray<f32, EncryptedContext, 2, 1> =
            Gen2DArray::from_array([[83.0], [43.0]], &ctx);

        println!("Calculating");

        let start = Instant::now();
        // let emc_received = &arr1 * &arr2;
        let emc_received = arr1.mul_opt(&arr2);
        // let emc_received = arr1.mul_strassen(&arr2);
        println!(
            "Duration {}s",
            Instant::now().duration_since(start).as_secs_f64()
        );

        let received: Gen2DArray<f32, EncryptedContext, 2, 1> =
            emc_received.apply(|v| v.clone().into());

        received.apply_zip(&exp, |(v, exp)| {
            assert_eq!(v, exp);
            ()
        });
    }

    #[test]
    fn enc_operations() {
        let ctx = Arc::new(EncryptedContext::new().remove_server_key());

        type N = EncryptedFixedPrecision<64, 32>;

        let v1: N = FromWithContext::from_ctx(1.0, &ctx);
        let v2: N = FromWithContext::from_ctx(1.0, &ctx);

        fn make_op<F, O>(name: &str, op: F)
        where
            F: Fn() -> O,
        {
            let mut dur = 0_f64;
            for i in 0..3 {
                println!("Operation {}: {}/3", name, i);
                let start = Instant::now();
                op();
                dur += start.elapsed().as_secs_f64();
            }
            println!("Operation {}: {}s", name, dur / 3.0);
        }

        make_op("Addition", || v1.clone() + &v2);
        make_op("Multiplication", || v1.clone() * &v2);
        // make_op("Division", || &v1 / &v2);
        make_op("Exp", || v1.clone().pow_2());
    }

    #[test]
    fn compare_operations() {
        let (ck, sk) = gen_keys();

        let v1 = ck.encrypt(false);
        let v2 = ck.encrypt(true);

        let test = |name: &str, op: fn(&Ciphertext, &Ciphertext, sk: &ServerKey) -> Ciphertext| {
            let mut v = v1.clone();
            let start = Instant::now();
            for _ in 0..1_000 {
                v = op(&v, &v2, &sk);
            }
            let dur = start.elapsed().as_secs_f64();
            println!("{}: Duration {}s, final val: {}", name, dur, ck.decrypt(&v));
        };

        test("And", |v1, v2, sk| sk.and(v1, v2));
        test("Or", |v1, v2, sk| sk.or(v1, v2));
        test("Xor", |v1, v2, sk| sk.xor(v1, v2));
        test("Nand", |v1, v2, sk| sk.nand(v1, v2));
        test("Xnor", |v1, v2, sk| sk.xnor(v1, v2));
        test("Nor", |v1, v2, sk| sk.nor(v1, v2));
    }
}

// New = 33s
// New = 34s
