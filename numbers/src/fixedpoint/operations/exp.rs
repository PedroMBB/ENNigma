use crate::{BooleanType, FixedPointNumber, FromWithContext};

impl<const SIZE: usize, const PRECISION: usize, T: BooleanType<C>, C>
    FixedPointNumber<SIZE, PRECISION, T, C>
{
    pub fn pow_2(self) -> Self {
        let n_int = SIZE - PRECISION - 1;
        let false_v = T::from_ctx(false, &self.context);
        let true_v = T::from_ctx(true, &self.context);

        let numb_int: Vec<_> = self.bits.iter().skip(PRECISION).take(n_int).collect();

        let msb = self.msb();
        let mut n_msb = msb.clone();
        n_msb.not_assign();

        let mut int_final: Vec<_> = (0..SIZE).map(|_| false_v.clone()).collect();

        for i in 0..(n_int as isize) {
            let i_val = Self::from_ctx((i as u16).into(), &self.context);
            let i_lst: Vec<_> = i_val
                .bits
                .iter()
                .skip(PRECISION)
                .take(n_int)
                .map(|v| T::xor(v, &msb))
                .collect();

            let eq = numb_int
                .iter()
                .zip(i_lst.iter())
                .map(|(v1, v2)| T::xnor_ref(*v1, v2))
                .reduce(|mut a, b| {
                    T::and_assign(&mut a, &b);
                    a
                })
                .expect("Number of int bits should be, at least, 1");

            let pos_2_pos = PRECISION + i as usize;
            let pos_v: Vec<_> = (0..SIZE)
                .map(|v: usize| match v == pos_2_pos {
                    false => &false_v,
                    true => &true_v,
                })
                .collect();

            let neg_2_pos = 0_usize.max((PRECISION as isize - i - 1) as usize);
            let neg_v: Vec<_> = (0..SIZE)
                .map(|v: usize| match v == neg_2_pos {
                    false => &false_v,
                    true => &true_v,
                })
                .collect();

            let val: Vec<_> = neg_v
                .into_iter()
                .zip(pos_v.into_iter())
                .map(|(n, p)| {
                    let n_v = T::and(msb, n);
                    let p_v = T::and(&n_msb, p);

                    n_v | p_v
                })
                .collect();

            let int_result: Vec<_> = val.iter().map(|v1| T::and(v1, &eq)).collect();

            int_final
                .iter_mut()
                .zip(int_result)
                .for_each(|(final_v, curr)| *final_v = T::or(final_v, &curr));
        }

        int_final[SIZE - 1] = false_v.clone();

        Self::from_bits(int_final, &self.context)
    }
}
#[cfg(test)]
mod tests {
    use std::{fs::File, sync::Arc};

    use crate::{FixedPointNumber, FromWithContext};

    #[test]
    fn f32_exp() {
        use std::io::Write;

        fn calc<const S: usize, const P: usize>(file: &mut File, v: f32) {
            let ctx = Arc::new(());
            let real = 2_f32.powf(v);

            let f1: FixedPointNumber<S, P, bool, ()> = FixedPointNumber::from_ctx(v, &ctx);
            println!("Original {:?}", f1.bits);
            let enc_res = f1.pow_2();

            let f2: FixedPointNumber<S, P, bool, ()> = FixedPointNumber::from_ctx(real, &ctx);
            println!("Expected {:?}", f2.bits);
            println!("received {:?}", enc_res.bits);

            let res: f32 = enc_res.into();

            writeln!(file, "{}\t{}\t{}", v, real, res).unwrap();

            // assert!(
            //     (res - real).abs() < 0.01,
            //     "f32_adition 2^{} = {} but received {} (diff: {})",
            //     v,
            //     2_f32.powf(v),
            //     res,
            //     (res - real).abs()
            // );
        }
        let mut file = File::create("exp.tsv").unwrap();

        // calc::<12, 6>(&mut file, 3.0);
        writeln!(&mut file, "x\tExpected\treceived").unwrap();
        let mut v = -8.0;
        while v < 8.0 {
            calc::<32, 16>(&mut file, v);

            v += 0.1;
        }
        file.flush().unwrap();
    }

    #[test]
    fn f32_exp_int() {
        use std::io::Write;

        fn calc<const S: usize, const P: usize>(_file: &mut File, v: f32) {
            let ctx = Arc::new(());
            let real: f32 = 2_f32.powf(v);

            let f1: FixedPointNumber<S, P, bool, ()> = FixedPointNumber::from_ctx(v, &ctx);
            // println!("Original {:?}", f1.bits);
            let enc_res = f1.pow_2();

            // let f2: FixedPointNumber<S, P, bool, ()> = FixedPointNumber::from_ctx(real, &ctx);
            // println!("Expected {:?}", f2.bits);
            // println!("received {:?}", enc_res.bits);

            let res: f32 = enc_res.into();

            // writeln!(file, "{}\t{}\t{}", v, real, res).unwrap();

            println!("{}: Rec {}, Exp {}", v, res, real);

            // assert!(
            //     (res - real).abs() < 0.01,
            //     "f32_adition 2^{} = {} but received {} (diff: {})",
            //     v,
            //     2_f32.powf(v),
            //     res,
            //     (res - real).abs()
            // );
        }
        let mut file = File::create("exp.tsv").unwrap();

        // calc::<12, 6>(&mut file, 3.0);
        writeln!(&mut file, "x\tExpected\treceived").unwrap();
        let mut v = -8.0;
        while v < 8.0 {
            calc::<64, 32>(&mut file, v);

            v += 1.0;
        }
        file.flush().unwrap();
    }
}
