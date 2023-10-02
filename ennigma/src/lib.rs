pub mod types;
pub use types::*;

pub use ::neuralnetworks::metrics::*;

pub mod trainer;
pub use trainer::*;

pub mod crypto {
    pub use crypto::*;
}
pub mod neuralnetworks {
    pub use ::neuralnetworks::*;
}
pub mod numbers {
    pub use numbers::*;
}

pub mod prelude;

#[cfg(test)]
#[cfg(feature = "x86_64")]
mod test {
    use std::{fs::File, io::Write, sync::Arc, time::Instant};

    use crypto::EncryptedContext;
    use neuralnetworks::{
        af::{self, AproxSigmoidAF},
        metrics::{AccuracyMetric, MeanSquareErrorMetric, Metric, QuadLossFunction},
        ActivationFn, ModelBuilder,
    };
    use numbers::{FromWithContext, Gen1DArray, Gen2DArray};

    use crate::{
        trainer::SharedCryptoTrainer, CryptoFFFFLayer, CryptoModel, EncryptedFixedPrecision,
    };

    #[test]
    fn test_encrypted_training() {
        let c = EncryptedContext::new();
        let ctx = Arc::new(c.remove_server_key());
        let mut rng = rand::thread_rng();
        let af = AproxSigmoidAF {};
        let loss_fn = QuadLossFunction {};

        // let server_ctx = Arc::new(ctx.get_server_context());
        let server_ctx = Arc::new(c.get_server_context());

        let mut model: CryptoModel<24, 12, 2, 1> =
            ModelBuilder::new(CryptoFFFFLayer::<24, 12, 8, 1, _, _>::new_random(
                &mut rng,
                af.clone(),
                loss_fn,
                &ctx,
            ))
            .add_layer(CryptoFFFFLayer::<24, 12, 2, 8, _, _>::new_random(
                &mut rng, af, loss_fn, &ctx,
            ))
            .build(&server_ctx);

        let trainer = SharedCryptoTrainer::new(&ctx);
        let metrics: Vec<
            Box<(dyn Metric<EncryptedFixedPrecision<24, 12>, EncryptedContext, 1> + 'static)>,
        > = vec![
            Box::new(MeanSquareErrorMetric {}),
            Box::new(AccuracyMetric {}),
        ];

        let input: Vec<Gen2DArray<EncryptedFixedPrecision<24, 12>, EncryptedContext, 1, 2>> = vec![
            FromWithContext::from_ctx(Gen1DArray::from_array([[0.0, 0.0]], &ctx), &ctx),
            FromWithContext::from_ctx(Gen1DArray::from_array([[0.0, 1.0]], &ctx), &ctx),
            FromWithContext::from_ctx(Gen1DArray::from_array([[1.0, 0.0]], &ctx), &ctx),
            FromWithContext::from_ctx(Gen1DArray::from_array([[1.0, 1.0]], &ctx), &ctx),
        ];
        let output = vec![
            FromWithContext::from_ctx(Gen1DArray::from_array([[0.0]], &ctx), &ctx),
            FromWithContext::from_ctx(Gen1DArray::from_array([[1.0]], &ctx), &ctx),
            FromWithContext::from_ctx(Gen1DArray::from_array([[1.0]], &ctx), &ctx),
            FromWithContext::from_ctx(Gen1DArray::from_array([[0.0]], &ctx), &ctx),
        ];

        println!("Serializing model");
        let mut f = File::create("model.json").expect("Should open file");
        let v = model.get_weights();
        serde_json::to_writer_pretty(&mut f, &v).expect("Should write");
        f.flush().expect("Should flush");

        println!("Training model");
        model.fit(
            100,
            4,
            FromWithContext::from_ctx(5.0, &server_ctx),
            (&input, &output),
            None,
            metrics,
            &trainer,
            &mut rng,
        );

        let mut expected = vec![];
        let mut results = vec![];

        let mut assertar = |input: [[f32; 2]; 1], output: [[f32; 1]; 1]| {
            let input = FromWithContext::from_ctx(Gen1DArray::from_array(input, &ctx), &ctx);
            let r: Gen2DArray<f32, EncryptedContext, 1, 1> = model.execute(&input).apply(|v| {
                let v: f32 = v.clone().switch_context(&ctx).into();
                v.round()
            });
            expected.push(r);
            results.push(Gen2DArray::from_array(output, &ctx));
        };

        assertar([[0.0, 0.0]], [[0.0]]);
        assertar([[1.0, 0.0]], [[1.0]]);
        assertar([[0.0, 1.0]], [[1.0]]);
        assertar([[1.0, 1.0]], [[0.0]]);

        println!("{:?}\n{:?}", expected, results);
    }

    #[test]
    fn test_sigmoidaprox() {
        let c = Arc::new(EncryptedContext::new().remove_server_key());

        let v: EncryptedFixedPrecision<16, 8> = FromWithContext::from_ctx(1.5, &c);

        let sigm: Box<dyn ActivationFn<EncryptedFixedPrecision<16, 8>, EncryptedContext, 1>> =
            Box::new(af::AproxSigmoidAF {});
        let sigmv2: Box<dyn ActivationFn<EncryptedFixedPrecision<16, 8>, EncryptedContext, 1>> =
            Box::new(af::AproxSigmoidAFV2 {});

        let start = Instant::now();
        let r = sigm.activate(&c, &v);
        println!("Duration V1: {}s", start.elapsed().as_secs());
        let start = Instant::now();
        let r = sigmv2.activate(&c, &v);
        println!("Duration V2: {}s", start.elapsed().as_secs());
    }
}
