use ennigma::neuralnetworks::metrics::Metric;
use ennigma::neuralnetworks::metrics::QuadLossFunction;
use ennigma::neuralnetworks::trainers::{
    ConsoleTrainer, FileTrainer, ForwardTrainerBuilder, StoreModelTrainer,
};
use ennigma::neuralnetworks::SerializableModel;
use ennigma::{
    prelude::*, DecryptionTrainer, FixedPrecisionAccuracyMetric,
    FixedPrecisionMeanSquareErrorMetric,
};
use rand::SeedableRng;
use std::fs::File;

const INPUT: usize = 2;
const OUTPUT: usize = 1;

const PRECISION: usize = 12;
const SIZE: usize = 4 + PRECISION + 1;
const LAYER_1: usize = 4 + PRECISION + 1;
const LAYER_2: usize = 4 + PRECISION + 1;

const METRICS_BITS: usize = 18;
const METRICS_PRECISION: usize = 12;

fn main() {
    if let Err(e) = dotenv::dotenv() {
        println!("Could not load an environment file: {:?}", e)
    }
    execute();
}

fn execute() {
    // Seed - Test Accuracy
    // 1956581677289523570 - 100%

    let seed: u64 = match std::env::var("SEED") {
        Err(_) => {
            use rand::RngCore;
            let mut r = rand::thread_rng();
            r.next_u64()
        }
        Ok(v) => v.parse().expect("Should receive a valid SEED"),
    };
    println!("SEED '{}'", seed);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    let (client_ctx, server_ctx) = generate_context();

    // let af = SigmoidPowAproxAF::<SIZE, PRECISION>::default();
    // let af = SigmoidLSTruncAproxAF::<SIZE, PRECISION>::default();
    // let af = ReLUAF::default();
    // let af = ennigma::af::ReLUTruncAF,ReLUTruncAF::default();
    let af = ennigma::neuralnetworks::af::SigmoidMyersAproxAF::default();

    let loss_fn = QuadLossFunction {};

    let map = |v: f32| -> NumberType<SIZE, PRECISION> {
        let v: NumberType<SIZE, PRECISION> = FromWithContext::from_ctx(v, &client_ctx);
        v.switch_context(&server_ctx)
    };

    let input: Vec<Gen1DArrayType<SIZE, PRECISION, INPUT>> = vec![
        Gen1DArrayType::from_array([[map(0.0), map(0.0)]], &server_ctx),
        Gen1DArrayType::from_array([[map(0.0), map(1.0)]], &server_ctx),
        Gen1DArrayType::from_array([[map(1.0), map(0.0)]], &server_ctx),
        Gen1DArrayType::from_array([[map(1.0), map(1.0)]], &server_ctx),
    ];
    let output: Vec<Gen1DArrayType<SIZE, PRECISION, OUTPUT>> = vec![
        Gen1DArrayType::from_array([[map(0.0)]], &server_ctx),
        Gen1DArrayType::from_array([[map(1.0)]], &server_ctx),
        Gen1DArrayType::from_array([[map(1.0)]], &server_ctx),
        Gen1DArrayType::from_array([[map(0.0)]], &server_ctx),
    ];

    let mut model: ModelType<SIZE, PRECISION, INPUT, OUTPUT> =
        ModelBuilder::new(
            FFFFLayerType::<SIZE, LAYER_1, PRECISION, 8, OUTPUT, _, _>::new_random(
                &mut rng,
                af.clone(),
                loss_fn,
                &server_ctx,
            ),
        )
        .add_layer(
            FFFFLayerType::<SIZE, LAYER_2, PRECISION, INPUT, 8, _, _>::new_random(
                &mut rng,
                af,
                loss_fn,
                &server_ctx,
            ),
        )
        .build(&server_ctx);

    let trainer_b = ForwardTrainerBuilder::new()
        .add_trainer(ConsoleTrainer::new())
        .add_trainer(StoreModelTrainer::new("xor", "xor"))
        .add_trainer(FileTrainer::new("xor", "xor_train.txt"));
    let trainer = DecryptionTrainer::new(trainer_b.build(), &client_ctx);

    let metrics: Vec<Box<(dyn Metric<NumberType<SIZE, PRECISION>, ContextType, 1> + 'static)>> = vec![
        Box::new(FixedPrecisionMeanSquareErrorMetric::<
            METRICS_BITS,
            METRICS_PRECISION,
        >::new()),
        Box::new(FixedPrecisionAccuracyMetric::<
            METRICS_BITS,
            METRICS_PRECISION,
        >::new()),
    ];

    let _metrics = model.fit(
        75,
        18,
        1,
        (&input, &output),
        None,
        metrics,
        &trainer,
        &mut rng,
    );

    let mut f = File::create("xor.json").expect("Should create model file");
    serde_json::to_writer(&mut f, &model.get_weights()).expect("Should write to model file");

    println!("SEED '{}'", seed);
}
