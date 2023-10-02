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
use examples::dataset_from_csv;
use rand::SeedableRng;
use std::fs::File;

const INPUT: usize = 7;
const OUTPUT: usize = 1;

const PRECISION: usize = 22;
const SIZE: usize = 5 + PRECISION + 1;
const LAYER_1: usize = 5 + PRECISION + 1;
// const LAYER_2: usize = 5 + PRECISION + 1;

const METRICS_BITS: usize = 48;
const METRICS_PRECISION: usize = 28;

fn main() {
    if let Err(e) = dotenv::dotenv() {
        println!("Could not load an environment file: {:?}", e)
    }
    execute();
}

fn execute() {
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
    // let af = ennigma::neuralnetworks::af::ReLUTruncAF::ReLUTruncAF::default();
    let af = ennigma::neuralnetworks::af::SigmoidMyersAproxAF::default();

    let loss_fn = QuadLossFunction {};

    let (train, val) = dataset_from_csv::<SIZE, PRECISION, INPUT, OUTPUT>(
        "../../../datasets/cicddos2019",
        "udp.train.csv",
        "udp.test.csv",
        &client_ctx,
        &server_ctx,
    );

    let mut model: ModelType<SIZE, PRECISION, INPUT, OUTPUT> =
        ModelBuilder::new(
            FFFFLayerType::<SIZE, LAYER_1, PRECISION, 7, OUTPUT, _, _>::new_random(
                &mut rng,
                af.clone(),
                loss_fn,
                &server_ctx,
            ),
        )
        // .add_layer(
        //     FFFFLayerType::<SIZE, LAYER_2, PRECISION, INPUT, 7, _, _>::new_random(
        //         &mut rng,
        //         af,
        //         loss_fn,
        //         &server_ctx,
        //     ),
        // )
        .build(&server_ctx);

    let trainer_b = ForwardTrainerBuilder::new()
        .add_trainer(ConsoleTrainer::new())
        .add_trainer(StoreModelTrainer::new("ddos", "ddos"))
        .add_trainer(FileTrainer::new("ddos", "ddos_train.txt"));
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
        5,
        24 * 10,
        -4,
        (&train.0, &train.1),
        Some(val),
        metrics,
        &trainer,
        &mut rng,
    );

    let mut f = File::create("ddos.json").expect("Should create model file");
    serde_json::to_writer(&mut f, &model.get_weights()).expect("Should write to model file");

    println!("SEED '{}'", seed);
}
