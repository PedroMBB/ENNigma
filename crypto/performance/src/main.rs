use clap::Parser;

mod types;

mod bitoperations;
mod fixedpoint;
mod nn;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None, arg_required_else_help = true)]
struct Arguments {
    #[arg(short, long, default_value_t = 0)]
    encfp_count: usize,
    #[arg(short, long, default_value_t = 0)]
    result_test: usize,
    #[arg(short, long, default_value_t = 0)]
    bits_operations: usize,
    #[arg(short, long, default_value_t = 0)]
    mayer_performance: usize,
}

fn main() {
    let args: Arguments = Arguments::parse();

    if args.encfp_count > 0 {
        fixedpoint::test_encrypted_operations(args.encfp_count);
    }
    if args.result_test > 0 {
        fixedpoint::test_encrypted_results(args.result_test);
    }
    if args.bits_operations > 0 {
        bitoperations::test_encrypted_bits(args.bits_operations);
    }
    if args.mayer_performance > 0 {
        nn::test_mayer_af(args.mayer_performance);
    }
}
