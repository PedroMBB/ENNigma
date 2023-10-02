#[cfg(feature = "plaintext")]
mod plaintext;
#[cfg(feature = "plaintext")]
pub use plaintext::*;

#[cfg(not(feature = "plaintext"))]
mod ciphertext;
#[cfg(not(feature = "plaintext"))]
pub use ciphertext::*;

#[cfg(feature = "neuralnetwork")]
pub use neuralnetworks::ModelBuilder;
pub use numbers::FromWithContext;
