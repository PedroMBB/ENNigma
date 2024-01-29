# **ENNigma** - Data Privacy Preservation in Neural Network using Homomorphic Encryption

This is the main git repositorio of ENNigma, an open-source framework to create a neural network models which use Homomorphic Encryption (HE) for ensuring data privacy.

## What is ENNigma

ENNigma is a framework that ensures data privacy in NN using Homomorphic Encryption (HE). It is designed to perform three key operations: creating models, training them, and classifying data. The framework is optimized to reduce the impact of HE usage on the computational complexity and classification performance of the final model, while still prioritizing privacy preservation.

A significant aspect of ENNigma is its aim to serve as a groundwork for future research in the field of privacy preservation in neural networks using HE. ENNigma addresses the gap in this research area by being available. While it an ideal solution without any compromises might be beyond our current technological reach, ENNigma's primary goal is to guide and accelerate future research in this field, making it more cohesive and efficient.

## Key features

- **Modular architecture**: ENNigma is designed with a modular structure, which facilitates more autonomous development and lessens the interdependence within the final solution. This modular approach not only enables independent development but also ensures that future modifications can be implemented more easily and rapidly.

- **Ready to use** - By providing a pre-built neural network framework, researchers are spared the need to create a neural network from the ground up for each new project. Consequently, they can dedicate more time to devising creative approaches to further progress in this field of research.

- **Easy to use** - ENNigma's API was conceived to be user-friendly interface, especially for those familiar with tools like TensorFlow. Its API bears similarities to these tools, reducing the learning curve for creating NN models.

- **Highly configurable** - The framework accommodates the unique hyper-parameters of different NN models through its API, eliminating the need for code modifications for new model structures. This flexibility applies to all its functionalities, including model creation and training.

- **No Server-Client Communication** - While this some can see this as a drawback, ENNigma does not inherently include client-server communication capabilities because its focused is on improving NN computational itself. However, if latter it reaches a point that it can be used in real-world scenarios, its modular architecture allows for the easy integration of such features if required for specific experiments or use cases.

## Getting Started

### Pre-requisites

- Supported Rust version installed on the system
- If executing some of the examples, the csv file with the dataset to run the examples

### Configuring

The execution of ENNigma can be configured using some environment variables and features flags. For example, by using the NUM_THREADS environment variable, the number of threads created by ENNigma can be configured.

### Usage

```bash
git clone https://github.com/PedroMBB/ENNigma.git
cd ENNigma

# Running banknote example
cargo run --release --bin banknote
# Running banknote example in plaintext
cargo run --release --bin banknote --features plaintext
```

## License

This software is distributed under the LGPL-3.0 license.

## Resources

### Master's Thesis

Barbosa, P. (2023). *ENNigma: A Framework for Private Neural Networks* [Master's thesis, Polytechnic Institute of Porto, School of Engineering]. [http://hdl.handle.net/10400.22/23994](http://hdl.handle.net/10400.22/23994)
