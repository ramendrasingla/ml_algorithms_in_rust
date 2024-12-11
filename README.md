# ML Algorithms in Rust

Welcome to **ML Algorithms in Rust**, a repository dedicated to implementing machine learning algorithms using the Rust programming language. The aim of this repository is to explore the performance benefits and challenges of building ML models in Rust, focusing on safety, speed, and scalability.

## About the Project

Machine Learning (ML) has predominantly been implemented in Python, thanks to libraries like TensorFlow and PyTorch. However, Rust is emerging as a compelling alternative due to its performance, memory safety, and concurrency features. This repository demonstrates how core ML algorithms can be built in Rust, leveraging libraries like `tch-rs` for deep learning and `ndarray` for general numerical operations.

## Current Projects

### Vision Transformer (ViT)

Branch: [`feature/vision_transformer`](https://github.com/ramendrasingla/ml_algorithms_in_rust/tree/feature/vision_transformer)

- **Description:** Implementation of the Vision Transformer (ViT) using Rust's `tch-rs` library.
- **Features:**
  - End-to-end model definition and training pipeline in Rust.
  - Performance benchmarking against a Python implementation using PyTorch.
  - Codebase designed for modularity and extensibility.
- **Goals:**
  - Evaluate the feasibility of training and deploying large ML models in Rust.
  - Benchmark Rust's performance in training speed, inference time, and resource utilization.

## Getting Started

### Prerequisites

To work with this repository, ensure the following tools are installed:

- Rust (latest stable version) – [Install Rust](https://www.rust-lang.org/tools/install)
- Cargo (Rust package manager)
- Python (for performance comparison)
- `libtorch` library for Rust - [libtorch-cpu](https://download.pytorch.org/libtorch/nightly/cpu/)
- PyTorch for Python benchmarking

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ramendrasingla/ml_algorithms_in_rust.git
   cd ml_algorithms_in_rust
   ```
2.	Navigate to the desired branch for a specific project (e.g., Vision Transformer):

   ```bash
   git checkout feature/vision_transformer
   ```
3. Build the Rust project:
   
   ```bash
   cargo build --release
   ```
5. Set up the Python environment for comparison:
   
   ```bash
   pip install -r requirements.txt
   ```
## Usage

1. **Run the Vision Transformer in Rust:**

   ```bash
   cargo run --bin benchmark
   ```
2. **Benchmark against Python (if applicable):**

  ```bash
  python src/benchmarks/python/benchmark_vit.py
  ```

## Benchmarks

Detailed benchmarking results comparing Rust and Python implementations can be found in the [`benchmarks`](benchmarks) directory. Key metrics include:

- Training time
- Inference speed
- Memory usage

## Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/<feature_name>`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/<feature_name>`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, feel free to reach out:

- **Author:** Ramendra Singla
- **GitHub:** [@ramendrasingla](https://github.com/ramendrasingla)
- **Medium:** [@ramendrasingla](https://medium.com/@ramendrasingla)
- **X:** [@singla_ram99](https://x.com/singla_ram99)
  
