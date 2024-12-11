use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use tch::{nn, Tensor, Device};
use algo::VisionTransformer;

pub fn benchmark_vision_transformer() {
    // Select device: CUDA or CPU
    let device = if Device::cuda_if_available().is_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    println!("Using device: {:?}", device);

    // Initialize the Vision Transformer
    let vs = nn::VarStore::new(device);
    let mut model = VisionTransformer::new(
        &vs,
        12,  // Number of Transformer blocks
        768, // Embedding dimension
        12,  // Number of attention heads
        3072, // MLP hidden dimension
        10,  // Number of output classes
        0.1, // Dropout rate
        16,  // Patch Size
        0.001 // Learning rate
    );

    // Random input tensor and target tensor
    let inputs = Tensor::randn(&[256, 3, 224, 224], (tch::Kind::Float, device));
    let targets = Tensor::randn(&[256, 10], (tch::Kind::Float, device)); // Example target tensor

    // File path for logging
    let log_file_path = "./data/benchmarks/vision_transformer_train.csv";

    // Perform benchmarking
    let mut new_data = vec![];
    for epoch in 1..=10 {
        let start = Instant::now();
        let loss = model.train_step(&inputs, &targets);
        let duration = start.elapsed().as_secs_f64();
        new_data.push((epoch, loss, duration));
        println!("Epoch {} - Loss: {:.4}, Duration: {:.2} seconds", epoch, loss, duration);
    }

    // Read existing data if file exists
    let mut existing_data = vec![];
    if Path::new(log_file_path).exists() {
        let file = File::open(log_file_path).expect("Failed to open existing log file");
        let reader = BufReader::new(file);
        for (i, line) in reader.lines().enumerate() {
            let line = line.expect("Failed to read line");
            if i == 0 { continue; } // Skip header
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 3 {
                let epoch = parts[0].parse::<usize>().unwrap_or(0);
                let rust_loss = parts[1].parse::<f64>().unwrap_or(0.0);
                let rust_duration = parts[2].parse::<f64>().unwrap_or(0.0);
                existing_data.push((epoch, rust_loss, rust_duration));
            }
        }
    }

    // Write merged data to file
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true) // Overwrite the file
        .open(log_file_path)
        .expect("Failed to create log file");
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "Epoch,Rust Loss,Rust Duration (seconds)")
        .expect("Failed to write header to log file");

    // Write data
    for (epoch, loss, duration) in new_data {
        writeln!(writer, "{},{:.4},{:.2}", epoch, loss, duration)
            .expect("Failed to write data to log file");
    }

    println!("Benchmark results logged to {}", log_file_path);
}

pub fn main() {
    benchmark_vision_transformer();
}