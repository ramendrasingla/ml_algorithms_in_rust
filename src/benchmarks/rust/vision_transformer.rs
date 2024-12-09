use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use tch::{nn, Tensor};
use tch::Device;
use algo::VisionTransformer;
// ::ViT::module::VisionTransformer;


pub fn benchmark_vision_transformer() {
    // Select device: MPS (Metal), CUDA, or CPU
    let device = if Device::cuda_if_available().is_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    println!("Using device: {:?}", device);

    // Initialize the Vision Transformer
    let vs = nn::VarStore::new(device);
    let model = VisionTransformer::new(
        &vs.root(),
        12,  // Number of Transformer blocks
        768, // Embedding dimension
        12,  // Number of attention heads
        3072, // MLP hidden dimension
        10,  // Number of output classes
        0.1, // Dropout rate
        16, // Patch Size
    );

    // Random input tensor
    let input = Tensor::randn(&[256, 3, 224, 224], (tch::Kind::Float, device));

    // File path for logging
    let log_file_path = "./data/benchmarks/vision_transformer.csv";

    // Perform benchmarking
    let mut new_durations = vec![];
    for epoch in 1..=10 {
        let start = Instant::now();
        let _output = model.forward(&input); // Forward pass
        let duration = start.elapsed().as_secs_f64();
        new_durations.push((epoch, duration));
        println!("Epoch {} Duration: {:.2} seconds", epoch, duration);
    }

    // Read existing data if file exists
    let mut existing_data = vec![];
    if Path::new(log_file_path).exists() {
        let file = File::open(log_file_path).expect("Failed to open existing log file");
        let reader = BufReader::new(file);
        for (i, line) in reader.lines().enumerate() {
            let line = line.expect("Failed to read line");
            if i == 0 {
                continue; // Skip header
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let epoch = parts[0].parse::<usize>().unwrap_or(0);
                let python_duration = parts[1].parse::<f64>().unwrap_or(0.0);
                existing_data.push((epoch, python_duration));
            }
        }
    }

    // Perform left join
    let mut merged_data = vec![];
    for (epoch, rust_duration) in new_durations {
        if let Some(&(existing_epoch, python_duration)) = existing_data.iter().find(|&&(e, _)| e == epoch) {
            merged_data.push((existing_epoch, python_duration, rust_duration));
        } else {
            merged_data.push((epoch, 0.0, rust_duration)); // 0.0 if Python Duration is missing
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
    writeln!(writer, "Epoch,Python Duration (seconds),Rust Duration (seconds)")
        .expect("Failed to write header to log file");

    // Write data
    for (epoch, python_duration, rust_duration) in merged_data {
        writeln!(writer, "{},{:.2},{:.2}", epoch, python_duration, rust_duration)
            .expect("Failed to write data to log file");
    }

    println!("Benchmark results logged to {}", log_file_path);
}

pub fn main() {
    benchmark_vision_transformer();
}