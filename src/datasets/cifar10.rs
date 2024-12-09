// use ndarray::{Array4, Array1};
// use ndarray_npy::NpzReader;
// use std::fs::File;

// pub fn load_cifar10(path: &str) -> (Array4<f32>, Array1<u8>) {
//     let file = File::open(path).expect("Could not open file.");
//     let mut npz = NpzReader::new(file).expect("Could not read .npz file.");

//     let images: Array4<f32> = npz.by_name("images").expect("Could not load images.");
//     let labels: Array1<u8> = npz.by_name("labels").expect("Could not load labels.");

//     (images, labels)
// }