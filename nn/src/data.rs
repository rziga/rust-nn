use std::{fs::*, vec};
use rand::{seq::SliceRandom, Rng};

use crate::matrix::Matrix;

pub trait Dataset {
    fn len(&self) -> usize;
    fn get_sample(&self, index: usize) -> (&Vec<f32>, &Vec<f32>);
}

pub struct CIFAR10 {
    pub images: Vec<Vec<f32>>,
    pub labels: Vec<Vec<f32>>,
}

impl CIFAR10 {
    pub fn new(files: Vec<&str>) -> Self {
        let mut images = vec![];
        let mut labels = vec![];
        for file in files {
            let (mut x, mut y) = Self::parse_file(file);
            images.append(&mut x);
            labels.append(&mut y);
        }
        Self { images, labels }
    }

    // read bytes and split them into image and label bytes
    fn parse_file(file: &str) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let bytes = read(file).unwrap();
        let mut x = vec![];
        let mut y = vec![];
        for chunk in bytes.chunks(3073) {
            // one-hot encode label
            let label = chunk[0] as usize;
            let mut oh = vec![0.0f32; 10];
            oh[label] = 1.0;
            y.push(oh);
            
            // convert image to f32 in range [0, 1]
            let mut img = chunk[1..]
                .iter().map( |x| (*x as f32) / 255.0 ).collect();
            x.push(img);
        }
        (x, y)
    }
}

impl Dataset for CIFAR10 {
    fn len(&self) -> usize {
        self.images.len()
    }

    fn get_sample(&self, index: usize) -> (&Vec<f32>, &Vec<f32>) {
        (
            &self.images[index],
            &self.labels[index],
        )
    }
}

pub struct DataLoader {
    pub batch_size: usize,
    dataset: Box<dyn Dataset>,
    indexes: Vec<usize>,
    current: usize,
}

impl DataLoader {
    pub fn new(batch_size: usize, dataset: Box<dyn Dataset>) -> Self {
        let l = dataset.len();
        Self { batch_size, dataset, indexes: (0..l).collect(), current: 0 }
    }

    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.indexes.shuffle(rng);
    }
}

impl Iterator for DataLoader {
    type Item = (Matrix, Matrix);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current + self.batch_size >= self.dataset.len() {
            return None
        }

        let mut x_batch = vec![];
        let mut y_batch = vec![];
        for i in self.current..self.current+self.batch_size {
            let (x, y) = self.dataset.get_sample(self.indexes[i]);
            x_batch.append(&mut x.clone());
            y_batch.append(&mut y.clone());
        }
        self.current += self.batch_size;
        
        let x_cols = x_batch.len() / self.batch_size;
        let y_cols = y_batch.len() / self.batch_size;
        Some((
            Matrix::from_vec(self.batch_size, x_cols, x_batch),
            Matrix::from_vec(self.batch_size, y_cols, y_batch)
        ))
    }
}