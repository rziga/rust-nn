use crate::matrix::Matrix;
use std::{fs::*, ops::Index, vec};

pub trait Dataset {
    type Output;
    fn len(&self) -> usize;
    fn getitem(&self)
}

pub struct CIFAR10 {
    images: Vec<Vec<f32>>,
    labels: Vec<Vec<f32>>,
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

impl Index<usize> for dyn Dataset {
    type Output = Self::Output;
    
    fn index(&self, index: usize) -> &Self::Output {
        &(self.images[index], self.labels[index])
    }
}
