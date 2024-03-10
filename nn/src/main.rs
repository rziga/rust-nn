mod matrix;
mod layer;
mod parameter;
mod loss;
mod optimizer;
mod lr_scheduler;
mod data;
mod metric;

use crate::{data::{BatchIter, Dataset}, layer::{Layer, Linear, ReLU, Sequential}, lr_scheduler::{ExponentialDecay, Scheduler}, matrix::Matrix, metric::accuracy};
use data::CIFAR10;
use loss::{Crossentropy, Loss};
use optimizer::{Adam, Optimizer, SGD};
use rand::{distributions::weighted, random, rngs::StdRng, SeedableRng};
use std::{fs, iter::Iterator, vec};

fn main() {
    
    let mut rng = StdRng::seed_from_u64(1337);
    let train_dataset = CIFAR10::new(vec![
        "../data/cifar-10-batches-bin/data_batch_1.bin",
        "../data/cifar-10-batches-bin/data_batch_2.bin",
        "../data/cifar-10-batches-bin/data_batch_3.bin",
        "../data/cifar-10-batches-bin/data_batch_4.bin",
    ]);
    let val_dataset = CIFAR10::new(vec![
        "../data/cifar-10-batches-bin/data_batch_5.bin",
    ]);
    let test_dataset = CIFAR10::new(vec![
        "../data/cifar-10-batches-bin/test_batch.bin",
    ]);
    
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(3072, 128, true, &mut rng)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(128, 128, true, &mut rng)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(128, 10, true, &mut rng)),
    ]);
    let mut loss_fn = Crossentropy::new();
    //let mut optim = SGD::new(0.1, 0.0);
    let mut optim = Adam::new(model.parameters(), 0.001, 0.9, 0.999, 1e-8, 0.0);
    let mut sch = ExponentialDecay::new(0.1, &optim);

    println!("epoch,loss,train acc,val acc");
    for epoch in 0..15 {
        // train epoch
        let mut train_accs = vec![];
        let mut losses = vec![];
        for (x, y) in train_dataset.batch_iter(128) {
            let logits = model.forward(&x);
            let loss = loss_fn.forward(&logits, &y);
            //println!("loss: {:.3}", loss);
            losses.push(loss);
            train_accs.push(accuracy(&logits, &y));
    
            optim.zero_grad(model.parameters());
            model.backward(&loss_fn.backward(1.0));
            optim.step(model.parameters());
        }
        let train_acc = (train_accs.iter().sum::<f32>()) / (train_accs.len() as f32);
        let train_loss = (losses.iter().sum::<f32>()) / (losses.len() as f32);

        // val epoch
        let mut val_accs = vec![];
        for (x, y) in val_dataset.batch_iter(128) {
            let logits = model.forward(&x);
            val_accs.push(accuracy(&logits, &y));
        }
        let val_acc = (val_accs.iter().sum::<f32>()) / (val_accs.len() as f32);
        
        println!("{},{},{},{}", epoch, train_loss, train_acc, val_acc);
        sch.step(&mut optim);
    }

    // test epoch at the end
    let mut test_accs = vec![];
    for (x, y) in test_dataset.batch_iter(128) {
        let logits = model.forward(&x);
        test_accs.push(accuracy(&logits, &y));
    }
    let test_acc = (test_accs.iter().sum::<f32>()) / (test_accs.len() as f32);
    println!("final test acc: {}", test_acc);
}
