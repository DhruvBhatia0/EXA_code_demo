use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::min;
use std::collections::HashSet;

#[derive(Eq, PartialEq, Hash)]
pub struct HashKey<const N: usize>([u32; N]);

#[derive(Copy, Clone)]
pub struct Vector<const N: usize>(pub [f32; N]);

impl<const N: usize> Vector<N> {
    pub fn dot_prod(&self, vector: &Vector<N>) -> f32 {
        return self.0.iter().zip(vector.0).map(|(a, b)| a * b).sum::<f32>();
    }

    pub fn subtract(&self, vector: &Vector<N>) -> Vector<N> {
        let result: [f32; N] = self
            .0
            .iter()
            .zip(vector.0)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        return Vector(result);
    }

    pub fn avg(&self, vector: &Vector<N>) -> Vector<N> {
        let coords: [f32; N] = self
            .0
            .iter()
            .zip(vector.0)
            .map(|(a, b)| (a + b) / 2.0)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        return Vector(coords);
    }

    pub fn to_hashkey(&self) -> HashKey<N> {
        let data: [u32; N] = self
            .0
            .iter()
            .map(|a| a.to_bits())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        return HashKey::<N>(data);
    }
    pub fn sq_euc_dis(&self, vector: &Vector<N>) -> f32 {
        return self
            .0
            .iter()
            .zip(vector.0)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
    }
}

fn main() {
    // Test vectors
    let v1 = Vector([1.0, 2.0, 3.0]);
    let v2 = Vector([4.0, 5.0, 6.0]);

    // Test dot product
    let dot_result = v1.dot_prod(&v2);
    println!("Dot product of {:?} and {:?} is {}", v1.0, v2.0, dot_result);

    // Test subtraction
    let sub_result = v2.subtract(&v1);
    println!(
        "Subtraction result of {:?} - {:?} is {:?}",
        v2.0, v1.0, sub_result.0
    );

    // avg test
    let avg_result = v1.avg(&v2);
    println!("Average of {:?} and {:?} is {:?}", v1.0, v2.0, avg_result.0);

    // Verify results
    assert_eq!(dot_result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    assert_eq!(sub_result.0, [3.0, 3.0, 3.0]); // [4-1, 5-2, 6-3] = [3, 3, 3]

    println!("All tests passed!");
}
