use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::min;
use std::collections::HashSet;

#[derive(Eq, PartialEq, Hash)]
pub struct HashKey<const N: usize>([u32; N]);

#[derive(Copy, Clone)]
pub struct Vector<const N: usize>(pub [f32; N]);
