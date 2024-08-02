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

// Hyper plane imp

struct HyperPlane<const N: usize> {
    coefficients: Vector<N>,
    constant: f32,
}

impl<const N: usize> HyperPlane<N> {
    pub fn point_is_above(&self, point: &Vector<N>) -> bool {
        self.coefficients.dot_prod(point) + self.constant >= 0.0
    }
}

enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode<N>>),
}
struct LeafNode<const N: usize>(Vec<usize>);
struct InnerNode<const N: usize> {
    hyperplane: HyperPlane<N>,
    left_node: Node<N>,
    right_node: Node<N>,
}
pub struct ANNIndex<const N: usize> {
    trees: Vec<Node<N>>,
    ids: Vec<i32>,
    values: Vec<Vector<N>>,
}

impl<const N: usize> ANNIndex<N> {
    fn build_hyperplane(
        indexes: &Vec<usize>,
        all_vecs: &Vec<Vector<N>>,
    ) -> (HyperPlane<N>, Vec<usize>, Vec<usize>) {
        let sample: Vec<_> = indexes
            .choose_multiple(&mut rand::thread_rng(), 2)
            .collect();
        // cartesian eq for hyperplane n * (x - x_0) = 0
        // n (normal vector) is the coefs x_1 to x_n
        let (a, b) = (*sample[0], *sample[1]);
        let coefficients = all_vecs[a].subtract(&all_vecs[b]);
        let point_on_plane = all_vecs[a].avg(&all_vecs[b]);
        let constant = -coefficients.dot_prod(&point_on_plane);
        let hyperplane = HyperPlane::<N> {
            coefficients,
            constant,
        };
        let (mut above, mut below) = (vec![], vec![]);
        for &id in indexes.iter() {
            if hyperplane.point_is_above(&all_vecs[id]) {
                above.push(id)
            } else {
                below.push(id)
            };
        }
        return (hyperplane, above, below);
    }
}

impl<const N: usize> ANNIndex<N> {
    fn build_a_tree(max_size: i32, indexes: &Vec<usize>, all_vecs: &Vec<Vector<N>>) -> Node<N> {
        if indexes.len() <= (max_size as usize) {
            return Node::Leaf(Box::new(LeafNode::<N>(indexes.clone())));
        }
        let (plane, above, below) = Self::build_hyperplane(indexes, all_vecs);
        let node_above = Self::build_a_tree(max_size, &above, all_vecs);
        let node_below = Self::build_a_tree(max_size, &below, all_vecs);
        return Node::Inner(Box::new(InnerNode::<N> {
            hyperplane: plane,
            left_node: node_below,
            right_node: node_above,
        }));
    }
}

impl<const N: usize> ANNIndex<N> {
    fn deduplicate(
        vectors: &Vec<Vector<N>>,
        ids: &Vec<i32>,
        dedup_vectors: &mut Vec<Vector<N>>,
        ids_of_dedup_vectors: &mut Vec<i32>,
    ) {
        let mut hashes_seen = HashSet::new();
        for i in 1..vectors.len() {
            let hash_key = vectors[i].to_hashkey();
            if !hashes_seen.contains(&hash_key) {
                hashes_seen.insert(hash_key);
                dedup_vectors.push(vectors[i]);
                ids_of_dedup_vectors.push(ids[i]);
            }
        }
    }

    pub fn build_index(
        num_trees: i32,
        max_size: i32,
        vecs: &Vec<Vector<N>>,
        vec_ids: &Vec<i32>,
    ) -> ANNIndex<N> {
        let (mut unique_vecs, mut ids) = (vec![], vec![]);
        Self::deduplicate(vecs, vec_ids, &mut unique_vecs, &mut ids);
        // Trees hold an index into the [unique_vecs] list which is not
        // necessarily its id, if duplicates existed
        let all_indexes: Vec<usize> = (0..unique_vecs.len()).collect();
        let trees: Vec<_> = (0..num_trees)
            .map(|_| Self::build_a_tree(max_size, &all_indexes, &unique_vecs))
            .collect();
        return ANNIndex::<N> {
            trees,
            ids,
            values: unique_vecs,
        };
    }
}

impl<const N: usize> ANNIndex<N> {
    fn tree_result(
        query: Vector<N>,
        n: i32,
        tree: &Node<N>,
        candidates: &mut HashSet<usize>,
    ) -> i32 {
        // take everything in node, if still needed, take from alternate subtree
        match tree {
            Node::Leaf(box_leaf) => {
                let leaf_values = &(box_leaf.0);
                let num_candidates_found = min(n as usize, leaf_values.len());
                for i in 0..num_candidates_found {
                    candidates.insert(leaf_values[i]);
                }
                return num_candidates_found as i32;
            }
            Node::Inner(inner) => {
                let above = (*inner).hyperplane.point_is_above(&query);
                let (main, backup) = match above {
                    true => (&(inner.right_node), &(inner.left_node)),
                    false => (&(inner.left_node), &(inner.right_node)),
                };
                match Self::tree_result(query, n, main, candidates) {
                    k if k < n => k + Self::tree_result(query, n - k, backup, candidates),
                    k => k,
                }
            }
        }
    }
}

impl<const N: usize> ANNIndex<N> {
    pub fn search_approximate(&self, query: Vector<N>, top_k: i32) -> Vec<(i32, f32)> {
        let mut candidates = HashSet::new();
        for tree in self.trees.iter() {
            Self::tree_result(query, top_k, tree, &mut candidates);
        }
        candidates
            .into_iter()
            .map(|idx| (idx, self.values[idx].sq_euc_dis(&query)))
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .take(top_k as usize)
            .map(|(idx, dis)| (self.ids[idx], dis))
            .collect()
    }
}

fn main() {
    const DIM: usize = 3; // Let's use 3D vectors for simplicity
    const NUM_VECTORS: usize = 1000;
    const NUM_TREES: i32 = 3;
    const MAX_SIZE: i32 = 10;

    // Generate random vectors
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vector<DIM>> = (0..NUM_VECTORS)
        .map(|_| {
            let coords: [f32; DIM] = std::array::from_fn(|_| rng.gen());
            Vector(coords)
        })
        .collect();

    // Generate random IDs
    let ids: Vec<i32> = (0..NUM_VECTORS as i32).collect();

    // Build the index
    let index = ANNIndex::<DIM>::build_index(NUM_TREES, MAX_SIZE, &vectors, &ids);

    // Generate a random query vector
    let query_vector = Vector([rng.gen(), rng.gen(), rng.gen()]);

    // Search for nearest neighbors
    let results = index.search_approximate(query_vector, 5);

    println!("Query vector: {:?}", query_vector.0);
    println!("Nearest neighbors:");
    for (id, distance) in results {
        println!("ID: {}, Distance: {}", id, distance);
    }
}
