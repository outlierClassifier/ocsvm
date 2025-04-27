pub mod utils;

use crate::signals::*;
use utils::*;

/// A Set is a collection of values that are associated with a label.
#[derive(Debug)]
pub struct Set {
    pub label: String,
    pub values: Vec<f64>,
    xs: Option<Xs>,
}

/// Xs is an special set that contains a single value.
#[derive(Clone, Debug)]
pub struct Xs {
    pub values: Vec<f64>,
}

pub struct SetCollection {
    sets: Vec<Set>,
}

/// A Taxonomy is vector of tuples where the first element is a label and second element is a value.
/// A tuple belongs to a taxonomy if the nearest centroid of the point is the label of the taxonomy.
pub struct Taxonomy {
    label: String,
    pub data: Vec<(String, f64)>,
}

impl From<&Signal> for Set {
    fn from(signal: &Signal) -> Self {
        let label = signal.label.clone();
        Set {
            label,
            values: signal.values.clone(),
            xs: None,
        }
    }
}

impl From<&SignalFeatures> for Set {
    fn from(feature: &SignalFeatures) -> Self {
        let label = format!("{:?}", feature.type_);
        Set {
            label,
            values: feature.values.clone(),
            xs: None,
        }
    }
}

impl From<SignalFeatures> for Set {
    fn from(feature: SignalFeatures) -> Self {
        Self::from(&feature)
    }
}

impl From<f64> for Xs {
    fn from(value: f64) -> Self {
        Xs {
            values: vec![value],
        }
    }
}

impl From<&Signal> for Xs {
    fn from(signal: &Signal) -> Self {
        Xs {
            values: signal.values.clone(),
        }
    }
}

impl From<Signal> for Xs {
    fn from(signal: Signal) -> Self {
        Xs::from(&signal)
    }
}

impl From<&Set> for Xs {
    fn from(set: &Set) -> Self {
        Xs {
            values: set.values.clone(),
        }
    }
}

impl From<Set> for Xs {
    fn from(set: Set) -> Self {
        Xs::from(&set)
    }
}

impl FromIterator<f64> for Xs {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        Xs {
            values: iter.into_iter().collect(),
        }
    }
}

impl Set {
    pub fn new(label: String, values: Vec<f64>) -> Set {
        Set {
            label,
            values,
            xs: None,
        }
    }

    pub fn calculate_centroid(&self) -> f64 {
        match self.xs.clone() {
            Some(x) => {
                (self.values.iter().sum::<f64>() + x.values.iter().sum::<f64>())
                    / (self.values.len() as f64 + x.values.len() as f64)
            }
            None => self.values.iter().sum::<f64>() / (self.values.len() as f64),
        }
    }

    pub fn add_xs(&mut self, xs: Xs) {
        self.xs = Some(xs);
    }

    pub fn remove_xs(&mut self) {
        self.xs = None;
    }
}

impl SetCollection {
    pub fn new() -> SetCollection {
        SetCollection { sets: Vec::new() }
    }

    pub fn add_set(&mut self, set: Set) {
        self.sets.push(set);
    }

    pub fn add_sets(&mut self, sets: Vec<Set>) {
        self.sets.extend(sets);
    }

    fn calculate_centroids(&mut self, xs: Xs, xs_label: String) -> Vec<f64> {
        let mut centroids = Vec::new();

        self.sets.iter_mut().for_each(|set| {
            if set.label == xs_label {
                set.add_xs(xs.clone());
                centroids.push(set.calculate_centroid());
                set.remove_xs();
            } else {
                centroids.push(set.calculate_centroid());
            }
        });

        centroids
    }

    pub fn calculate_centroids_matrix(&mut self, xs: &Xs) -> Matrix {
        let mut centroids_matrix = Vec::new();

        // Get all labels first
        let labels: Vec<String> = self.sets.iter().map(|set| set.label.clone()).collect();

        // Calculate centroids for each label
        for label in labels {
            centroids_matrix.push(self.calculate_centroids(xs.clone(), label));
        }

        Matrix::new(centroids_matrix)
    }

    /// Given a value, returns the label of the set with the closest centroid.
    fn _categorize(&self, x: &f64) -> String {
        let mut closest_set = String::new();
        let mut min_distance = f64::MAX;

        for set in self.sets.iter() {
            let centroid = set.calculate_centroid();
            let distance = (centroid - x).abs();

            if distance < min_distance {
                min_distance = distance;
                closest_set = set.label.clone();
            }
        }

        closest_set
    }

    /// Returns the taxonomies where xs is categorized, for each row in the centroids matrix.
    pub fn taxonomize(&mut self, xs: Xs) -> Vec<Taxonomy> {
        // Por cada fila de la matriz de centroides
        // Necesito una taxonomia por cada Set (columnas de la matriz de centroides, o numero de sets)
        // Cada taxonomia se inicializa con el label del set
        // Un punto pertenece a la taxonomia con el centroide mas cercano dentro de su fila
        let centroids_matrix = self.calculate_centroids_matrix(&xs);
        let mut taxonomies = Vec::new();
        for (i, _rows) in centroids_matrix.get_vec_vec().iter().enumerate() {
            let mut taxonomy = Vec::new();
            // Create taxonomies
            for set in &self.sets {
                let tx = Taxonomy::new(set.label.clone());
                taxonomy.push(tx);
            }

            // Centroids for categorization
            let centroids: Vec<(String, f64)> = centroids_matrix.get_vec_vec()[i]
                .iter()
                .zip(self.sets.iter())
                .map(|(c, s)| (s.label.clone(), *c))
                .collect();
            // println!("{:?}", centroids);
            let categorize = |x: &f64| -> String {
                let mut closest_set = String::new();
                let mut min_distance = f64::MAX;

                for (label, centroid) in &centroids {
                    let distance = (centroid - x).abs();

                    if distance < min_distance {
                        min_distance = distance;
                        closest_set = label.clone();
                    }
                }

                closest_set
            };

            // Categorizations
            let mut categorized_values = Vec::new();

            for (j, set) in &mut self.sets.iter_mut().enumerate() {
                if i == j {
                    let x = Xs {
                        values: xs.values.clone(),
                    };
                    set.add_xs(x);
                } else {
                    set.remove_xs();
                }
                // println!("{:?}", set);
                for &v in &set.values {
                    let category = categorize(&v);
                    categorized_values.push((set.label.clone(), v, category));
                }

                if set.xs.is_some() {
                    let xs_mean = set.xs.as_ref().unwrap().values.iter().sum::<f64>()
                        / set.xs.as_ref().unwrap().values.len() as f64;
                    let category = categorize(&xs_mean);
                    categorized_values.push((set.label.clone(), xs_mean, category.clone()));

                    // Aqui sabemos a que taxonomia pertenece el punto xs
                    //println!("Xs belongs to {:?}", category);

                    // Podemos eliminar las taxonomias que no contienen el punto xs
                    taxonomy.retain(|t| t.label == category);

                    set.remove_xs();
                }
            }

            // Assign data to taxonomies
            for tx in &mut taxonomy {
                for (set_label, value, category) in &categorized_values {
                    if &tx.label == category {
                        tx.add_data((set_label.clone(), *value));
                    }
                }
            }
            taxonomies.push(taxonomy);
        }
        taxonomies.into_iter().flatten().collect()
    }

    /// Returns the probability Matrix of the taxonomies.
    pub fn pobabilize(&self, taxonomies: &Vec<Taxonomy>) -> Matrix {
        let mut prob_taxonomies = Vec::new();
        let size = taxonomies
            .iter()
            .map(|tx| tx.data.len() as f64)
            .collect::<Vec<f64>>();

        let mut unique_labels = taxonomies
            .iter()
            .map(|tx| {
                tx.data
                    .iter()
                    .map(|(l, _)| l.clone())
                    .collect::<Vec<String>>()
            })
            .flatten()
            .collect::<Vec<String>>(); // No se si esto funciona en todos los casos

        unique_labels.sort();
        unique_labels.dedup();

        taxonomies.iter().for_each(|tx| {
            let prob = unique_labels
                .iter()
                .zip(&size)
                .map(|(label, size)| {
                    let count = tx.data.iter().filter(|(l, _)| l == label).count() as f64;
                    count / size
                })
                .collect();

            prob_taxonomies.push(prob);
        });
        Matrix::new(prob_taxonomies)
    }
    pub fn probabilize_avg(&self, taxonomies: &Vec<Taxonomy>) -> Vec<f64> {
        let prob_taxonomies = self.pobabilize(taxonomies);
        let mut prob_avg = Vec::new();
        for i in 0..prob_taxonomies.cols_count() {
            let mut sum = 0.0;
            for j in 0..prob_taxonomies.rows_count() {
                sum += prob_taxonomies.get(j, i);
            }
            prob_avg.push(sum / prob_taxonomies.rows_count() as f64);
        }
        prob_avg
    }
}

impl Taxonomy {
    pub fn new(label: String) -> Taxonomy {
        Taxonomy {
            label,
            data: Vec::new(),
        }
    }

    pub fn add_data(&mut self, data: (String, f64)) {
        self.data.push(data);
    }

    pub fn to_string(&self) -> String {
        let mut taxonomy_str = format!("{}: ", self.label.replace("y", "T"));
        for (label, value) in &self.data {
            taxonomy_str.push_str(&format!("({}, {}) ", value, label));
        }
        taxonomy_str
    }

    pub fn vec_to_string(taxonomies: &Vec<Taxonomy>) -> String {
        taxonomies
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<String>>()
            .join("\n")
    }
}
