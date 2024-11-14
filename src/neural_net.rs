use nalgebra::{DMatrix, DVector};
use rand::Rng;
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::ser::SerializeStruct;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};

fn serialize_dmatrix<S>(matrix: &DMatrix<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let flattened_data: Vec<f64> = matrix.iter().cloned().collect();
    let mut state = serializer.serialize_struct("DMatrix", 3)?;
    state.serialize_field("data", &flattened_data)?;
    state.serialize_field("nrows", &matrix.nrows())?;
    state.serialize_field("ncols", &matrix.ncols())?;
    state.end()
}

fn deserialize_dmatrix<'de, D>(deserializer: D) -> Result<DMatrix<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct DMatrixData {
        data: Vec<f64>,
        nrows: usize,
        ncols: usize,
    }

    let helper = DMatrixData::deserialize(deserializer)?;
    Ok(DMatrix::from_row_slice(helper.nrows, helper.ncols, &helper.data))
}

fn serialize_dvector<S>(vector: &DVector<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let data: Vec<f64> = vector.iter().cloned().collect();
    data.serialize(serializer)
}

fn deserialize_dvector<'de, D>(deserializer: D) -> Result<DVector<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let data: Vec<f64> = Vec::deserialize(deserializer)?;
    Ok(DVector::from_vec(data))
}

#[derive(Debug, Serialize, Deserialize)]
struct Layer {
    #[serde(serialize_with = "serialize_dmatrix", deserialize_with = "deserialize_dmatrix")]
    weights: DMatrix<f64>,
    #[serde(serialize_with = "serialize_dvector", deserialize_with = "deserialize_dvector")]
    biases: DVector<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNet {
    layers: Vec<Layer>,
    #[serde(serialize_with = "serialize_dmatrix", deserialize_with = "deserialize_dmatrix")]
    output_weights: DMatrix<f64>,
    #[serde(serialize_with = "serialize_dvector", deserialize_with = "deserialize_dvector")]
    output_biases: DVector<f64>,
    learning_rate: f64,
}

impl NeuralNet {
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut layers = Vec::with_capacity(hidden_sizes.len());
        let mut input_dim = input_size;

        for &hidden_size in hidden_sizes {
            let weights = DMatrix::from_fn(input_dim, hidden_size, |_, _| rng.gen::<f64>());
            let biases = DVector::from_fn(hidden_size, |_, _| rng.gen::<f64>());
            layers.push(Layer { weights, biases });
            input_dim = hidden_size;
        }

        let output_weights = DMatrix::from_fn(input_dim, output_size, |_, _| rng.gen::<f64>());
        let output_biases = DVector::from_fn(output_size, |_, _| rng.gen::<f64>());

        NeuralNet {
            layers,
            output_weights,
            output_biases,
            learning_rate,
        }
    }

    pub fn feedforward(&self, x: &DMatrix<f64>) -> (Vec<DMatrix<f64>>, DMatrix<f64>) {
        let mut activations = Vec::with_capacity(self.layers.len());
        let mut input = x.clone();

        for layer in &self.layers {
            let bias_matrix = DMatrix::from_column_slice(layer.biases.len(), 1, layer.biases.as_slice()).transpose();
            let bias_matrix_tiled = bias_matrix.insert_rows(0, input.nrows() - 1, 0.0);
            let z = input * &layer.weights + bias_matrix_tiled;
            let activation = z.map(Self::relu);
            activations.push(activation.clone());
            input = activation;
        }

        let output_bias_matrix = DMatrix::from_column_slice(self.output_biases.len(), 1, self.output_biases.as_slice()).transpose();
        let output_bias_matrix_tiled = output_bias_matrix.insert_rows(0, input.nrows() - 1, 0.0);
        let output_layer_input = input * &self.output_weights + output_bias_matrix_tiled;
        let output = output_layer_input;

        (activations, output)
    }

    pub fn backpropagate(&mut self, x: &DMatrix<f64>, y: &DMatrix<f64>, epochs: usize) {
        for _ in 0..epochs {
            let (activations, output) = self.feedforward(x);
            let mut output_delta = y - &output;
            let last_activation = activations.last().unwrap().clone();
            self.output_weights += last_activation.transpose() * &output_delta * self.learning_rate;
            self.output_biases += output_delta.row_sum().transpose() * self.learning_rate;
            let activation_clones: Vec<DMatrix<f64>> = activations.into_iter().collect();
            let num_layers = self.layers.len();

            for (i, layer) in self.layers.iter_mut().rev().enumerate() {
                let prev_activation = if i + 1 == num_layers {
                    x.clone()
                } else {
                    activation_clones[activation_clones.len() - 2 - i].clone()
                };

                let activation_prime = activation_clones[activation_clones.len() - 1 - i].map(Self::relu_prime);
                let hidden_error = output_delta * self.output_weights.transpose();
                let hidden_delta = hidden_error.component_mul(&activation_prime);
                if prev_activation.ncols() != hidden_delta.nrows() {
                    eprintln!("Mismatch detected: prev_activation ({:?}), hidden_delta ({:?})", prev_activation.shape(), hidden_delta.shape());
                    return;
                }

                layer.weights += prev_activation.transpose() * &hidden_delta * self.learning_rate;
                layer.biases += hidden_delta.row_sum().transpose() * self.learning_rate;
                output_delta = hidden_delta;
            }
        }
    }



    pub fn predict(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let (_, output) = self.feedforward(x);
        output
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let nn = serde_json::from_reader(reader)?;
        Ok(nn)
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    fn relu_prime(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}
