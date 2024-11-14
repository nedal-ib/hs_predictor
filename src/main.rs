mod neural_net;
mod data_loader;

use neural_net::NeuralNet;
use data_loader::load_data;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let (train_inputs, train_labels, test_inputs, test_labels) = load_data("data/housing.csv")?;

    let input_neurons = 5;
    let hidden_sizes = [10, 8];
    let output_neurons = 1;
    let learning_rate = 0.001;

    let mut nn = NeuralNet::new(input_neurons, &hidden_sizes, output_neurons, learning_rate);

    nn.backpropagate(&train_inputs, &train_labels, 5000);

    let train_predictions = nn.predict(&train_inputs);
    let train_error = &train_labels - &train_predictions;
    let train_mse = train_error.component_mul(&train_error).mean();
    println!("Training Mean Squared Error: {:.4}", train_mse);

    let test_predictions = nn.predict(&test_inputs);
    let test_error = &test_labels - &test_predictions;
    let test_mse = test_error.component_mul(&test_error).mean();
    println!("Test Mean Squared: {:.4}", test_mse);

    nn.save("model.json")?;
    println!("Model saved to model.json");

    let loaded_nn = NeuralNet::load("model.json")?;
    let loaded_predictions = loaded_nn.predict(&test_inputs);
    println!("Loaded model test predictions: {:?}", loaded_predictions);

    Ok(())
}
