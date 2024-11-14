use nalgebra::DMatrix;
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use std::error::Error;

pub fn load_data(filename: &str) -> Result<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(filename)?;
    let mut inputs = Vec::new();
    let mut labels = Vec::new();

    for result in rdr.records() {
        let record = match result {
            Ok(record) => record,
            Err(_) => continue,
        };

        if record.len() == 6 {
            let input: Vec<f64> = record.iter().take(5).map(|x| x.parse().unwrap()).collect();
            let label: Vec<f64> = record.iter().skip(5).map(|x| x.parse().unwrap()).collect();

            inputs.extend(input);
            labels.extend(label);
        } else {
            eprintln!("Skipping malformed row: {:?}", record);
        }
    }

    let inputs = DMatrix::from_row_slice(inputs.len() / 5, 5, &inputs);
    let labels = DMatrix::from_row_slice(labels.len(), 1, &labels);

    let maxs = inputs.column_iter().map(|col| col.max()).collect::<Vec<f64>>();
    let mins = inputs.column_iter().map(|col| col.min()).collect::<Vec<f64>>();
    let normalized_inputs = DMatrix::from_fn(inputs.nrows(), inputs.ncols(), |i, j| {
        (inputs[(i, j)] - mins[j]) / (maxs[j] - mins[j])
    });

    let mut indices: Vec<usize> = (0..normalized_inputs.nrows()).collect();
    indices.shuffle(&mut rand::thread_rng());

    let split_at = (0.8 * indices.len() as f64) as usize;
    let (train_indices, test_indices) = indices.split_at(split_at);

    let train_inputs = DMatrix::from_rows(&train_indices.iter().map(|&i| normalized_inputs.row(i)).collect::<Vec<_>>());
    let test_inputs = DMatrix::from_rows(&test_indices.iter().map(|&i| normalized_inputs.row(i)).collect::<Vec<_>>());
    let train_labels = DMatrix::from_rows(&train_indices.iter().map(|&i| labels.row(i)).collect::<Vec<_>>());
    let test_labels = DMatrix::from_rows(&test_indices.iter().map(|&i| labels.row(i)).collect::<Vec<_>>());

    Ok((train_inputs, train_labels, test_inputs, test_labels))
}
