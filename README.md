# DNN From Scratch

This project implements a simple dense neural network from scratch using C. The goal is to understand the fundamental concepts behind dense neural networks and how they work.

The best-achieved accuracy with this implementation on the MNIST dataset is 87.4%.

## Requirements

- GCC (GNU Compiler Collection)

## Usage

1. Compile the training script:
    ```bash
    gcc -O2 -std=c11 train.c -o train
    ```
2. Run the training script:
    ```bash
    ./train
    ```
3. Compile the inference script:
    ```bash
    gcc -O2 -std=c11 inference.c -o inference
    ```
4. Run the inference script:
    ```bash
    ./inference
    ```

## Project Structure

- `train.c`: The script to train the neural network.
- `inference.c`: The script to run inference with the trained neural network.
- `last.txt`: The weights of the last-trained neural network.
- `best.txt`: The weights of the best-trained neural network.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
