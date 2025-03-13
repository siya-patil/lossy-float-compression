import numpy as np
import os
from float_compression import extract_components, pack_components, save_packed_to_file, load_packed_from_file, unpack_components, reconstruct_from_components
from utils import plot_comparison, plot_error_distribution, calculate_error_metrics, benchmark_compression, benchmark_decompression, save_original_to_file, load_original_from_file, compare_statistics

def main():
    # Parameters
    num_samples = 100000  # Number of samples per distribution
    num_bits_to_zero = 12 # Number of LSBs to zero out

    uniform_data = np.random.uniform(-1000, 1000, num_samples).astype(np.float32)  
    gaussian_data = np.random.normal(0, 10, num_samples).astype(np.float32)  
    exponential_data = np.random.exponential(scale=10, size=num_samples).astype(np.float32) 

    # Combine into a dictionary for easy access
    data = {
        "gaussian": gaussian_data,
        "exponential": exponential_data,
        "uniform": uniform_data
    }

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Compress, benchmark, and save data for each distribution
    results = {}
    for name, dataset in data.items():
        print(f"\nProcessing {name} distribution...")
        original_filename = f"data/{name}_original.bin"
        compressed_filename = f"data/{name}_compressed.bin"

        # Save original data to binary file
        save_original_to_file(dataset, original_filename)

        # Load original data from binary file
        dataset = load_original_from_file(original_filename)

        # Benchmark compression
        compression_time, original_size, compressed_size = benchmark_compression(dataset, num_bits_to_zero, compressed_filename)
        print(f"Compression time: {compression_time:.4f} seconds")

        # Benchmark decompression
        decompression_time, reconstructed_data = benchmark_decompression(compressed_filename, num_samples)
        print(f"Decompression time: {decompression_time:.4f} seconds")

        # Calculate error metrics
        error_metrics = calculate_error_metrics(dataset, reconstructed_data)
        print("Error Metrics:")
        for metric, value in error_metrics.items():
            print(f"  {metric}: {value}")

        compare_statistics(dataset, reconstructed_data, name)

        # Save results
        results[name] = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_time": compression_time,
            "decompression_time": decompression_time,
            "error_metrics": error_metrics,
            "compressed_filename": compressed_filename,
            "reconstructed_data": reconstructed_data,
        }

    # Print file size comparison
    print("\nFile Size Comparison:")
    for name, result in results.items():
        savings = 100 * (1 - result["compressed_size"] / result["original_size"])
        print(f"{name}: Original = {result['original_size']} bytes, Compressed = {result['compressed_size']} bytes, Savings = {savings:.2f}%")

    # Visualize results
    for name, result in results.items():
        plot_comparison(data[name], result["reconstructed_data"], f"{name.capitalize()} Distribution: Original vs. Reconstructed")
        plot_error_distribution(data[name], result["reconstructed_data"], f"{name.capitalize()} Distribution: Error Distribution")

if __name__ == "__main__":
    main()