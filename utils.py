from scipy.stats import skew, kurtosis
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from float_compression import pack_components, save_packed_to_file, extract_components, load_packed_from_file, reconstruct_from_components, unpack_components

def plot_comparison(original, reconstructed, title):
    """
    Plot the original vs. reconstructed data.
    :parameter original: Original data (1D array).
    :parameter reconstructed: Reconstructed data (1D array).
    :parameter title: Title of the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(original, label="Original Data", alpha=0.7)
    plt.plot(reconstructed, label="Reconstructed Data", alpha=0.7)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def plot_error_distribution(original, reconstructed, title):
    """
    Plot the distribution of errors between original and reconstructed data.
    :parametereter original: Original data (1D array).
    :parametereter reconstructed: Reconstructed data (1D array).
    :parametereter title: Title of the plot.
    """
    errors = np.abs(original - reconstructed)
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.show()

def calculate_error_metrics(original, reconstructed):
    """
    Calculate error metrics between original and reconstructed data.
    :parameter original: Original data (1D array).
    :parameter reconstructed: Reconstructed data (1D array).
    :return: Dictionary of error metrics.
    """
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    relative_error = np.mean(np.abs(original - reconstructed) / (np.abs(original) + 1e-10))  # Avoid division by zero
    return {
        "MSE": f"{mse:.10f}",  # Format as a decimal with 10 decimal places
        "MAE": f"{mae:.10f}",  # Format as a decimal with 10 decimal places
        "Relative Error": f"{relative_error:.6f}",  # Format as a decimal with 6 decimal places
    }

def save_original_to_file(data, filename):
    """
    Save the original data to a binary file.
    :parameter data: Input data (1D array).
    :parameter filename: Name of the output file.
    """
    with open(filename, "wb") as f:
        f.write(data.tobytes())

def load_original_from_file(filename, dtype=np.float32):
    """
    Load the original data from a binary file.
    :parameter filename: Name of the input file.
    :parameter dtype: Data type of the original data.
    :return: Loaded data as a numpy array.
    """
    with open(filename, "rb") as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data

def benchmark_compression(data, num_bits_to_zero, filename):
    """
    Benchmark the compression process.
    :parameter data: Input data (1D array).
    :parameter num_bits_to_zero: Number of least significant bits to zero out.
    :parameter filename: Name of the output file.
    :return: Tuple of (compression_time, original_size, compressed_size).
    """
    start_time = time.time()
    original_size = data.nbytes  # Size of original data in bytes
    packed_data = [pack_components(*extract_components(x, num_bits_to_zero)) for x in data]
    save_packed_to_file(packed_data, filename)
    compressed_size = os.path.getsize(filename)
    compression_time = time.time() - start_time
    return compression_time, original_size, compressed_size

def benchmark_decompression(filename, num_samples):
    """
    Benchmark the decompression process.
    :parameter filename: Name of the input file.
    :parameter num_samples: Number of samples in the dataset.
    :return: Tuple of (decompression_time, reconstructed_data).
    """
    start_time = time.time()
    packed_data = load_packed_from_file(filename)
    reconstructed_data = np.array([reconstruct_from_components(*unpack_components(packed)) for packed in packed_data])
    decompression_time = time.time() - start_time
    return decompression_time, reconstructed_data

def compare_statistics(original, reconstructed, title):
    """
    Compare statistical parameters of the original and reconstructed data.
    :parameter original: Original data (1D array).
    :parameter reconstructed: Reconstructed data (1D array).
    :parameter title: Title for the comparison (e.g., distribution name).
    """
    # Calculate statistics for original data
    original_mean = np.mean(original)
    original_variance = np.var(original)
    original_std = np.std(original)
    original_skewness = skew(original)
    original_kurtosis = kurtosis(original)

    # Calculate statistics for reconstructed data
    reconstructed_mean = np.mean(reconstructed)
    reconstructed_variance = np.var(reconstructed)
    reconstructed_std = np.std(reconstructed)
    reconstructed_skewness = skew(reconstructed)
    reconstructed_kurtosis = kurtosis(reconstructed)

    # Print comparison
    print(f"\nStatistical Comparison for {title}:")
    print(f"{'Parameter':<20} {'Original':<15} {'Reconstructed':<15} {'Difference':<15}")
    print(f"{'Mean':<20} {original_mean:<15.6f} {reconstructed_mean:<15.6f} {abs(original_mean - reconstructed_mean):<15.6f}")
    print(f"{'Variance':<20} {original_variance:<15.6f} {reconstructed_variance:<15.6f} {abs(original_variance - reconstructed_variance):<15.6f}")
    print(f"{'Standard Deviation':<20} {original_std:<15.6f} {reconstructed_std:<15.6f} {abs(original_std - reconstructed_std):<15.6f}")
    print(f"{'Skewness':<20} {original_skewness:<15.6f} {reconstructed_skewness:<15.6f} {abs(original_skewness - reconstructed_skewness):<15.6f}")
    print(f"{'Kurtosis':<20} {original_kurtosis:<15.6f} {reconstructed_kurtosis:<15.6f} {abs(original_kurtosis - reconstructed_kurtosis):<15.6f}")