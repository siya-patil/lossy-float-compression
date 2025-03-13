# Lossy-Compressed Floating Point Data for High Energy Physics

## Overview
This project implements a lossy compression technique for 32-bit floating-point numbers by zeroing out a specified number of least significant bits (LSBs) in the mantissa. The goal is to reduce storage while maintaining sufficient precision for high-energy physics computations.

## Compression Strategy
A 32-bit IEEE 754 floating-point number consists of:
- **1-bit sign**
- **8-bit exponent**
- **23-bit mantissa**

In this approach:
- The least significant bits (LSBs) of the mantissa are zeroed out based on a predefined threshold.
- In this implementation, I have zeroed out 12 least significant bits of the mantissa. This is a tunable parameter
## Methods
- Bit Masking is applied to zero out the last 12 bits of the mantissa.
- After zeroing out the bits, count the number of trailing zeroes.
- I have eliminated the redundant zeroes that were occupying 12 bits and instead stored the number of trailing zeroes.
- The number of trailing zeroes is stored using 4 bits (allowing a maximum of 16 zeroed bits, represented as `1111`).
- The compressed representation is stored in 3 bytes (24 bits), instead of the original 4 bytes (32 bits).

## Why 3 Bytes for Storage?
Instead of storing the full 32-bit representation, store only:
- **Sign bit (1 bit)**
- **Exponent bits (8 bits)**
- **Remaining nonzero mantissa bits** (since `23 - num_zeroed_bits` remain). Here, I have stored 23-12=11 bits.
- **Number of trailing zeroes (4 bits)**

Since the maximum number of trailing zeroes is 16 (stored as `1111` in 4 bits), and in my current implementation, I am zeroing out 12 bits (stored as `1010` in 4 bits), the total storage required is:

`1 (sign) + 8 (exponent) + 11 (mantissa after zeroing out 12 bits) + 4 (trailing zero count) = 24 bits = 3 bytes`

Therefore, this implementation stores the compressed data in 3 bytes while the original data used 4 bytes. 
I have made this optimization to reduce the memory footprint by 25% while allowing precision recovery.

## Repository Structure
float_compression.py - compression, decompression processes

utils.py - visualization, error metrics, and performance benchmarking

data/ - contains original data and compressed data binary files for 3 distributions (uniform, gaussian, exponential)

## Implementation
- The code processes floating-point numbers and applies the lossy compression.
- The compressed data can later be decompressed with an aim to recover the original values.

## Usage
- The implementation is designed for efficient storage in high-energy physics applications.
- Future improvements can optimize trade-offs between storage size and numerical accuracy.

## Future Work
- Implement adaptive precision control based on application requirements.
- Optimize the compression method for different numerical distributions in high-energy physics datasets.
- Integration with real HEP datasets.
- Explore Hybrid compression techniques

