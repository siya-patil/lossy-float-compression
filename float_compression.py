import numpy as np

def extract_components(x, num_bits):
    """
    Extract the sign bit, exponent bits, non-zero mantissa bits, and trailing zeroes.
    :parameter x: Input floating-point number or array.
    :parameter num_bits: Number of least significant bits to zero out.
    :return: Tuple of (sign_bit, exponent_bits, non_zero_mantissa, trailing_zeroes).
    """
    x = np.asarray(x, dtype=np.float32)  # Use 32-bit floats
    x_int = x.view(np.int32)  # Reinterpret float bits as integers

    # Extract sign bit (bit 31)
    sign_bit = (x_int >> 31) & 0x1

    # Extract exponent bits (bits 23-30)
    exponent_bits = (x_int >> 23) & 0xFF

    # Extract mantissa bits (bits 0-22)
    mantissa = x_int & 0x007FFFFF

    # Handle special cases (zero, infinity, NaN)
    is_zero = mantissa == 0 and exponent_bits == 0
    is_inf_or_nan = exponent_bits == 0xFF

    if is_zero or is_inf_or_nan:
        return sign_bit, exponent_bits, 0, 0  # No trailing zeroes for special cases

    # Zero out the least significant `num_bits` of the mantissa
    mask = ~((1 << num_bits) - 1) # Mask with `num_bits` LSBs set to 0
    mantissa_zeroed = mantissa & mask

    # Count trailing zeroes in the mantissa
    trailing_zeroes = 0
    if mantissa_zeroed != 0:
        while (mantissa_zeroed & 1) == 0:
            trailing_zeroes += 1
            mantissa_zeroed >>= 1

    # Extract non-zero mantissa bits (limited to 11 bits)
    non_zero_mantissa = mantissa_zeroed & 0x7FF  # 11 bits

    return sign_bit, exponent_bits, non_zero_mantissa, trailing_zeroes 

def pack_components(sign_bit, exponent_bits, non_zero_mantissa, trailing_zeroes):
    """
    Pack the sign bit, exponent bits, non-zero mantissa bits, and trailing zeroes into 3 bytes.
    :parameter sign_bit: Sign bit (1 bit).
    :parameter exponent_bits: Exponent bits (8 bits).
    :parameter non_zero_mantissa: Non-zero mantissa bits (11 bits).
    :parameter trailing_zeroes: Number of trailing zeroes (4 bits).
    :return: Packed 3-byte array.
    """
    # Pack sign bit (1 bit), exponent bits (8 bits), non-zero mantissa bits (11 bits), and trailing zeroes (4 bits)
    packed = (
        (int(sign_bit) << 23) |  # Sign bit (bit 23)
        (int(exponent_bits) << 15) |  # Exponent bits (bits 15-22)
        (int(non_zero_mantissa) << 4) |  # Non-zero mantissa bits (bits 4-14)
        (int(trailing_zeroes) & 0xF) # Trailing zeroes (bits 0-3)
      )  
    return packed.to_bytes(3, byteorder='big')  # Convert to 3 bytes

def save_packed_to_file(packed_data, filename):
    """
    Save the packed components to a binary file.
    :parameter packed_data: List of packed 3-byte arrays.
    :parameter filename: Name of the output file.
    """
    with open(filename, "wb") as f:
        for packed in packed_data:
            f.write(packed)  # Save as 3 bytes

def load_packed_from_file(filename):
    """
    Load the packed components from a binary file.
    :parameter filename: Name of the input file.
    :return: List of packed 3-byte integers.
    """
    packed_data = []
    with open(filename, "rb") as f:
        while True:
            # Read 3 bytes for each packed number
            packed_bytes = f.read(3)
            if not packed_bytes:
                break
            packed = int.from_bytes(packed_bytes, byteorder='big')  # Convert to integer
            packed_data.append(packed)
    return packed_data

def unpack_components(packed):
    """
    Unpack the sign bit, exponent bits, non-zero mantissa bits, and trailing zeroes from a 3-byte integer.
    :parameter packed: Packed 3-byte integer.
    :return: Tuple of (sign_bit, exponent_bits, non_zero_mantissa, trailing_zeroes).
    """
    # Unpack sign bit (bit 23)
    sign_bit = (packed >> 23) & 0x1

    # Unpack exponent bits (bits 15-22)
    exponent_bits = (packed >> 15) & 0xFF

    # Unpack non-zero mantissa bits (bits 4-14)
    non_zero_mantissa = (packed >> 4) & 0x7FF  # 11 bits

    # Unpack trailing zeroes (bits 0-3)
    trailing_zeroes = packed & 0xF  # 4 bits

    return sign_bit, exponent_bits, non_zero_mantissa, trailing_zeroes

def reconstruct_from_components(sign_bit, exponent_bits, non_zero_mantissa, trailing_zeroes):
    """
    Reconstruct the original floating-point number from the components.
    :parameter sign_bit: Sign bit (1 bit).
    :parameter exponent_bits: Exponent bits (8 bits).
    :parameter non_zero_mantissa: Non-zero mantissa bits (11 bits).
    :parameter trailing_zeroes: Number of trailing zeroes (4 bits).
    :return: Reconstructed floating-point number.
    """
    # Shift the non-zero mantissa to restore trailing zeroes
    mantissa = non_zero_mantissa << trailing_zeroes

    # Combine sign bit, exponent bits, and mantissa
    x_int = (sign_bit << 31) | (exponent_bits << 23) | mantissa

    # Convert to a NumPy integer and then to a floating-point number
    return np.array([x_int], dtype=np.uint32).view(np.float32)[0]