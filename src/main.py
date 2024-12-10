import os
import matplotlib.pyplot as plt
import math
import time
import random

n = 20343797
num_bits = 4096
random.seed(42)

def basic_l2r_square_and_multiply(a, e, n, w_s = 0, w_m = 0):
    s = len(e) # num of bits
    b = 1
    weights = []
    for i in range(0, s):
        b = (b * b) % n # square
        # k is already a left to right representation
        weights.append(w_s)
        if e[i] == 1: # conditional multiply
            b = (b * a) % n
            weights.append(w_m)
    return weights

def dummy_l2r_square_and_multiply(a, e, n, w_s = 0, w_m = 0):
    s = len(e) # num of bits
    b = 1
    weights = []
    for i in range(0, s):
        b = (b * b) % n # square
        # k is already a left to right representation
        weights.append(w_s)
        if e[i] == 1: # conditional multiply
            b = (b * a) % n
        else:
            _ = (b * 1) % n
        weights.append(w_m)
    return weights

def generate_number_with_hamming_weight(bit_length, hamming_weight):
    """
    Generate a single number with a specific Hamming weight.
    """
    if hamming_weight > bit_length or hamming_weight < 0:
        raise ValueError("Hamming weight must be between 0 and bit length")
    
    # Create a bit array with `hamming_weight` ones followed by zeros
    bits = [1] * hamming_weight + [0] * (bit_length - hamming_weight)
    
    # Convert the bit array to an integer
    return int("".join(map(str, bits)), 2)

def part1():
    powers_of_2 = [2**i for i in range(1, num_bits + 1)]
    num_bits = [math.log2(power_of_2) for power_of_2 in powers_of_2]
    times = []
    t = 0
    avg = 20
    for e in powers_of_2:
        e = [int(b) for b in bin(e)[2:]]
        for _ in range(0, avg):
            start_time = time.time()
            basic_l2r_square_and_multiply(2, e, n)
            total_time = time.time() - start_time
            t = t + total_time
        t = t / avg
        times.append(t)
    plt.plot(num_bits, times)
    plt.xlabel("Num Bits, S")
    plt.ylabel("Execution Time")
    plt.title("Execution time vs S")
    plt.xlim(1, num_bits[-1])
    plt.ylim(0, max(times))
    plt.savefig("part1.png")

def part2(part = 0):
    times = []
    hamming_weights = list(range(0, num_bits + 1))
    avg = 10
    for hamming_weight in hamming_weights:
        # 4097 options
        exponent = generate_number_with_hamming_weight(num_bits, hamming_weight)
        t = 0
        exponent = [int(b) for b in bin(exponent)[2:]]
        for _ in range(0, avg):
            if part == 0:
                start_time = time.time()
                basic_l2r_square_and_multiply(2, exponent, n)
                total_time = time.time() - start_time
            elif part == 4:
                start_time = time.time()
                dummy_l2r_square_and_multiply(2, exponent, n)
                total_time = time.time() - start_time
            t = t + total_time
        t = t / avg
        times.append(t)
    plt.plot(hamming_weights, times)
    plt.xlabel("Hamming Weight")
    plt.ylabel("Execution Time")
    plt.title("Hamming Weight vs S")
    plt.xlim(hamming_weights[0], hamming_weights[-1])
    plt.ylim(0, max(times))
    plt.savefig(f"hamming_weight_vs_s_part_{part}.png")

def part3(part = 0):
    hamming_weights = list(range(0, num_bits + 1))
    weights = [] # we can for example do a sum of the weights of every operation
    for hamming_weight in hamming_weights:
        # 4097 options
        exponent = generate_number_with_hamming_weight(num_bits, hamming_weight)
        exponent = [int(b) for b in bin(exponent)[2:]]
        if part == 0:
            weight = basic_l2r_square_and_multiply(2, exponent, n, w_s=1, w_m=2)
        elif part == 4:
            weight = dummy_l2r_square_and_multiply(2, exponent, n, w_s=1, w_m=2)
        weights.append(sum(weight))
    plt.plot(hamming_weights, weights)
    plt.xlabel("Hamming Weight")
    plt.ylabel("Sum of weights for 2 ^ e with x bits")
    plt.title("Sum of weights for 2 ^ e with x bits")
    plt.xlim(hamming_weights[0], hamming_weights[-1])
    plt.ylim(0, num_bits * 4)
    plt.savefig(f"sum_weights_part_{part}.png")



def part4():
    part2(4)
    part3(4)

def generate_random_binary_list(size=num_bits):
    return [random.choice([0, 1]) for _ in range(size + 1)]


def inject_dummy_l2r_square_and_multiply(a, e, n, inject_fault_at_i = -1):
    s = len(e) # num of bits
    b = 1
    for i in range(0, s):
        b = (b * b) % n # square
        # k is already a left to right representation
        if e[i] == 1: # conditional multiply
            if inject_fault_at_i == i:
                b = (b * (a - 1)) % n
            else:
                b = (b * a) % n
        else:
            _ = (b * 1) % n
    return b


def part5():
    e = generate_random_binary_list()
    orig_value = inject_dummy_l2r_square_and_multiply(2, e, n)
    key_bits = []
    for i in range(0, num_bits + 1):
        observed_val = inject_dummy_l2r_square_and_multiply(2, e, n, inject_fault_at_i=i)
        if observed_val == orig_value:
            key_bits.append(0)
        else:
            key_bits.append(1)
    observed_key = int(''.join(map(str, key_bits)), 2)
    original_key = int(''.join(map(str, e)), 2)
    assert observed_key == original_key, "OBSERVED KEY IS NOT EQUAL TO ORIGINAL KEY"

if __name__ == "__main__":
    #part1()
    #part2()
    #part3()
    #part4()
    part5()