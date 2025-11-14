import triton
import triton.language as tl
import numpy as np

@triton.jit
def add_vectors(A, B, C, n):
   for i in range(n):
       C[i] = A[i] + B[i]

def main():
    """
    Demonstrates the add_vectors Triton kernel with sample data.
    """
    print("EXECUTION START")

    # Define input data
    n = 1024  # Size of the vectors
    A = np.random.rand(n)
    B = np.random.rand(n)
    C = np.zeros(n)

    # Create Triton tensors
    A_tensor = tritons.Tensor(A)
    B_tensor = tritons.Tensor(B)
    C_tensor = tritons.Tensor(C)
    n_tensor = tritons.Tensor(np.array(n))

    # Execute the Triton kernel
    try:
        add_vectors(A_tensor, B_tensor, C_tensor, n_tensor)
    except Exception as e:
        print(f"Error during kernel execution: {e}")
        return

    # Copy the result back from Triton tensors to a NumPy array
    C_np = C_tensor.to_numpy()

    # Print the result (optional)
    # print("Result:", C_np)

    print("EXECUTION COMPLETE")

if __name__ == "__main__":
    main()