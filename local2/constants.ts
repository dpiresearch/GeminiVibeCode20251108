
import { Language } from './types';

export const LANGUAGES: Language[] = [Language.CUDA, Language.TRITON, Language.MOJO];

export const DEFAULT_CUDA_CODE = `// CUDA kernel to add two vectors
__global__ void addVectors(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
`;
