import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Example vectors
king = np.array([1, 1, 0, 0])
comparison_vectors = {
    'Queen': np.array([1, 1, 1, 0]),
    'Queen_2': np.array([2, 2, 0.1, 0]),
    'Queen_3': np.array([3, 3, 0.2, 0]),
    'apple': np.array([0, 0, 10, 10])
}

def vector_metrics (vector_a: np.ndarray, vector_b: np.ndarray) -> tuple[float, float, float, float]:

    if not isinstance(vector_a, np.ndarray) or not isinstance(vector_b, np.ndarray):
        raise ValueError("vector_a and vector_b must be NumPy arrays.")
    if vector_a.shape != vector_b.shape:
        raise ValueError("vector_a and vector_b must have the same shape.")

    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    cosine = dot_product / (norm_a * norm_b) if (norm_a * norm_b) != 0 else 0
    Euclidean = np.linalg.norm(vector_a - vector_b)
    manhattan = np.linalg.norm(vector_a - vector_b, ord=1)
    
    return dot_product, cosine, Euclidean, manhattan

def compare_vectors(reference_vector: np.ndarray, comparison_vectors_dict: dict[str, np.ndarray]) -> pd.DataFrame:
    if not isinstance(reference_vector, np.ndarray):
        raise ValueError("reference_vector must be a NumPy array.")
    if not isinstance(comparison_vectors_dict, dict):
        raise ValueError("comparison_vectors_dict must be a dictionary.")

    results = []

    for name, vector in comparison_vectors_dict.items():
        if not isinstance(vector, np.ndarray):
            raise ValueError(f"The value for '{name}' in comparison_vectors_dict must be a NumPy array.")

        dot_product, cosine, Euclidean, manhattan = \
            vector_metrics(reference_vector, vector)

        results.append({
            'Comparison word': name,
            'Dot product': dot_product,
            'Cosine similarity': cosine,
            'Euclidean distance': Euclidean,
            'Manhattan distance': manhattan
        })

    df = pd.DataFrame (results)
    df = df.set_index("Comparison word")
    
    return df

if __name__ == "__main__":
    df_results = compare_vectors(king, comparison_vectors)
    print(df_results)