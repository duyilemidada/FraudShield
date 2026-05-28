# redteam/detect_label_flip.py
# Label consistency check using k-nearest neighbours.
# Add this check to train_fraud_model.py before model training.
 
import numpy as np
from sklearn.neighbors import NearestNeighbors
 
def label_consistency_check(X_prep, y, k=10, threshold=0.80):
    """
    Flag training samples whose label disagrees with the majority of
    their k nearest neighbours. Potential label-flip indicators.

    maybe consider how to choose the best value for k based on your dataset size
    """
    y_np = np.array(y)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
    #nn = NearestNeighbors(...): Configures the spatial mapping tool.n_neighbors=k + 1: Looks for 11 neighbors instead of 10. Why? Because a data point's closest neighbor is always itself. We need 1 extra neighbor to discard the self-match.n_jobs=-1: Tells the computer to use all its processor cores to do the math as fast as possible.
    nn.fit(X_prep)
    _, indices = nn.kneighbors(X_prep)
    #_, indices = ...: For every single transaction, this finds its 11 closest neighbors. It saves the ID (row numbers) of those neighbors into a variable called indices. (The _ just means we are ignoring the exact physical distances, as we only care about who the neighbors are).
 
    suspect_indices = []
    for i, nbrs in enumerate(indices):
        #for i, nbrs in...: Starts a loop to investigate every transaction one by one. i is the current transaction's row number, and nbrs is the list of its 11 neighbors.
        # Exclude self (first neighbour is always the point itself)
        neighbour_labels = y_np[nbrs[1:]]
        #nbrs[1:]: Skips the very first neighbor (which is just the transaction itself) and grabs the other 10.y_np[...]: Looks up the actual fraud labels (0 or 1) for those 10 neighbors.
        majority_label = int(np.mean(neighbour_labels) >= 0.5)
        if majority_label != y_np[i]:
            # Check how overwhelming the majority is
            agreement = np.mean(neighbour_labels == majority_label)
            if agreement >= threshold:
                suspect_indices.append(i)
 
    print(f"[label_consistency] Flagged {len(suspect_indices)} / {len(y_np)} samples")
    return np.array(suspect_indices)
 
# Usage in train_fraud_model.py:
# suspect = label_consistency_check(X_train_prep, y_train)
# X_clean = np.delete(X_train_prep, suspect, axis=0)
# y_clean = np.delete(np.array(y_train), suspect)
