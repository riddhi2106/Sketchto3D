import numpy as np
import matplotlib.pyplot as plt

from load_data import load_image
from preprocess import preprocess_image
from edge_detect import detect_edges
from coords_extract import extract_coordinates


def main():
    # Step 1: Load image
    image = load_image(index=0)

    # Step 2: Preprocess
    processed = preprocess_image(image)

    # Step 3: Edge detection
    edges = detect_edges(processed)

    # Step 4: Extract coordinates
    coords = extract_coordinates(edges)

    # Step 5: Visualization
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(processed, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Edges")
    plt.imshow(edges, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Coordinates")
    plt.scatter(coords[:, 1], coords[:, 0], s=1)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

    # Step 6: Print info
    print("Total edge points:", len(coords))
    print("Sample coordinates:\n", coords[:10])

    # ✅ Step 7: SAVE OUTPUT (VERY IMPORTANT)
    np.save("../coords.npy", coords)
    print("✅ Coordinates saved to ../coords.npy")


if __name__ == "__main__":
    main()