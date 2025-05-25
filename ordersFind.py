import os
import cv2
import numpy as np
from PIL import Image
import imagehash

# ========================
#  HASHING & CLUSTERING
# ========================
class UnionFind:
    """Union-Find data structure for custom clustering."""

    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx != fy:
            self.parent[fy] = fx


def compute_hashes(image_folder):
    """Compute dHash for all images after enhancing them."""
    hashes, filenames = [], []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(image_folder, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue

            enhanced_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(enhanced_rgb)
            h = imagehash.dhash(pil_img)
            hashes.append(h)
            filenames.append(filename)
    return filenames, hashes


def cluster_hashes(hashes, threshold=5):
    """Cluster hashes using Union-Find and Hamming distance threshold."""
    n = len(hashes)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= threshold:
                uf.union(i, j)
    clusters = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)
    return clusters


# ========================
#  FRAME ORDERING (SSIM)
# ========================
def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]


def order_frames(filenames, image_folder):
    """Order frames using SSIM-based greedy algorithm."""
    images = []
    for filename in filenames:
        img = cv2.imread(os.path.join(image_folder, filename))
        if img is not None:
            images.append((filename, cv2.resize(img, (256, 256))))
    if not images:
        return []

    ordered = [images[0]]
    remaining = images[1:]

    while remaining:
        best_match = None
        best_score = -1
        last_img = ordered[-1][1]

        for candidate in remaining:
            score = compute_ssim(last_img, candidate[1])
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_match:
            ordered.append(best_match)
            remaining.remove(best_match)
        else:
            break

    return [item[0] for item in ordered]


# ========================
#  MAIN FUNCTION
# ========================
def main(image_folder):
    filenames, hashes = compute_hashes(image_folder)
    if not filenames:
        print("No valid images found.")
        return

    clusters = cluster_hashes(hashes)
    grouped_files = [[filenames[idx] for idx in cluster] for cluster in clusters.values()]

    # Order each cluster
    ordered_clusters = [order_frames(cluster, image_folder) for cluster in grouped_files]

    # Print results
    for i, cluster in enumerate(ordered_clusters):
        print(f"Video Sequence {i + 1}:")
        print("\n".join(cluster))
        print("-" * 40)


if __name__ == "__main__":
    image_folder = "/home/ubuntu-user/Downloads/ITI-GERD-main/Images"
    main(image_folder)