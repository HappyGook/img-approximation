from PIL import Image
import numpy as np

def reconstruct(U, S, Vt, k):
    reconstructed = np.zeros(shape=(int(U.shape[0]), int(Vt.shape[1])))
    for i in range(k):
        u_i = U[:, i]
        v_i = Vt[i, :]
        reconstructed += S[i] * np.outer(u_i, v_i)


    return reconstructed, reconstructed.shape

# Open the image and convert it to grayscale
img = Image.open("input.jpg").convert("L")

# Convert image to matrix
img_arr = np.array(img)

# Compute the SVD
U,S,Vt = np.linalg.svd(img_arr)
print(f"SVD dimensions: dim(U)={np.shape(U)}, dim(S)={np.shape(S)}, dim(V^T)={np.shape(Vt)} ")

k = 1

print(reconstruct(U, S, Vt, k))