from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def import_and_convert(path):
    # Open the image and convert it to grayscale
    img = Image.open(path).convert("L")
    return np.array(img)

def reconstruct(U, S, Vt, k):
    reconstructed = np.zeros(shape=(int(U.shape[0]), int(Vt.shape[1])))
    for i in range(k):
        u_i = U[:, i]
        v_i = Vt[i, :]
        reconstructed += S[i] * np.outer(u_i, v_i)


    return reconstructed

# Convert image to matrix
img_arr = import_and_convert("input.jpg")

# Compute the SVD
U,S,Vt = np.linalg.svd(img_arr)
print(f"SVD dimensions: dim(U)={np.shape(U)}, dim(S)={np.shape(S)}, dim(V^T)={np.shape(Vt)} ")

k = 50

print(reconstruct(U, S, Vt, k))

plt.imshow(reconstruct(U, S, Vt, k), cmap='gray')
plt.title(f"Image approximation of rank {k}")
plt.show()