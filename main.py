from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def import_and_convert(path):
    # Open the image and convert it to grayscale
    image = Image.open(path).convert("L")
    return np.array(image)

def reconstruct(array, iters):
    u,s,vt = np.linalg.svd(array)
    print(f"SVD dimensions: dim(u)={np.shape(u)}, dim(S)={np.shape(s)}, dim(V^T)={np.shape(vt)} ")

    reconstructed = np.zeros(shape=(int(u.shape[0]), int(vt.shape[1])))

    # Iteratively compute the approximation
    # Not the fastest option, but more intuitive
    for i in range(iters):
        u_i = u[:, i]
        v_i = vt[i, :]
        reconstructed += s[i] * np.outer(u_i, v_i)
        plot_image(reconstructed, i+1)
        print(f"Rank {i+1} approximation computed. Used {i+1} singular values and {i+1} singular vectors. \n "
              f"Total values used: {np.shape(u_i)[0]} + {np.shape(v_i)[0]} * {i+1} = {(np.shape(u_i)[0] + np.shape(v_i)[0] + 1) * (i+1)} \n"
              f"Relative error: {np.linalg.norm(array - reconstructed) / np.linalg.norm(array):.4f}")

    return reconstructed

def plot_image(image, iters):
    plt.imshow(image, cmap='gray')
    plt.title(f"Image approximation of rank {iters}")
    plt.annotate(f"Rank {iters}", xy=(0.5, 0.1), xycoords='axes fraction', fontsize=12, ha='center')
    plt.show()

if __name__=="__main__":
    path = input("Enter the image path: ")
    k = int(input("Enter the number of singular values to use for reconstruction: "))
    # Convert image to matrix
    img = import_and_convert(path)

    # Compute the SVD and reconstruct the image from k vectors
    reconstructed_img = reconstruct(img, k)



