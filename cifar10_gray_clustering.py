import pickle
import numpy as np

batch = pickle.load(open("/workspace/CHAHAK/dsml/project/data/cifar-10-batches-py/gray_data_batch_1.pkl", "rb"), encoding="bytes")

batch.keys()

image_data = batch[b'data'].reshape(10000, -1, 1)
image_data = image_data.squeeze()
image_data.shape

from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

kmeans = KMeans(n_clusters=10)
estimator = make_pipeline(StandardScaler(), kmeans).fit(image_data)
kmeans.labels_[:10]

from sklearn.metrics import homogeneity_score

homogeneity_score(batch[b'labels'], estimator[-1].labels_)

from tqdm import tqdm

def compress_img(img, k):
    img = img.reshape(32, 32)
    U, D, V = np.linalg.svd(img, full_matrices=False)
    compressed_image = U[:, :k] @ np.diag(D[:k]) @ V[:k, :]
    return compressed_image

compressed_images = []
k = 10
for img in tqdm(image_data):
    cimg = compress_img(img, 10)
    cimg_vector = cimg.reshape(-1,)
    compressed_images.append(cimg_vector)

compressed_images = np.asarray(compressed_images)
compressed_images.shape

kmeans = KMeans(n_clusters=10)
svd_estimator = make_pipeline(StandardScaler(), kmeans).fit(image_data)
homogeneity_score(batch[b'labels'], svd_estimator[-1].labels_)

import numpy as np
import matplotlib.pyplot as plt

test = image_data[0].reshape(32, 32)
_, d, _ = np.linalg.svd(test)
fig, ax = plt.subplots()
ax.plot(d, 'ko', markersize=2, label="Singular values")
ax.grid()
ax.legend()
