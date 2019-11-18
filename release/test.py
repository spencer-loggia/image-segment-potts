import models
from data import load_image
import numpy as np
import pickle
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt


X = load_image('./data/image_1.jpg')
np.random.seed(1)
model = models.MRF(J=1.2, K=4, n_em_iter=3, n_vi_iter=3) #TODO revert to k=4, both iters 3
model.fit(X=X)
pickle.dump(model, open('./models/testing', 'wb'))
predictions = model.predict(X=X)
print(predictions)

out_file = 'image_1.jpg'

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(X, cmap='gray', vmin=0, vmax=255)
plt.title('Raw data')
plt.axis('off')
ax = fig.add_subplot(1, 2, 2)
plt.imshow(predictions)
plt.title('Segmentation')
plt.axis('off')
plt.savefig('./out/' + out_file)




