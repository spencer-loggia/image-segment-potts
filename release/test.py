import models
from data import load_image


X = load_image('./data/image_1.jpg')
model = models.MRF(J=1.2, K=4, n_em_iter=3, n_vi_iter=3)
model.fit(X=X)


