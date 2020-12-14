import keras
import keras.layers as layers
import pandas as pd
from project.utils import *

train_data = load_train_csv("../data")
# You may optionally use the sparse matrix.
sparse_matrix = load_train_sparse("../data")
val_data = load_valid_csv("../data")
test_data = load_public_test_csv("../data")

X = np.array([train_data['user_id'], train_data['question_id']])
y = np.array(train_data['is_correct'])

model = keras.Sequential(
    [
        layers.Dense(542, input_dim=542, activation='relu'),
        layers.Dense(1774, input_dim=1774, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epohcs=100, batch_size=40)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))