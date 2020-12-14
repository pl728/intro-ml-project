from project.utils import *
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from project.part_b.irt_subject import load_questions_meta_csv, load_subject_meta_csv
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor

train_data = load_train_csv("../data")
# You may optionally use the sparse matrix.
sparse_matrix = load_train_sparse("../data")
val_data = load_valid_csv("../data")
test_data = load_public_test_csv("../data")
question_metadata = load_questions_meta_csv("../data/question_meta.csv")
subject_metadata = load_subject_meta_csv("../data/subject_meta.csv")

X = np.array([train_data['user_id'], train_data['question_id']]).reshape((56688, 2))
y = np.array(train_data['is_correct']).reshape((56688,))

# model = DecisionTreeClassifier()
model = RandomForestClassifier(n_estimators=500, max_depth=30)
model.fit(X, y)

X_test, y_test = np.array([test_data['user_id'], test_data['question_id']]).reshape((3543, 2)), np.array(
    test_data['is_correct']).reshape((3543,))
y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    print(y_pred[i])

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
