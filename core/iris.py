from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

score_train = knn.score(X_train, y_train)
score_test = knn.score(X_test, y_test)

print("未使用最佳参数")
print(f"训练集得分: {round(score_train * 100, 2)}%")
print(f"测试集得分: {round(score_test * 100, 2)}%")

n_neighbors = tuple(range(1, 11))

cv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={"n_neighbors": n_neighbors}, cv=5)
cv.fit(X, y)
best_params = cv.best_params_
print(f"最佳参数: {best_params}")
print(f"最佳得分: {cv.best_score_}")

best_score = 0
best_n = None
for n in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)

    score_train = knn.score(X_train, y_train)
    score_test = knn.score(X_test, y_test)

    print(f"参数: {n}")
    print(f"训练集得分: {round(score_train * 100, 2)}%")
    print(f"测试集得分: {round(score_test * 100, 2)}%")

    if score_test > best_score:
        best_score = score_test
        best_n = n

print(f"最佳得分: {best_score}")
print(f"最佳参数: {best_n}")

