import data_processing_interface as dpi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

rs_num = 42
# Denemek için bu değiştirilebilir

y = dpi.my_data['Performance Index'].values

X = dpi.my_data.drop(columns=['Performance Index']).values
# Performance Index'i droplarken geri kalanı X'e atıyoruz.

if y.ndim == 1:
    y = y.reshape((-1, 1))

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs_num)

MLR = LinearRegression()
MLR.fit(X_train, y_train)

#katsayılar
print("Coefficient ve intercept:")
print(MLR.coef_)
print(MLR.intercept_)

y_predicted = MLR.predict(X_test)

print("r2 score:")
print(r2_score(y_test, y_predicted))

plt.plot(y_test, label='gerçek')
plt.plot(y_predicted, label='tahmin')
plt.legend()
plt.show()

# Sonuç olarak diğer tüm özelliklere bakarak Performance Index'i tahmin eden modelimizin
# accuracy'si 99%'a yakındır.

