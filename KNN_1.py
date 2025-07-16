# sklearn : ML libary
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# 1 Veri seti incelemesi
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"]= cancer.target

# 2 Makine öğrenmesi modelinin seçilmesi (knn sınıfladırma)

# 3 Modelin train edilmesi
X = cancer.data #features
y = cancer.target #target

# train test split
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size = 0.3, random_state=42)

#olcekledirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# knn modeli oluşturma ve train etme
knn = KNeighborsClassifier(n_neighbors=3) # Model oluşturma komsu parametresini unutma
knn.fit(X_train, y_train) # fit fonk verimizi(samples, target) kullanarak knn algoritması eğitir

# 4 Sonuçların değerlendirilmesi 
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)
print("dogruluk:", accuracy_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion matrix: ", conf_matrix)



# 5 Hiperparametre ayarlaması

accuracy_vaules = []
k_values = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_vaules.append(accuracy)
    k_values.append(k)
    
    
plt.figure()
plt.plot(k_values, accuracy_vaules, marker= 'o', linestyle = '-')
plt.title('k degerine gore dogruluk')
plt.xlabel('k degeri')
plt.ylabel('dogruluk')
plt.xticks(k_values)
plt.grid(True)
    