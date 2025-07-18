# KNN Meme Kanseri Sınıflandırma Projesi

## İçindekiler
Genel Bakış

Veri Seti Bilgisi

Kullanılan Teknolojiler

Proje Adımları

Sonuçlar

Görselleştirme

Kurulum ve Kullanım

## Genel Bakış
Bu proje meme tümörlerinin iyi huylu veya kötü huylu olup olmadığını tahmin etmek için K-En Yakın Komşu (KNN) algoritması kullanır. Veri ön işleme, model eğitimi, hiperparametre ayarlama ve performans değerlendirme adımları içermektedir.


## Veri Seti Bilgisi
Kaynak: sklearn.datasets.load_breast_cancer 


Toplam Örnek: 569

Özellik Sayısı: 30

Sınıflar:

0 =Kötü Huylu

1 =İyi Huylu

## ⚙️ Kullanılan Teknolojiler
Python 3.10+

scikit-learn

pandas

matplotlib


## Proje Adımları

### kütüphane kurumlumları 

kod dosyasının başına 

veri seti için: sklearn

veri analizi için: pandas

veri görselleştirme için: matplotlib kütüphanelerinin ilgili bçlümlerini kurduk.

<img width="660" height="173" alt="Ekran görüntüsü 2025-07-18 110611" src="https://github.com/user-attachments/assets/9f0e2b81-e4c1-4b8b-8a1e-139155495466" />

### veri seti incelemesi
<img width="648" height="94" alt="image" src="https://github.com/user-attachments/assets/0d8b60cf-4f25-4e46-9639-13a3bfebe807" />

load_breast_cancer(): Scikit-learn kütüphanesinden gelen, meme kanseri verilerini içeren hazır bir veri setini yükler.

pd.DataFrame(): Bu verileri pandas kullanarak bir tablo (dataframe) formatına çeviririz.

df["target"] = cancer.target: Hedef sınıf (iyi huylu veya kötü huylu) verisini de tabloya ekleriz.

Amacımız veri setini daha rahat analiz edebilmek için tablo haline getirmek ve hedef sınıfı dahil etmek.


### Makine Öğrenmesi Modeli Seçimi 

Kullanılacak model olarak K-En Yakın Komşu (KNN) algoritması tercih edilmiştir.

```python
from sklearn.neighbors import KNeighborsClassifier
```

KNN, örneklerin en yakın komşularına bakarak sınıflandırma yapan denetimli bir algoritmadır.

### Modelin Eğitilmesi (Training)

```python
X = cancer.data
y = cancer.target
```

X: Girdi değişkenleri (özellikler features).

y: Çıkış (etiket target), yani her örneğin iyi huylu mu yoksa kötü huylu mu olduğu bilgisi.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Veri seti eğitim ve test olmak üzere ikiye bölündü

%70 eğitim, %30 test olarak ayrıldı (bazı kaynaklarda %80 %20 de olabiliyor). random_state ise tekrar çalıştırıldığında aynı sonucu vermesi için sabitlenmiştir(genelde 42 sayısı kullannılır).

```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```

Komşu sayısı k=3 seçilerek KNN modeli oluşturulur ve eğitim verisi ile eğitilir.

Test verisi üzerinden tahmin yapılır.

```python
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)
print("dogruluk:", accuracy)

```

Sonuçlarıfonksiyonlarla 1 üzerinden değerlendiriyoruz. confusion matrix ile doğru ve yanlış sınıflandırılan örnekler detaylı şekilde analiz edilir.

### Hiperparametre Ayarlaması (k Değerinin Belirlenmesi - optimizasyon ayArı)

```python
accuracy_vaules = []
k_values = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_vaules.append(accuracy)
    k_values.append(k)
```

Farklı k değerleri denenerek doğruluk değerleri hesaplanır.

Amacımız hangi k değeriyle modelin en iyi sonucu verdiğini bulmak.

```python
plt.figure()
plt.plot(k_values, accuracy_vaules, marker= 'o', linestyle = '-')
plt.title('k degerine gore dogruluk')
plt.xlabel('k degeri')
plt.ylabel('dogruluk')
plt.xticks(k_values)
plt.grid(True)
```

k değerine göre doğruluk oranı tablo haline getirilir.

tablo grafiği sayesinde hangi k değerinin model için en uygun olduğu daha net anlaşılır. (doğruluğu yüksek olan en küçük k değerini alırız)

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/5c4ab85d-0498-45bc-8941-473fa0831b51" />

grafikte görüldüğü üzere 9,10,11,12 değerleri uygun. en küçüğü 9, işlem parçacığını yormamak için onu kullanırız.

## Kurulum ve kullanım

gerekli kütüphaneleri kurun

```python
pip install scikit-learn pandas matplotlib
```
python kod dosyasını çalıştırın

```python
python KNN_1.py
```

## Geliştirici

**İsmail Enes Ertaş**  
Bilgisayar Mühendisliği Öğrencisi

🔗 [GitHub Profilim](https://github.com/ismailenesertas)  
🔗 [LinkedIn Profilim](https://www.linkedin.com/in/ismail-enes-ertas)











