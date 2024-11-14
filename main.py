import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# CSV dosyasını oku
df = pd.read_csv("prices.csv")

# Tarih sütununu datetime formatına çevirin 
df['Date'] = pd.to_datetime(df['Date']) 
# Veriyi tarih sırasına göre sıralayın 
df = df.sort_values('Date')

# Sütun adlarını güncelle
df.columns = df.columns.str.replace('Close/Last', 'close')

# Para birimi sembollerini temizle
df['close'] = df['close'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# NaN değerleri temizle
df = df.dropna(subset=['close'])

# Veri hazırlığı fonksiyonu
def prepare_data(df, forecast_col, forecast_out, test_size):
    label = df[forecast_col].shift(-forecast_out)  # Label sütunu oluşturuluyor
    X = np.array(df[[forecast_col]])  # Özellik array'i
    X = preprocessing.scale(X)  # Özelliklerin ölçeklenmesi
    X_lately = X[-forecast_out:]  # Tahmin için kullanılan veri
    X = X[:-forecast_out]  # Eğitim ve test verisi
    label.dropna(inplace=True)  # NaN değerleri atma
    y = np.array(label)  # Etiketler
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0)  # Veri bölme
    return X_train, X_test, Y_train, Y_test, X_lately

# Veri hazırlığı
forecast_col = 'close'
forecast_out = 5
test_size = 0.2
X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size)

# Modeli eğitme
learner = LinearRegression()
learner.fit(X_train, Y_train)

# Modelin doğruluğu
score = learner.score(X_test, Y_test)

# Tahmin yapma
forecast = learner.predict(X_lately)

# Sonuçları yazdırma
response = {'test_score': score, 'forecast_set': forecast}
print(response)

# **Görselleştirme**

# Gerçek 'close' fiyatlarını zamanla çizme
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['close'], label='Real Price')

# Tahmin edilen fiyatları zamanla çizme
future_dates = df['Date'].iloc[-forecast_out:].values  # Gelecek tarihleri almak
plt.plot(future_dates, forecast, label='Predicted Price', linestyle='--', color='red')

# Başlık, etiketler ve legend
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)  # Tarihleri daha okunabilir yapmak için döndürme
plt.legend()

# Grafiği gösterme
plt.tight_layout()
plt.show()
