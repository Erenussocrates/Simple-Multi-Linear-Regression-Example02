import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the data
my_data = pd.read_csv("Student_Performance.csv")

# Buradan görüyoruz ki Null'umuz yok. Head'e baktığımız zaman görüyoruz ki "Extracurricular Activities"
# adında bir kategorik feature'umuz var, ve bunun seçenekleri sadece Yes ve No'dan oluşuyor.
# Sadece iki label seçeneği olduğundan ordinality'den bahsedilemeyeceğini düşündüğümden
# bu feature'un daha kolay işleme dahil olması için bu kısımda Label Encoding kullanılmasının
# yeterli olduğu kanısına vardım. Bunun üzerine, "Performance Index" feature'u en Normal Distribution
# olarak görünen feature olduğu için, bu örnekte bunu Target olarak kullanmayı seçtim.
# Korelasyonlara da baktıktan sonra görebiliriz ki arasında yüksek korelasyona sahip tek iki
# feature "Performance Index" ve "Previous Scores", ve "Performance Index"i de Target olarak kullanmaya
# karar verdiğimden, Feature'lar arası başka yüksek korelasyon kalmadığından hiçbir Feature'u
# elememeye karar verdim.

# Check for null values
def check_nulls(data):
    print("My data table has any null field: ", end="")
    print(data.isnull().values.any())

    if data.isnull().values.any():
        print(data.isnull().sum().sum())

    print("Columns of the DataFrame: ", end="")
    print(data.columns)

    if data.empty:
        print("The DataFrame is empty.")
    else:
        print("The DataFrame is not empty.")

# Display the first few rows of the data
def display_head(data):
    print(data.head())

    print("Extracurricular Activities options: ")
    print(data['Extracurricular Activities'].unique())

# Plot pairplot
def plot_pairplot(data):
    pairplot = sns.pairplot(data)
    pairplot.figure.tight_layout(pad=2.0)
    pairplot.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    plt.show()

# Plot correlation heatmap
def plot_correlation(data):
    plt.figure(figsize=(18,10))
    heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    plt.show()

    print(my_data.corr())

    # En yüksek mutlak 5 korelasyonu sıralayalım
    for col in my_data:
        en_yuksek_degerler = abs(my_data.corr()[col]).nlargest(n=5)  
        # en yüksek korelasyona sahip 5 değeri alalım
        print(en_yuksek_degerler)
        # eğer 0.75'ten büyük değer varsa yazdır.
        for index, value in en_yuksek_degerler.items():
            if 1 > value >= 0.75:
                print(index, col, "değişkenleri yüksek korelasyona sahip: ", value)

# Encode categorical feature
def encode_categorical(data):
    label_encoder = LabelEncoder()
    data['Extracurricular Activities'] = label_encoder.fit_transform(data['Extracurricular Activities'])
    return data

# Encode the categorical feature
my_data = encode_categorical(my_data)

# User interaction loop
while True:
    user_choice = input("1) Null varmı bak\n2) Head'e bak\n3) Pairplot'a bak\n4) Korelasyon'a bak\n")
    
    if user_choice == "1":
        check_nulls(my_data)
    elif user_choice == "2":
        display_head(my_data)
    elif user_choice == "3":
        plot_pairplot(my_data)
    elif user_choice == "4":
        plot_correlation(my_data)
    else:
        break
