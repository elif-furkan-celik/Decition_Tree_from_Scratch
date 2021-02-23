import numpy as np
import pandas as pd
from pprint import pprint

dataset = pd.read_csv('/content/drive/MyDrive/zoo.csv',
                      names=['hair','feathers','eggs','milk',
                                                   'airbone','aquatic','predator','toothed','backbone',
                                                  'breathes','venomous','fins','legs','tail','domestic','catsize','class',])#Özelllik isimleri ve label ismi yandaki gibi olacak şekilde tüm datayı okuduk
print(dataset)

"""Datasetteki veriler:"""

for i in dataset.columns:
    print(dataset[i].value_counts())#Burada datasetteki her bir özelliğin kaç farklı değeri ve onların kaç sayıda olduğunu gösterir.

"""Feature Selection:"""

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

embeded_lr_feature = []

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=5) #Logistic regression tabanlı feature selection modeli oluşturur.
embeded_lr_selector.fit(dataset[dataset.columns[:-1]], dataset['class'])#Modele datanı ve labelını vererek çalıştırır.
embeded_lr_support = embeded_lr_selector.get_support()#En iyi özellikleri seçer
embeded_lr_feature.append(dataset[dataset.columns[:-1]].loc[:,embeded_lr_support].columns.tolist())#En iyi özelliklerin isimlerini alır.

print(embeded_lr_feature)

for i in dataset.columns[:-1]:
    if i != embeded_lr_feature[0][0] and i != embeded_lr_feature[0][1] and i != embeded_lr_feature[0][2] and i != embeded_lr_feature[0][3] and i != embeded_lr_feature[0][4]:#Datasetteki en iyi özellikler dışındaki tüm özellikleri siler.
        dataset = dataset.drop(i, axis=1)
print(dataset)

"""Normalization: (Burada her biri iki class olduğu için değerler değişmedi.)"""

for i in dataset.columns[:-1]:
    dataset[i][:] = list(map(lambda x: ((x-min(dataset[i][:])) / (max(dataset[i][:]) - min(dataset[i][:]))) , dataset[i][:]))#Datasetteki tüm değerleri max-min normalization methoduyla(0-1 aralığına) normalize eder.
print(dataset)

"""Histogramlar:"""

dataset['feathers'].hist()#feathers özelliğinin histogramını çizer.

dataset['toothed'].hist()#toothed özelliğinin histogramını çizer.

dataset['backbone'].hist()#backbone özelliğinin histogramını çizer.

dataset['breathes'].hist()#breathes özelliğinin histogramını çizer.dataset['backbone'].hist()#safety özelliğinin histogramını çizer.

dataset['tail'].hist()#tail özelliğinin histogramını çizer.

dataset['class'].hist()#class labelının histogramını çizer.

train_data = dataset.iloc[:int(dataset.shape[0]*80/100)].reset_index(drop=True)#Datanın %75 ini train yapıyoruz.
test_data = dataset.iloc[int(dataset.shape[0]*80/100):].reset_index(drop=True)#Datanın %25 sini test yapıyoruz.
print(train_data.shape, test_data.shape)

"""Entropy hesaplama:"""

def entropi(kolon):
    labellar, sayılar = np.unique(kolon, return_counts = True) # bir kolondaki farklı değerleri ve o değerlerin sayılarını bulur.
    entropy = 0
    #bu kolondaki toplam etropy'i hesaplar.
    for i in range(len(labellar)):
      entropy += (-sayılar[i] / np.sum(sayılar)) * np.log2(sayılar[i] / np.sum(sayılar)) # fonksiyonu her değer için tek tek uygulayıp toplar.
    return entropy

"""Bilgi kazancını hesaplama:"""

def bilgi_kazancı(data, özellik, label):
    data_entropi = entropi(data[label]) #label için entropy değerini hesaplar
    özellikler, özellik_sayıları = np.unique(data[özellik], return_counts = True)

    ağırlıklı_özellik_entropileri = 0
    # her bir label değeri için ağırlıklı olan entropi değerini bulup toplar
    for i in range(len(özellikler)):
        ağırlıklı_özellik_entropileri += (özellik_sayıları[i] / np.sum(özellik_sayıları)) * (entropi(data.where(data[özellik] == özellikler[i]).dropna()[label]))#labelların ağırlıklı entropi değerlerini bulur

    kazanc = data_entropi - ağırlıklı_özellik_entropileri #bilgi kazancını hesaplar

    return kazanc

"""Kara ağacını oluşturma fonksiyonu:"""

def karar_ağacı(data, gerçek_data, özellikler, label="class", ebeveyn = None):
    if len(np.unique(data[label])) <= 1:#Label sayısı 1 yada 1 den azsa ilk labelı döndürür.
        return np.unique(data[label])[0]

    elif len(data)==0:#Data boşsa en fazla sayıdaki labelı döndürür.
        return np.unique(gerçek_data[label])[np.argmax(np.unique(gerçek_data[label],return_counts=True)[1])]
    
    elif len(özellikler) ==0:#Hiç özellik yoksa ebeveynini döndürür.
        
        return ebeveyn
  
    else:
        ebeveyn = np.unique(data[label])[np.argmax(np.unique(data[label], return_counts=True)[1])]

        bilgi_kazancı_değerleri = []

        for özellik in özellikler:
            bilgi_kazancı_değerleri.append(bilgi_kazancı(data, özellik, label)) #Özelliklerin bilgi kazançlarını döndürür

        bilgi_kazancı_değerleri = np.array(bilgi_kazancı_değerleri)
        
        en_iyi_değer_indeksi = np.argmax(bilgi_kazancı_değerleri)#En iyi değerin indeksini bulur.
        
        en_iyi_değer = özellikler[en_iyi_değer_indeksi]#En iyi değeri bulur.

        ka_ağacı = {en_iyi_değer:{}}#En iyi özelliği karar ağacına ekler.

        geçici_array = []
        #Tüm özelliklerden en iyi özelliği çıkararak yeni özellikler array i oluşturur.
        for özellik in özellikler:
            if özellik != en_iyi_değer:
                geçici_array.append(özellik)
                
        özellikler = np.array(geçici_array)
        
        for değer in np.unique(data[en_iyi_değer]):

            alt_data = data.where(data[en_iyi_değer] == değer).dropna()#Datasetten en iyi özelliği çıkarıp yeni bir alt data oluşturur.
            
            alt_karar_ağacı = karar_ağacı(alt_data, gerçek_data, özellikler, label, ebeveyn)#Karar ağacı fonksiyonunu alt data için bir daha çağırır.
            
            ka_ağacı[en_iyi_değer][değer] = alt_karar_ağacı#Karar ağacına değerleri ekleyerek oluşturur.
            
        return ka_ağacı

k_ağacı = karar_ağacı(train_data, train_data, train_data.columns[:-1])#karar_ağacı fonksiyonunu çağırarak karar ağacını oluşturur.

"""Karar ağacını yazdırma:"""

pprint(k_ağacı)#Karar ağacını pprint() fonksiyonunu kullanarak yazdırır.

"""Karar ağacına daha düzgün bir görünüş kazandırma:"""

def printTree(tree, d = 0):
    if (tree == None or len(tree) == 0):
        print("\t" * d, "-")
    else:
        for key, val in tree.items():#Bu döngüde karar ağacının her bir itemine tek tek bakarak itemin yerine göre boşluk veya parantez ekleyerek karar ağacına daha düzgün bir görüntü kazandırır.
            if (isinstance(val, dict)):
                print("\t" * d, key)
                printTree(val, d+1)
            else:
                print("\t" * d, key, str('(') , val , str(')'))

printTree(k_ağacı)

def tahmin(dizi, karar_ağacı):
    for anahtar in list(dizi.keys()):
        if anahtar in list(karar_ağacı.keys()):#Anahtarın karar ağacında olup olmadığına bakar.
            try:
                sonuç = karar_ağacı[anahtar][dizi[anahtar]]#Anahtarın karar ağacındaki yerini saptar.
            except:
                return -1

            if isinstance(sonuç, dict):
                return tahmin(dizi,sonuç)
            else:
                return sonuç

def test(data, k_ağacı):
    
    diziler = data.iloc[:,:-1].to_dict(orient = "records")#Yeni bir dizi oluşturarak datayı labeldan arındırır.
    
    tahmin_edilen = pd.DataFrame(columns=["tahmin_edilen"])#Tahmin değerlerini kaydetmek için bir dataframe oluşturur.
    
    for i in range(len(data)):
        tahmin_edilen.loc[i,"tahmin_edilen"] = tahmin(diziler[i], k_ağacı)#Tek tek datadaki her bir örneği tahmin eder.
         
    print('Tahmin doğruluğu = {}'.format(np.sum(tahmin_edilen["tahmin_edilen"] == data["class"])/len(data)))#Tahmin edilen labelları gerçek labellarla karşılaştırarak doğruluk oranını bulur.

test(test_data, k_ağacı)

print(tahmin({'feathers': 1, 'toothed': 0, 'backbone': 1, 'breathes': 0, 'tail': 0}, k_ağacı))