# Financial-Distress
Algoritma Support Vector Machine - kNearest Neighboor with Python

IMPORT LIBRARY PANDAS DAN LIBRARY MACHINE LEARNING
Import panda as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics impr accuracy_score

IMPOR DATA
#LOAD DATA
dataset = "data.csv"
df = pd.read_csv(dataset)

print(df)

MEMISAHKAN FEATURES DAN LABEL
#data splitting
features = [c for c in df.columns.values if c not in ['Nama_Perusahaan', 'Distress', 'Tahun']]
#define the label column
target = 'Distress'

Notes: karena peneliti sebelumnya menggunakan 190 record dengan pembagian 90% : 10% maka diperoleh 171 data training, dan sisanya 19 record data testing merupakan daya perusahaan objek penelitian yang dipilih secara acak oleh sistem untuk pengujian model. berikut syntax pemisah data training dan data testing

#100 data for training
train_data = df[:100]

#the last 90 data
last_data = df[100:]

#10% testing data (19 data) get from the last 90 data
test_data = last_data.sample(n=19)
X_test = test_data[features]
y_test = test_data[target]

#the reminder data
merged = last_data.merge(test_data, how='left', indicator=True)
merged = merged[merged['_merge']=='left_only']

#contact the remainder data to train data
train_data = pd.concat([train_data, merge], sort=False)

X_train = train_data[features]
y_train = train_data[target]

print("Jumlah Data Train : ",X_train.shape[0])
print("Jumlah Data Testing : ",X_test.shape[0])

1. SUPPORT VECTOR MACHINE
TRAINING DATA
#train data with 10-fold cross validaation
k_cross_val = 10
clf = SVC(kernel = 'linear', C = 10)
clf.fit(X_train, y_train)
train_accuracies = cross_val_score(clf, X_train, y_train, cv=k_cross_val, scoring='accuracy')
train_accuracy = train_accuracies.mean()
print("Train Accuracy:", train_accuracy)

TESTING DATA
#test data
y_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
print (y_pred)

2. K-NEAREST NEIGHBOR
TRAINING DATA
#train data with 10-folds cross validation
k_cross_val = 10
k_neighbors = 7
clf = KNeighborsClassifier(n_neighbor=k_neighbors)
clf.fit(X_train, y_train)
train_accuracies = cross_val_score(clf, X_train, y_train, cv=k_cross_val, scoring='accuracy')
train_accuracy =train_accuracies.mean()
print("Train Accuracy:", train_accuracy)

TESTING DATA
#test data
y_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
print (y_pred)
