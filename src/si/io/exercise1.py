

from src.si.io.csv_file import read_csv

# EXERCICIO 1
#1.1
path = 'C:\Users\joana\OneDrive\Documentos\GitHub\si\datasets\iris\iris.csv'
dataset = read_csv(path,features=True, label=True)

print(dataset)

#1.2

penultima_feature = dataset.X[:, -2]
dimensao = penultima_feature.shape

print("Penultimate Independent Variable:")
print(penultima_feature)
print("Dimension of the Resulting Array:")
print(dimensao)

#1.3

last_10_samples = dataset.X[-10:]
mean = last_10_samples.mean(axis=0)

print("Last 10 Samples:")
print(last_10_samples)
print("Mean for Each Independent Variable/Feature:")
print(mean)

#1.4

new_selection_samples = dataset.X[dataset.X <= 6].all(axis=1)
num_selected_samples = new_selection_samples.shape[0]

exclude_class = 'Irissetosa'

selected_samples = dataset.data[dataset.target != exclude_class]
num_selected_samples = selected_samples.shape[0]

