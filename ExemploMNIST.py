######################################################
###            D E E P   L E A R N I N G           ###
######################################################
### Exemplo de Deep Learning utilizando MNIST      ###
### Prof. Filipo Mor - 02 de junho de 2025         ###
######################################################

# Importar bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Carregar o dataset MNIST (imagens de dígitos manuscritos)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Pré-processamento: normalizar as imagens para valores entre 0 e 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Visualizar as primeiras 9 imagens do conjunto de treinamento
fig, axes = plt.subplots(3, 3, figsize=(8,8))
for i in range(9):
    # Calcula a posição na grade
    row = i // 3
    col = i % 3
    # Exibe a imagem
    axes[row, col].imshow(train_images[i], cmap='gray')
    axes[row, col].set_title(f'Label: {train_labels[i]}')  # mostra o rótulo
    axes[row, col].axis('off')  # remove os eixos para melhor visualização
plt.tight_layout()
plt.show()

# Construção do modelo de rede neural
model = models.Sequential()
# Adiciona uma camada deFlatten para transformar as imagens 28x28 em um vetor de 784 elementos
model.add(layers.Flatten(input_shape=(28, 28)))
# Adiciona uma camada oculta com 128 neurônios e função de ativação ReLU
model.add(layers.Dense(128, activation='relu'))
# Adiciona uma camada de saída com 10 neurônios (para dígitos 0-9), usando softmax para classificação
model.add(layers.Dense(10, activation='softmax'))

# Compilação do modelo:
# - Otimizador: 'adam', eficiente para muitos problemas
# - Função de perda: 'sparse_categorical_crossentropy', adequada para classificação com labels inteiros
# - Métrica: 'accuracy' para acompanhar a precisão durante o treinamento
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
# - epochs=5: número de passagens pelo dataset
# - validation_split=0.2: usar 20% dos dados de treino para validação
model.fit(train_images, train_labels, epochs=5, validation_split=0.2)
model.save('mnist_model001.h5')

test_loss, test_acc = model.evaluate(test_images, test_labels)
# Avaliação do modelo com o dataset de teste
test_loss, test_acc = model.evaluate(test_images, test_labels)
model.save('model001.keras')
print(f"Precisão no teste: {test_acc:.4f}")
