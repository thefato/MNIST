###########################################################################
###                       TREINO DA REDE NEURAL                         ###
###########################################################################


import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # ADICIONEI Dropout para prevenir o overfitting já que comecei a aumentar a quantidade de epocas
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator # adicionei o imageDataGenerator para fazer data augmentation, que seria uma técnica de aumentar o dataset com transformações das imagens originais, como rotação, zoom, etc.



####################################################################################
#####       Função para carregar imagens BMP de uma pasta específica           #####
####################################################################################

def carregar_imagens(pasta, digits):
    images = []
    labels = []
    for filename in os.listdir(pasta):
        if filename.endswith('.bmp'):
            parts = filename.split('_')
            digit_part = parts[0]
            if digit_part.isdigit() and int(digit_part) in digits:
                filepath = os.path.join(pasta, filename)
                img = Image.open(filepath).convert('RGB') # Garante escala RGB para MNIST
                images.append(np.array(img))
                labels.append(int(digit_part))
    return np.array(images), np.array(labels)



###################################################################################
##########     Entrada do usuário para o intervalo dos dígitos      ###############     
###################################################################################

start_digit = int(input("Informe o dígito inicial (exemplo: 0): "))
end_digit = int(input("Informe o dígito final (exemplo: 3): "))

digits_to_use = list(range(start_digit, end_digit + 1))
print(f"Treinando com os dígitos: {digits_to_use}")



###################################################################################
########             Carrega e Filtra Imagens MNIST Originais            ##########
###################################################################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train_filter = (y_train >= start_digit) & (y_train <= end_digit)
test_filter = (y_test >= start_digit) & (y_test <= end_digit)

x_train_filtered = x_train[train_filter]
y_train_filtered = y_train[train_filter]
x_test_filtered = x_test[test_filter]
y_test_filtered = y_test[test_filter]


#como foi adicionado azul e branco, é necessário transformar as anteriores de escala cinza em RGB
x_train_filtered = np.stack([x_train_filtered]*3, axis=-1)  
x_test_filtered = np.stack([x_test_filtered]*3, axis=-1)   


###################################################################################
######              Carregar Imagens Invertidas e Branco e Azul          ##########
###################################################################################

folder_inverted = 'mnist_bmp_inverted_final'  # pasta que foi criada pelo ultimo inversor de cores
x_inverted_train, y_inverted_train = carregar_imagens(folder_inverted, digits_to_use)
x_inverted_test, y_inverted_test = carregar_imagens(folder_inverted, digits_to_use)



###################################################################################
######                      Combina todos os dados                       ##########
###################################################################################

x_train_total = np.concatenate((x_train_filtered, x_inverted_train), axis=0)
y_train_total = np.concatenate((y_train_filtered, y_inverted_train), axis=0)
x_test_total = np.concatenate((x_test_filtered, x_inverted_test), axis=0)
y_test_total = np.concatenate((y_test_filtered, y_inverted_test), axis=0)



###################################################################################
######                          Normalização                             ##########
###################################################################################

x_train_total = x_train_total.astype('float32') / 255.
x_test_total = x_test_total.astype('float32') / 255.



###################################################################################
######                    One hot encoding das labels                    ##########
###################################################################################

y_train_categorical = to_categorical(y_train_total, 10)
y_test_categorical = to_categorical(y_test_total, 10)



###################################################################################
######            DATA AUGMENTATION com ImageDataGenerator               ##########
###################################################################################

# foi necessário pelo fato de que as imagens poderiam estar um pouco inclinadas ou com zoom, no caso é só para tornar ele mais preciso
datagen = ImageDataGenerator(
    rotation_range=10,        # gira a imagem em até o que ta no igual, que seria 10 graus
    zoom_range=0.1,           # aplica zoom semelhante ao de cima porém depois do ponto, que no caso está em 10%
    width_shift_range=0.1,    # desloca horizontalmente mesmo funcionamento do de cima porém horizontalmente
    height_shift_range=0.1,   # desloca verticalmente  mesmo funcionamento do zoom porém verticalmente
    horizontal_flip=False,    # MNIST não se beneficia de inversão horizontal (6 vira 9, etc.)
    vertical_flip=False,      # MNIST não se beneficia de inversão vertical
    
    fill_mode='nearest'       # preenche pixels novos criados por transformações para completar a imagem mesmo
)
# prepara o gerador para o conjunto de treinamento
datagen.fit(x_train_total)



###################################################################################
######                     Construção do Modelo CNN                      ##########
###################################################################################

model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,3), padding='same'), # input_shape é (altura, largura, canais(3 por ser RGB))
    BatchNormalization(),
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,3), padding='same'), # input_shape é (altura, largura, canais(3 por ser RGB))
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3), # adicionado Dropout após o primeiro bloco Conv+Pool  sempre previnindo overfitting

    Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28,28,3), padding='same'), # input_shape é (altura, largura, canais(3 por ser RGB))
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3), # adicionado Dropout após o segundo bloco Conv+Pool  sempre previnindo overfitting

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5), # adicionado Dropout após a camada Dense (com uma taxa maior)  sempre previnindo overfitting
    Dense(10, activation='softmax') # camada de saída tem 10 neurônios para os 10 dígitos
])



###################################################################################
######                       Compilação do Modelo                        ##########
###################################################################################

model.compile(optimizer='adam',
              loss='categorical_crossentropy', # usa categorical_crossentropy porque as labels são one-hot encoded
              metrics=['accuracy'])



###################################################################################
######                        Treino do modelo                           ##########
###################################################################################
model.fit(datagen.flow(x_train_total, y_train_categorical, batch_size=(128)),
          epochs=20,
          validation_data=(x_test_total, y_test_categorical))

#aumentei o numero de epocas para ser mais preciso



###################################################################################
######                        Salvar o modelo                            ##########
###################################################################################
nome_modelo = input("Digite o nome para salvar o modelo (sem extensão): ")

arquivo_h5 = f"{nome_modelo}.h5"
model.save(arquivo_h5)
print(f"Modelo salvo em {arquivo_h5}")

arquivo_keras = f"{nome_modelo}.keras"
model.save(arquivo_keras)
print(f"Modelo salvo em {arquivo_keras}")

print("Processo concluído!")