# DeepLearning MNIST Project

Este projeto utiliza técnicas de Deep Learning para classificar dígitos manuscritos do dataset MNIST.

## Estrutura do Projeto

- **extratorImagensMNIST.py**: Extrai imagens do dataset MNIST.
- **inversorCores.py**: Inverte as cores das imagens para gerar variações.
- **inversorCoresAzulBranco.py**: Gera variações com as cores azul e branco
- **treinaComDigitosOriginaisInvertidos.py**: Treina o modelo com dígitos originais e suas variações(invertida e outras cores).
- **reconhecedorAvancadoDigitos.py**: Reconhece dígitos utilizando o modelo treinado.

## Como Usar

1. Instale as dependências:
    ```bash
    pip install tensorflow
    pip install numpy
    pip install pillow
    ```
2. Execute o treinamento do modelo nessa ordem(treinamento ja feito, e arquivos mantidos):
    ```bash
    1- python extratorImagensMNIST.py
    2- python inversorCores.py
    3- python inversorCoresAzulBranco.py
    4- python treinaComDigitosOriginaisInvertidos.py
    5- python reconhecedorAvancadoDigitos.py
    ```

## Referências

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Documentação PyTorch](https://pytorch.org/docs/stable/index.html)
- [Documentação Prof Filipo](https://github.com/ProfessorFilipo/PythonAI)


## Resultados Esperados

Após o treinamento, o modelo é capaz de reconhecer dígitos manuscritos em diferentes combinações de cores:

 preto com fundo branco;
 branco com fundo preto;
 branco com fundo azul;
 azul com fundo branco.

O modelo treinado mantém alta precisão e pode ser testado com imagens personalizadas.
