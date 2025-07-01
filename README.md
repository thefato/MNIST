# DeepLearning MNIST Project

Este projeto utiliza técnicas de Deep Learning para classificar dígitos manuscritos do famoso dataset MNIST. O objetivo é criar um modelo robusto, capaz de reconhecer dígitos não apenas em sua forma original, mas também em variações de cores e invertidas.

## Estrutura do Projeto

O projeto é organizado em scripts sequenciais e pastas para gerenciar o fluxo de dados:

- **`extratorImagensMNIST.py`**: Script inicial que baixa o dataset MNIST e extrai as imagens para um formato utilizável.
- **`inversorCores.py`**: Processa as imagens extraídas, invertendo suas cores (por exemplo, de preto no branco para branco no preto).
- **`inversorCoresAzulBranco.py`**: A partir das imagens invertidas, gera variações adicionais, transformando-as em esquemas de cores azul e branco.
- **`treinaComDigitosOriginaisInvertidos.py`**: O script principal de treinamento, que combina os dígitos originais do MNIST com todas as variações geradas (invertidas e coloridas) para treinar um modelo de Rede Neural Convolucional (CNN).
- **`reconhecedorAvancadoDigitos.py`**: Um script para inferência que permite carregar um modelo já treinado e reconhecer dígitos em novas imagens, oferecendo uma interface interativa para seleção de modelo e imagem.
- **`mnist_bmp_inverted_final/`**: Diretório criado e populado pelos scripts de inversão, contendo as imagens processadas.
- **`imgs/`**: Diretório opcional para armazenar imagens de teste que podem ser usadas pelo `reconhecedorAvancadoDigitos.py`.
- **`requirements.txt`**: Lista de todas as dependências Python necessárias para o projeto.
- **Modelos Salvos (`.keras` / `.h5`)**: Os modelos treinados são salvos nestes formatos pelo script de treinamento.

## Como Usar

1. Instale as dependências:
    ```bash
    # Crie e ative o ambiente virtual (Ubuntu/macOS)
    python3 -m venv venv
    source venv/bin/activate
    
    # Para Windows, use:
    # python -m venv venv
    # .\venv\Scripts\activate
    
    # Instale todas as dependências listadas no arquivo requirements.txt
    pip install -r requirements.txt
    ```
2. Execute o treinamento do modelo nessa ordem(treinamento ja feito, e arquivos mantidos):
    ```bash
    # Passo 1: Extrair imagens do MNIST
    python extratorImagensMNIST.py
    
    # Passo 2: Inverter cores das imagens (cria 'mnist_bmp_inverted_final' com imagens invertidas)
    python inversorCores.py
    
    # Passo 3: Gerar variações azul e branco (atualiza 'mnist_bmp_inverted_final')
    python inversorCoresAzulBranco.py
    
    # Passo 4: Treinar o modelo com os dados combinados e aumentados
    # Este script solicitará o intervalo de dígitos (ex: 0 a 3) e, ao final, o nome para salvar o modelo treinado.
    python treinaComDigitosOriginaisInvertidos.py
   
    # Passo 5: O script irá listar os modelos treinados disponíveis na pasta atual
    # e as imagens na pasta 'imgs/' (se existir), pedindo para você escolher qual usar.
    python reconhecedorAvancadoDigitos.py
    # Caso queira apenas reconhecer os digitos, execute: python reconhecedorAvancadoDigitos.py, ele irá mostrar os modelos treinados, você escolhe o modelo que deseja usar.
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

## Autores
- Gabriel Fonseca
- Ernesto Terra dos Santos
- Marcus Apolinário
- Gabriel Antonietti
