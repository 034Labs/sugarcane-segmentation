# 🎋 Sugarcane Segmentation: Inteligência Artificial para Plantio Vertical Autônomo

Este repositório contém o desenvolvimento de um sistema de visão computacional focado na detecção e segmentação de instância de toletes de cana-de-açúcar. O projeto constitui o núcleo de inteligência de um manipulador robótico projetado para otimizar sistemas de **plantio vertical**.

---

## 📋 Contexto e Motivação

Este trabalho surge como uma evolução tecnológica de um protótipo de plantadeira vertical mecanizada desenvolvido anteriormente. A análise técnica da versão predecessora revelou uma **complexidade mecânica excessiva** nos sistemas de dosagem e transporte.

A solução proposta consiste em simplificar o hardware mecânico, delegando a tarefa de identificação e manipulação a **braços robóticos guiados por visão computacional**, reduzindo custos de manutenção e aumentando a precisão operacional.

---

## 🎯 Objetivos do Projeto

O foco central deste trabalho é o desenvolvimento da camada de software e inteligência:

* **Criação de Dataset Especializado**: Consolidação de um banco de imagens de toletes de cana em depósitos, abordando desafios de oclusão (amontoamento) e iluminação variável.
* **Segmentação de Instância**: Treinamento do modelo **RF-DETR** para obter não apenas a posição ($x, y$), mas o contorno e a orientação exata dos toletes para a garra do robô.
* **Extração de Coordenadas**: Fornecimento de dados precisos para o cálculo de cinemática inversa do braço robótico, convertendo pixels em coordenadas de mundo real.

---

## 🛠️ Tecnologias e Arquitetura

O sistema fundamenta-se em tecnologias de ponta para garantir processamento em tempo real:

* **Linguagem**: Python
* **Framework de Deep Learning**: PyTorch 
* **Arquitetura de Visão**: **RF-DETR** (Roboflow Detection Transformer), selecionado pela sua capacidade de lidar com contextos globais de imagem via mecanismos de atenção 
* **Processamento Geométrico**: Algoritmos para conversão de coordenadas para o espaço tridimensional ($X, Y, Z$) 

---

## 📊 Métricas de Avaliação

Para validar a eficácia do modelo no ambiente de armazenamento, são utilizadas as seguintes métricas:

* **mAP (mean Average Precision)**: Precisão geral na detecção dos objetos.
* **IoU (Intersection over Union)**: Qualidade e precisão da máscara de segmentação.
* **Latência de Inferência**: Tempo de processamento (ms) para garantir movimentos fluidos e em tempo real do manipulador.

---

## ⚖️ Licença e Propriedade Intelectual

Este software está em processo de **registro junto ao INPI** (Instituto Nacional da Propriedade Industrial). A publicação neste repositório visa a colaboração acadêmica e o desenvolvimento de soluções tecnológicas para o setor sucroenergético.

---

## 🚀 Quick Start (RF-DETR Study Code)

> Nota: o contexto deste repositório descreve o objetivo final em cana-de-acucar, mas o script atual foi estruturado para **estudo da framework RF-DETR** com classes COCO (ex.: `person`).

### 1) Instalar dependencias

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Rodar inferencia em imagem

```bash
python main.py --image images/junior-galdino-066.jpg --class person --threshold 0.3 --output results/output
```

Saida esperada: `results/output.jpg`

### 3) Rodar inferencia em video

```bash
python main.py --video videos/pessoas.mp4 --class person --threshold 0.3 --output results/output
```

Saida esperada: `results/output.mp4`

### 4) Parametros uteis

- `--class`: classe COCO alvo (padrao: `person`)
- `--threshold`: limiar de confianca
- `--resize`: fator de escala de entrada/saida (ex.: `0.5`, `1.0`, `1.5`)
- `--output`: prefixo do arquivo de saida (sem extensao)

### 5) Estrutura de codigo (modular)

- `main.py`: ponto de entrada
- `segmentation_app/cli.py`: argumentos, modelo e roteamento
- `segmentation_app/inference.py`: inferencia por frame
- `segmentation_app/pipeline.py`: pipeline de imagem/video e metricas
