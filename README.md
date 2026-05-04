# 🎋 Sugarcane Segmentation: Inteligência Artificial para Plantio Vertical Autônomo

Este repositório contém o desenvolvimento de um sistema de visão computacional focado na detecção e segmentação de instância de toletes de cana-de-açúcar. [cite_start]O projeto constitui o núcleo de inteligência de um manipulador robótico projetado para otimizar sistemas de **plantio vertical**[cite: 345, 346].

---

## 📋 Contexto e Motivação

[cite_start]Este trabalho surge como uma evolução tecnológica de um protótipo de plantadeira vertical mecanizada desenvolvido anteriormente[cite: 347]. [cite_start]A análise técnica da versão predecessora revelou uma **complexidade mecânica excessiva** nos sistemas de dosagem e transporte[cite: 348].

[cite_start]A solução proposta consiste em simplificar o hardware mecânico, delegando a tarefa de identificação e manipulação a **braços robóticos guiados por visão computacional**, reduzindo custos de manutenção e aumentando a precisão operacional[cite: 349].

---

## 🎯 Objetivos do Projeto

O foco central deste trabalho é o desenvolvimento da camada de software e inteligência:

* [cite_start]**Criação de Dataset Especializado**: Consolidação de um banco de imagens de toletes de cana em depósitos, abordando desafios de oclusão (amontoamento) e iluminação variável[cite: 350].
* [cite_start]**Segmentação de Instância**: Treinamento do modelo **RF-DETR** para obter não apenas a posição ($x, y$), mas o contorno e a orientação exata dos toletes para a garra do robô[cite: 351].
* [cite_start]**Extração de Coordenadas**: Fornecimento de dados precisos para o cálculo de cinemática inversa do braço robótico, convertendo pixels em coordenadas de mundo real[cite: 352].

---

## 🛠️ Tecnologias e Arquitetura

O sistema fundamenta-se em tecnologias de ponta para garantir processamento em tempo real:

* **Linguagem**: Python
* [cite_start]**Framework de Deep Learning**: PyTorch [cite: 353]
* [cite_start]**Arquitetura de Visão**: **RF-DETR** (Roboflow Detection Transformer), selecionado pela sua capacidade de lidar com contextos globais de imagem via mecanismos de atenção 
* [cite_start]**Processamento Geométrico**: Algoritmos para conversão de coordenadas para o espaço tridimensional ($X, Y, Z$) [cite: 355]

---

## 📊 Métricas de Avaliação

[cite_start]Para validar a eficácia do modelo no ambiente de armazenamento, são utilizadas as seguintes métricas[cite: 356]:

* **mAP (mean Average Precision)**: Precisão geral na detecção dos objetos.
* [cite_start]**IoU (Intersection over Union)**: Qualidade e precisão da máscara de segmentação[cite: 357].
* [cite_start]**Latência de Inferência**: Tempo de processamento (ms) para garantir movimentos fluidos e em tempo real do manipulador[cite: 358].

---

## ⚖️ Licença e Propriedade Intelectual

[cite_start]Este software está em processo de **registro junto ao INPI** (Instituto Nacional da Propriedade Industrial)[cite: 359]. [cite_start]A publicação neste repositório visa a colaboração acadêmica e o desenvolvimento de soluções tecnológicas para o setor sucroenergético[cite: 360].

[cite_start]*(Verifique com o NIT da sua instituição antes de definir a licença final no GitHub - ex: MIT, Apache 2.0 ou GPLv3)*[cite: 339].