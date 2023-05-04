## 1. Transfer Learning Benchmarks

| **dataset**    | **epoch (bbox/wbox)** | **moms (BlackVIP)** | **gamma (BlackVIP)** | **spsa_c (BlackVIP)** | **p_eps (BlackVIP)** | **moms (VP bbox)** | **spsa_a (VP bbox)** | **spsa_c (VP bbox)** | **init_lr (BAR)** | **min_lr (BAR)** | **lr (VP wbox)** |
|----------------|--------------|--------------|---------------|----------------|----------------|--------------|--------------|---------------|----------------|----------------|----------------|
| caltech101     | 5000 / 1000      | 0.9               | 0.2                | 0.005               | 0.2        | 0.3             | 20.0              | 0.01               | 10.0               | 0.1                 | 40.0                 |
| oxford_pets    | 5000 / 1000      | 0.9               | 0.1                | 0.01                | 1          | 0.3             | 10.0              | 0.005              | 10.0               | 0.1                 | 40.0                 |
| stanford_cars  | 2500 / 1000      | 0.9               | 0.2                | 0.01                | 0.3        | 0.5             | 10.0              | 0.01               | 10.0               | 0.1                 | 5.0                 |
| oxford_flowers | 5000 / 1000      | 0.5               | 0.2                | 0.02                | 1          | 0.3             | 10.0              | 0.005              | 5.0                | 0.1                 | 40.0                 |
| food101        | 5000 / 1000      | 0.9               | 0.1                | 0.01                | 0.3        | 0.3             | 10.0              | 0.01               | 5.0                | 0.1                 | 40.0                 |
| fgvc_aircraft  | 5000 / 1000      | 0.3               | 0.1                | 0.01                | 0.5        | 0.3             | 10.0              | 0.01               | 10.0               | 0.1                 | 40.0                 |
| sun397         | 1000 / 500       | 0.5               | 0.1                | 0.01                | 1          | 0.3             | 20.0              | 0.01               | 5.0                | 0.01                | 40.0                 |
| dtd            | 5000 / 1000      | 0.7               | 0.2                | 0.01                | 0.3        | 0.9             | 10.0              | 0.01               | 5.0                | 0.1                 | 40.0                 |
| svhn           | 5000 / 1000      | 0.9               | 0.2                | 0.005               | 1          | 0.9             | 10.0              | 0.01               | 5.0                | 0.1                 | 40.0                 |
| eurosat        | 5000 / 1000      | 0.9               | 0.2                | 0.005               | 0.4        | 0.3             | 10.0              | 0.005              | 5.0                | 0.01                | 40.0                 |
| resisc45       | 5000 / 1000      | 0.95              | 0.1                | 0.01                | 0.3        | 0.3             | 10.0              | 0.01               | 5.0                | 0.01                | 40.0                 |
| clevr          | 5000 / 1000      | 0.9               | 0.2                | 0.005               | 1          | 0.9             | 20.0              | 0.01               | 10.0               | 0.1                 | 40.0                 |
| ucf101         | 5000 / 1000      | 0.9               | 0.1                | 0.01                | 0.3        | 0.3             | 20.0              | 0.01               | 5.0                | 0.01                | 40.0                 |
| imagenet       | 500  / 400       | 0.9               | 0.2                | 0.005               | 0.3        | 0.3             | 20.0              | 0.01               | 10.0               | 0.1                 | 1.0                 |

* bbox: black-box setting
* wbox: white-box setting

<hr />

## 2. Synthetic Datasets

| **dataset**    | **moms (BlackVIP)** | **alpha (BlackVIP)** | **spsa_a (BlackVIP)** | **spsa_c (BlackVIP)** | **p_eps (BlackVIP)** | **moms (VP bbox)** | **alpha (VP bbox)** | **spsa_a (VP bbox)** | **spsa_c (VP bbox)** | **init_lr (BAR)** | **min_lr (BAR)** | **lr (VP wbox)** |
|-----------------------|--------------|---------------|--------------|--------------|----------------|----------------|--------------|--------------|--------------|---------------|----------------|----------------|
| colour_biased_mnist (0.8/0.2) | 0.9      | 0.4  | 0.01          | 0.01         | 1         | 0.9         | 0.4      | 10.0         | 0.005         | 5.0           | 0.1            | 40.0                 |
| colour_biased_mnist (0.9/0.1) | 0.9      | 0.4  | 0.01          | 0.01         | 1         | 0.9         | 0.4      | 10.0         | 0.005         | 5.0           | 0.1            | 40.0                 |
| locmnist (1:1)                | 0.9      | 0.5  | 0.01          | 0.005        | 1         | 0.9         | 0.5      | 10.0         | 0.01          | 5.0           | 0.5            | 10.0                 |
| locmnist (1:4)                | 0.95     | 0.5  | 0.02          | 0.01         | 1         | 0.9         | 0.5      | 10.0         | 0.01          | 5.0           | 0.01           | 10.0                 |

* We searched the hyperparameters of all methods on the 16-shot training set and shared them for 32-shot inference.

<hr />

## 3. Ablation Study

### 3.1. Archtecture

### 3.1.1. BlackVIP

| **coor_backbone** | **tar_backbone** | **alpha** | **moms** | **gamma** | **spsa_c** |
|-------------------|------------------|-----------|----------|-----------|------------|
| vit-mae-base      | rn50             | 0.4       | 0.3      | 0.1       | 0.01       |
| vit-mae-base      | rn101            | 0.4       | 0.3      | 0.2       | 0.005      |
| vit-mae-base      | vit_b32          | 0.4       | 0.9      | 0.1       | 0.01       |
| vit-mae-base      | vit_b16          | 0.4       | 0.9      | 0.2       | 0.005      |
| dino-resnet-50    | rn50             | 0.5       | 0.9      | 0.2       | 0.01       |
| dino-resnet-50    | rn101            | 0.4       | 0.5      | 0.2       | 0.01       |
| dino-resnet-50    | vit_b32          | 0.5       | 0.9      | 0.1       | 0.01       |
| dino-resnet-50    | vit_b16          | 0.4       | 0.9      | 0.2       | 0.01       |

### 3.1.2. BAR and VP w/ SPSA-GC


### 3.2. Pre-trained Weights and Optimization Algorithm

<hr />

## 4. Additional Information
* **Runtime**: In our paper, we showed the possibility of black-box adaptation of the visual foundation model, but for some datasets, many iterations (API calls) are required to improve performance sufficiently. As a result, training takes longer than one expected.
* **Sensitivity**: The few-shot evaluation protocol and zeroth-order gradient approximiation make the training unstable. The performance volatility of black-box methods are relatively large than white-box methods across not only hyperparameters, but also to random seed, GPU, pytorch version.