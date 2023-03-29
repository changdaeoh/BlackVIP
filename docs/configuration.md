## 1. Transfer Learning Benchmarks

| **dataset**    | **epoch** | **blackvip-moms** | **blackvip-gamma** | **blackvip-spsa_c** | **blackvip-p_eps** |
|----------------|-----------|-------------------|--------------------|---------------------|---------------------|
| caltech101     | 5000      | 0.9               | 0.2                | 0.005               | 0.2                 |
| oxford_pets    | 5000      | 0.9               | 0.1                | 0.01                | 1                   |
| stanford_cars  | 2500      | 0.9               | 0.2                | 0.01                | 0.3                 |
| oxford_flowers | 5000      | 0.5               | 0.2                | 0.02                | 1                   |
| food101        | 5000      | 0.9               | 0.1                | 0.01                | 0.3                 |
| fgvc_aircraft  | 5000      | 0.3               | 0.1                | 0.01                | 0.5                 |
| sun397         | 1000      | 0.5               | 0.1                | 0.01                | 1                   |
| dtd            | 5000      | 0.7               | 0.2                | 0.01                | 0.3                 |
| svhn           | 5000      | 0.9               | 0.2                | 0.005               | 1                   |
| eurosat        | 5000      | 0.9               | 0.2                | 0.005               | 0.4                 |
| resisc45       | 5000      | 0.95              | 0.1                | 0.01                | 0.3                 |
| clevr          | 5000      | 0.9               | 0.2                | 0.005               | 1                   |
| ucf101         | 5000      | 0.9               | 0.1                | 0.01                | 0.3                 |
| imagenet       | 500       | 0.9               | 0.2                | 0.005               | 0.3                 |


<hr />

## 2. Synthetic Datasets

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