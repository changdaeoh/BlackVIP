# BlackVIP: Black-Box Visual Prompting for Robust Transfer Learning
> We provide the official PyTorch Implementation of '[BlackVIP: Black-Box Visual Prompting for Robust Transfer Learning](https://arxiv.org/abs/2303.14773)' (CVPR 2023) <br/> 
>>[Changdae Oh](https://changdaeoh.github.io/), [Hyeji Hwang](https://github.com/hyezzz), [Hee-young Lee](https://github.com/hy18284), [YongTaek Lim](https://github.com/teang1995), [Geunyoung Jung](), [Jiyoung Jung](https://scholar.google.co.kr/citations?user=wc_MQkoAAAAJ&hl=en), [Hosik Choi](https://scholar.google.co.kr/citations?user=0pzb3WAAAAAJ&hl=en), and [Kyungwoo Song](https://scholar.google.com/citations?user=HWxRii4AAAAJ&hl=ko)

<br/>


## Abstract
<p align="center">
<img src="docs/fig1_illustration.png" alt= "" width="" height="250">
</p>

> With the surge of large-scale pre-trained models (PTMs), fine-tuning these models to numerous downstream tasks becomes a crucial problem. Consequently, parameter efficient transfer learning (PETL) of large models has grasped huge attention. While recent PETL methods showcase impressive performance, they rely on optimistic assumptions: 1) the entire parameter set of a PTM is available, and 2) a sufficiently large memory capacity for the fine-tuning is equipped. However, in most real-world applications, PTMs are served as a black-box API or proprietary software without explicit parameter accessibility. Besides, it is hard to meet a large memory requirement for modern PTMs. In this work, we propose black-box visual prompting (BlackVIP), which efficiently adapts the PTMs without knowledge about model architectures and parameters. BlackVIP has two components; 1) Coordinator and 2) simultaneous perturbation stochastic approximation with gradient correction (SPSA-GC). The Coordinator designs input-dependent image-shaped visual prompts, which improves few-shot adaptation and robustness on distribution/location shift. SPSA-GC efficiently estimates the gradient of a target model to update Coordinator. Extensive experiments on 16 datasets demonstrate that BlackVIP enables robust adaptation to diverse domains without accessing PTMs' parameters, with minimal memory requirements.

<br/>

## Research Highlights
<p align="center">
<img src="docs/blackvip_framework.png" alt= "" width="90%" height="90%">
</p>

* **Input-Dependent Dynamic Visual Prompting:** To our best knowledge, this is the first paper that explores the input-dependent visual prompting on black-box settings. For this, we devise `Coordinator`, which reparameterizes the prompt as an autoencoder to handle the input-dependent prompt with tiny parameters.
* **New Algorithm for Black-Box Optimization:** We propose a new zeroth-order optimization algorithm, `SPSA-GC`, that gives look-ahead corrections to the SPSA's estimated gradient resulting in boosted performance. 
* **End-to-End Black-Box Visual Prompting:** By equipping Coordinator and SPSA-GC, `BlackVIP` adapts the PTM to downstream tasks without parameter access and large memory capacity. 
* **Empirical Results:** We extensively validate BlackVIP on 16 datasets and demonstrate its effectiveness regarding _few-shot adaptability_ and _robustness on distribution/object-location shift_.

<br/>
<hr/>

## Coverage of this repository
### _Methods_
* `BlackVIP` (Ours)
* `BAR`
* `VP` (with our SPSA-GC)
* `VP`
* `Zero-Shot Inference`
### _Experiments_
* **main performance** (Tab. 2 and Tab. 3 of paper)
  * two synthetic datasets - [`Biased MNIST`, `Loc-MNIST`]
  * 14 transfer learning benchmarks - [`Caltech101`, `OxfordPets`, `StanfordCars`, `Flowers102`, `Food101`, `FGVCAircraft`, `SUN397`, `DTD`, `SVHN`, `EuroSAT`, `Resisc45`, `CLEVR`, `UCF101`, `ImageNet`]
* **ablation study** (Tab. 5 and Tab. 6 of paper)
  * varying backbone
  * varying coordinator weights, spsa vs. spsa-gc

<br/>

## Installation & Requriments
```
soon :)
```

<br/>

## Data preparation
```
soon :)
```

<br/>

## Run
```
soon :)
```

<br/>
<hr />

## Contact
For any questions, discussions, and proposals, please contact to `changdae.oh@uos.ac.kr` or `kyungwoo.song@gmail.com`

<br/>

## Citation
If you use our code in your research, please kindly consider citing:
```bibtex
@inproceedings{oh2023blackvip,
  title={BlackVIP: Black-Box Visual Prompting for Robust Transfer Learning},
  author={Oh, Changdae and Hwang, Hyeji and Lee, Hee-young and Lim, YongTaek, and Jung, Geunyoung, and Jung, Jiyoung, and Choi, Hosik, and Song, Kyungwoo},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

<br/>

## Acknowledgements
Our overall experimental pipeline is based on [CoOp, CoCoOp](https://github.com/KaiyangZhou/CoOp) repository. For baseline construction, we bollowed/refered the code from repositories of [VP](https://github.com/hjbahng/visual_prompting), [BAR](https://github.com/yunyuntsai/Black-box-Adversarial-Reprogramming), and [AR](https://github.com/savan77/Adversarial-Reprogramming). We appreciate the authors (Zhou et al., Bahng et al., Tsai et al.) and Savan for sharing their code.

