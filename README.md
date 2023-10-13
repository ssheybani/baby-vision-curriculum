# Curriculum Learning With Infant Egocentric Videos
### Saber Sheybani, Himanshu Hansaria, Justin Newell Wood, Linda B. Smith, Zoran Tiganj
### Neurips 2023 Spotlight
#### Links: (Preprint), (Video)

images (from assets)
<div style="text-align: center;"><img src="assets/fig1.png" height="250px" ></div>

## Abstract:

Infants possess a remarkable ability to rapidly learn and process visual inputs. As an infant's mobility increases, so does the variety and dynamics of their visual inputs. Is this change in the properties of the visual inputs beneficial or even critical for the proper development of the visual system? To address this question, we used video recordings from infants wearing head-mounted cameras to pre-train state-of-the-art self-supervised video autoencoders. Critically, we separated the infant data by age group and evaluated the importance of training with a curriculum aligned with developmental order. We found that data from the youngest age group were necessary to kick-start learning. These results highlight the importance of slow visual inputs for the development of visual intelligence and provide a foundation for reverse engineering the learning mechanisms in newborn brains using image computable models from artificial intelligence.


## Code base Organization
```
baby-vision-curriculum
└── pretraining: python code used for pretraining the models with various objectives and architectures
│   ├── generative
│   |   └── pretrain_videomae_v3.1.py
│   ├── predictive
│   |   └── pretrain_vjepa_v1.1.py
│   └── contrastive
└── └── pretrain_simclr_v1.py
│
└── benchmarks: python code used for benchmarking any checkpoint on the tasks
│   ├── compute_embeddings_videomae.py
│   ├── compute_embeddings_jepa.py
│   ├── compute_embeddings_simclr.py
└── slurmscripts: linux bash code used for submitting jobs that train and evaluate models.
└── notebooks: Jupyter notebook files used for creating the figures in the manuscript.
```


## Dependencies:
pytorch, torchvision
huggingface transformers (for VideoMAE)
tqdm

VideoMAE models need to be pretrained on multiple GPUs as they take up substantial GPU memory. We use PyTorch DistributedDataParallel for that.

### Citation

```
  @article{sheybani23curriculum,
    title={Curriculum Learning with Infant Egocentric Videos},
    author={Sheybani, Saber and Hansaria, Himanshu and Wood, Justin and Smith, Linda and Tiganj, Zoran},
    journal={Advances in Neural Information Processing Systems},
    year={2023}
  }
```
