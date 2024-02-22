# Spatio-Temporal Approximation: A Training-Free SNN Conversion for Transformers

## Requirements
torch==1.9.0+cu111 \
clip==1.0

## Conversion Experiments on CIFAR-10
#### zeroshot experiment
```
python main.py --sample_data cifar10 
```
#### standard experiment using finetuned model
```
python main.py --sample_data cifar10 --load_vcm --vcm_path finetuned_model_path
```

## Conversion Experiments on CIFAR-10.1 & CIFAR-10.2
#### zeroshot experiment
```
python main.py --sample_data cifar10 --test_datas cifar101 cifar102
```

Currently, our experiments are done on GPUs based on high-precision floating-point computation, thus the cost of inference have not yet been optimized. Make sure you have 60G of memory available when you set batch_size=50. In future work, when this pipeline is succuessfully deployed on low-precion hardware platforms, memory and runtime will be effectively saved.
