# Cons-Def

# Consensus Adversarial Defense Method Based on Augmented Examples

Anhui Normal University

School of Computer and Information, Wuhu, Anhui, China

241002

Xintao Ding (xintaoding@163.com)

Adversarial defense, consensus defense, python, mnist, Cifar10, ImageNet
This work is run for Consensus defense against FGSM, CW, PGD, and DeepFool attacks on MNIST, CIFAR-10, and ImageNet.
Both white-attacks and black-attacks are implemented in this repository.
Because CW attacks are implemented on encapsulated pb model, there is a code model_meta2pb.py to transform tensorflow model to pb model.
There is also a stream file maker and fetch code cifar10_create_tf_record.py
The repository is run under tensorflow-gpu-1.12.0, Python3.5.

This work is accepted in IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS for publication:

@article{ding2022consensus,
  title={Consensus Adversarial Defense Method Based on Augmented Examples},
  author={Ding, Xintao and Cheng, Yongqiang and Luo, Yonglong and Li, Qingde and Gope, Prosanta},
  journal={IEEE Transactions on Industrial Informatics},
  year={2022},
  publisher={IEEE}
}

Thanks the authors of CleverHans, this demo is implemented based on some library functions from CleverHans.

Thanks the git user panjq, we borrowed Inecption model on ImageNet from his site.
