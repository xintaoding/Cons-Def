# Cons-Def
Adversarial defense, consensus defense, python, mnist, Cifar10, ImageNet
This work is run for Consensus defense against FGSM, CW, PGD, and DeepFool attacks on MNIST, CIFAR-10, and ImageNet.
Both white-attacks and black-attacks are implemented in this repository.
Because CW attacks are implemented on encapsulated pb model, there is a code model_meta2pb.py to transform tensorflow model to pb model.
There is also a stream file maker and fetch code cifar10_create_tf_record.py
The repository is run under tensorflow-gpu-1.12.0, Python3.5.

Thanks the git user panjq (pan_jinquan@163.com), we borrowed some training codes of Inecption model on ImageNet.

This work is accepted in IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS for publication:
Xintao Ding, Yongqiang Cheng, Yonglong Luo, Qingde Li, and Prosanta Gope, Consensus Adversarial Defense Method Based on Augmented Examples, IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS.
