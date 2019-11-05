# Generating Personalized Recipes from Historical User Preferences
This is our PyTorch implementation for the paper:

*[Generating Personalized Recipes from Historical User Preferences](https://arxiv.org/pdf/1909.00105.pdf), EMNLP 2019*

The code is tested on a Linux server (with NVIDIA GeForce Titan X Pascal / NVIDIA GeForce GTX 1080 Ti) with PyTorch 1.1.0 and Python 3.6.

# Requirements
* Python 3
* Pytorch v1.0+

# Data
Backing data can be found [on Kaggle](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions).

# Running Models
To train a model, see the [`recipe_gen/models/<model>/train.py`](https://github.com/majumderb/recipe-personalization/blob/master/recipe_gen/models/baseline/train.py) file for that particular model (Baseline `train.py` linked). Likewise, run the `test.py` in the folder with arguments as listed to evaluate.

# Citation
If you find this repository useful for your research, please cite our paper:
```
@inproceedings{majumder2019emnlp,
  title={Generating Personalized Recipes from Historical User Preferences},
  author={Majumder, Bodhisattwa Prasad* and Li, Shuyang* and Ni, Jianmo and McAuley, Julian},
  booktitle={EMNLP},
  year={2019}
}
```
