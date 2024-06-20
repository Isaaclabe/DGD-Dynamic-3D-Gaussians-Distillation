# DGD: Dynamic 3D Gaussians Distillation

Isaac Labe, Noam Issachar, Itai Lang, Sagie Benaim<br>
| [Webpage](https://isaaclabe.github.io/DGD-Website/) | [Full Paper](https://arxiv.org/pdf/2405.19321) | [arXiv](https://arxiv.org/abs/2405.19321) |

## Abstract
We tackle the task of learning dynamic 3D semantic radiance fields given a single monocular video as input. Our learned semantic radiance field captures per-point semantics as well as color and geometric properties for a dynamic 3D scene, enabling the generation of novel views and their corresponding semantics. This enables the segmentation and tracking of a diverse set of 3D semantic entities, specified using a simple and intuitive interface that includes a user click or a text prompt. To this end, we present DGD, a unified 3D representation for both the appearance and semantics of a dynamic 3D scene, building upon the recently proposed dynamic 3D Gaussians representation. Our representation is optimized over time with both color and semantic information. Key to our method is the joint optimization of the appearance and semantic attributes, which jointly affect the geometric properties of the scene. We evaluate our approach in its ability to enable dense semantic 3D object tracking and demonstrate high-quality results that are fast to render, for a diverse set of scenes.

**The code will be released soon...**
## Pipeline

![Teaser image](assets/pipeline.png)


## Dataset

In our paper, we use:

- synthetic dataset from [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html).
- real-world dataset from [Hyper-NeRF](https://hypernerf.github.io/).

We organize the datasets as follows:

```shell
├── data
│   | D-NeRF 
│     ├── hook
│     ├── standup 
│     ├── ...
│   | HyperNeRF
│     ├── interp
│     ├── misc
│     ├── vrig
```

## Setup

### Environment

```shell
git clone https://github.com/Isaaclabe/DGD-Dynamic-3D-Gaussians-Distillation.git --recursive
cd DGD-Dynamic-3D-Gaussians-Distillation

conda create -n DGD_env python=3.7
conda activate DGD_env

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -q plyfile
pip install git+https://github.com/openai/CLIP.git
pip install timm
pip install -r requirements.txt
```

### Setup the submodules

To run the training and rendering code, you need to setup the rasterizer and the Lseg-CLIP model. This is done by using the following instruction,
```shell
# The following part setup the gaussian rasterizer module:

cd DGD-Dynamic-3D-Gaussians-Distillation/submodules/diff-gaussian-rasterization
python setup.py build_ext
mkdir DGD-Dynamic-3D-Gaussians-Distillation/diff_gaussian_rasterization
mv DGD-Dynamic-3D-Gaussians-Distillation/submodules/diff-gaussian-rasterization/build/lib.linux-x86_64-cpython-310/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so DGD-Dynamic-3D-Gaussians-Distillation/diff_gaussian_rasterization
mv DGD-Dynamic-3D-Gaussians-Distillation/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py DGD-Dynamic-3D-Gaussians-Distillation/diff_gaussian_rasterization

# The following part setup the simple knn module:

cd DGD-Dynamic-3D-Gaussians-Distillation/submodules/simple-knn
python setup_knn.py build_ext
mkdir DGD-Dynamic-3D-Gaussians-Distillation/simple_knn
mv DGD-Dynamic-3D-Gaussians-Distillation/submodules/simple-knn/build/lib.linux-x86_64-cpython-310/simple_knn/_C.cpython-310-x86_64-linux-gnu.so DGD-Dynamic-3D-Gaussians-Distillation/simple_knn

# The following part setup the Lseg module:

cd DGD-Dynamic-3D-Gaussians-Distillation/lseg_minimal
python setup.py build develop
rm -rf DGD-Dynamic-3D-Gaussians-Distillation/lseg_minimal/lseg
!mv DGD-Dynamic-3D-Gaussians-Distillation/lseg_minimal/build/lib/lseg/ DGD-Dynamic-3D-Gaussians-Distillation/lseg_minimal
rm -rf DGD-Dynamic-3D-Gaussians-Distillation/lseg_minimal/build
```

## Train

### Use the DINOv2 foundation model

To run the optimizer using the DINOv2 foundation model, simply use

```shell
python train.py -s path/to/your/dataset -m output/exp-name --fundation_model "DINOv2" --sementic_dimension 384
```

### Use the Lseg-CLIP foundation model

To run the optimizer using the Lseg-CLIP foundation model, you need to download a pre-trained model of the Lseg minimal network [here](https://huggingface.co/datasets/IsaacLabe/Lseg_minimal_model/tree/main). Then, you can use

```shell
python train.py -s path/to/your/dataset -m output/exp-name --Lseg_model_path path/to/your/Lseg-model --fundation_model "Lseg_CLIP" --sementic_dimension 512 --loss_reduce 10
```

## BibTex

```bibtex
@misc{labe2024dgd,
      title={DGD: Dynamic 3D Gaussians Distillation}, 
      author={Isaac Labe and Noam Issachar and Itai Lang and Sagie Benaim},
      year={2024},
      eprint={2405.19321},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

Our repo is developed based on [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [DFFs](https://github.com/pfnet-research/distilled-feature-fields) and [Deformable 3D Gaussians](https://ingra14m.github.io/Deformable-Gaussians/). Many thanks to the authors for opensoucing the codebase.
