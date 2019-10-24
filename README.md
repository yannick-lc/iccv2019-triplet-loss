# Modeling Inter and Intra-Class Relations in the Triplet Loss for Zero-Shot Learning

This code allows to reprodure the results published in the [ICCV'19 article](http://openaccess.thecvf.com/content_ICCV_2019/papers/Le_Cacheux_Modeling_Inter_and_Intra-Class_Relations_in_the_Triplet_Loss_for_ICCV_2019_paper.pdf) and its [supplementary material](http://openaccess.thecvf.com/content_ICCV_2019/supplemental/Le_Cacheux_Modeling_Inter_and_ICCV_2019_supplemental.pdf).

All the material is contained in a unique jupyter notebook that you run with:

```
jupyter notebook Results.ipynb
```

Dependencies are:
*    python 2.7.9 
*    torch 1.1.0
*    sklearn 0.19.2
*    numpy 1.14.3
*    pandas 0.23.0
*    matplotlib 2.2.2
*    scipy 1.0.0
*    jupyter

## Get the CUB dataset
To visualize the images, you need to download the [CUB 200 2011 dataset](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz). The corresponding [project page](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) gives the reference to cite if you use it in your work.

You need to decompress the dataset into *CUB_200_2011/* then change the path in *data/df.cvs* accordingly, e.g:

``` perl
perl -i.old -p -e  's#/scratch_global/yannick#'$PWD'#' data/df.csv
```

## Citation
Please cite the following article if you use this code in your work:

Y. Le Cacheux, H. Le Borgne and M. Crucianu. Modeling Inter and Intra-Class Relations in the Triplet Loss for Zero-Shot Learning. In *Proceedings of the IEEE International Conference on Computer Vision, ICCV*, Seoul, Korea, Oct. 27 - Nov. 2, 2019

```
@inproceedings{lecacheux2019zsl,
     title  = {Modeling Inter and Intra-Class Relations in the Triplet Loss for Zero-Shot Learning},
     author = {Le Cacheux, Yannick and Le Borgne, Herv{\'e} and Crucianu, Michel},
  booktitle = {the IEEE International Conference on Computer Vision (ICCV)},
      month = {October},
     series = {ICCV},
       year = {2019}
 }

```
