Alibaba Cloud German AI Challenge 2018

AI For Earth Observation

1. Description
The task here is to perform Local Climate Zones (LCZ) classification in cities over the globe. LCZ is a generic, climate-based classification of the morphology of urban areas and natural landscapes [2] (Please see the section “Local Climate Zone” for more details of the classes.). Because of the rapid urbanization in the last decades, LCZ classification has attracted great attention in the remote sensing community in recent years. It has various potential use cases, such as demographic study in informal settlement etc.
The input data of the contest is co-registered Sentinel-1 and Sentinel-2 image patches over 42 different cities in different geographic and culture regions. Hence, the dataset is known in the remote sensing community as “LCZ42”.
The contest aims to promote innovation in classification algorithms, especially concerning the transferability of the proposed algorithm to different cultural regions, SAR and optical data fusion strategies, and big data processing techniques. The ranking is based on quantitative accuracy parameters computed with respect to test samples from cities unseen during training.
2. Local Climate Zone
LCZ is a generic, climate-based typology of urban and natural landscapes, which delivers information on basic physical properties of an area that can be used by land use planners or climate modelers [2]. LCZ can be used as a first order discretization of urban areas (see World Urban Database and Access Portal Tools initiative WUDAPT, http://www.wudapt.org). The definition of the LCZ classes is as follows [3].
3. The Data: LCZ42


The dataset is provided in HDF5 format. It contains co-registered image patches from Sentinel-1 and Sentinel-2, and the corresponding ground truth LCZ labels from 42 different cities. The dataset contains the following files:

N and M are the number of samples in the training and test sets. The class label is one-hot encoded, i.e. a 17-dimensional vector with only one element being 1, and the rest being 0. The image patches from Sentinel-1 and Sentinel-2 are co-registered and have identical spatial dimension (32 * 32 pixel, 10m x 10m pixel size). There are 8 bands in Sentinel-1 patches , whereas 10 in Sentinel-2 patches.

Sentinel-1 data bands

Sentinel-2 data bands


Details about the Sentinel-2 bands can be found at: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/overview

Training & Validation Set


There are two datasets available: training and validation. Note that you can also use the validation set for complementing the training set. There is no absolute requirement of how this dataset should be used.


Due to the large file size, we split the training set into several files. After downloading them, you can extract ‘traininig.h5’ by:


…
cat training.h5.z01 training.h5.z02 training.h5.z03 training.h5.z04 training.h5.zip > training.full.zip
unzip training.full.zip
…

Example to read the data

The following is a python example to read the dataset:

```
import h5py
import numpy as np
fid = h5py.File('training.h5', 'r')
# Loading sentinel-1 data patches
s1 = np.array(fid['sen1'])
# Loading sentinel-2 data patches
s2 = np.array(fid['sen2'])
# Loading labels
labels = np.array(fid['label'])
```

Example to Train A Model

You can find an ipython notebook example to train a simple model here:

http://tianchi-tum.oss-eu-central-1.aliyuncs.com/analysis.ipynb

4. References

[1] United Nations, World Urbanization Prospects: 2014 Revision. United Nation, 2014.

[2] B. Bechtel et al., “Mapping Local Climate Zones for a Worldwide Database of the Form and Function of Cities,” ISPRS Int. J. Geo-Inf., vol. 4, no. 1, pp. 199–219, Feb. 2015.

[3] I. D. Stewart and T. R. Oke, “Local climate zones for urban temperature studies,” Bull. Am. Meteorol. Soc., vol. 93, no. 12, pp. 1879–1900, 2012.

5. Submission

The contestants should submit the result of the predicted labels in CSV format with each row containing the one-hot encoded class label. And the order of the rows should correspond to the order of the test set.

Example:


0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
…

6. Evaluation Criteria


The test set is divided into two sets A and B. The leader board will initially show the ranking and score on the set A to help players adjust the model. And in the last two days we will switch to set B for the ranking. The final result on the test set B shall prevail.

The overall accuracy will be used as the evaluation criteria: Accuracy = m/n, where n is the number of samples in the test set and m is the number of correctly labelled samples.
