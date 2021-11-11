Pulmonary Texture Classification via a Semi-Supervised Deep Learning Method



Diffuse lung disease (DLD) is a kind of heterogeneous disease characterized by extensive and frequent changes of both lungs. The normal lung tissue of the patient is replaced by scar tissue, which brings pain and inconvenience to the unfortunate patient. Because of the similar clinical manifestations, it is difficult to diagnose them accurately. Therefore, computer analysis of high resolution CT (HRCT) of suspected patients is a good way to alleviate this problem.

A semi-supervised method based on pseudo-label is used to classify high resolution lung CT images. This method uses the pre-trained network on labeled data sets to predict unlabeled data, in which the predictions are selected if the network have confidence for them and participate in the next round of training. Experiments show that for a fixed number of unlabeled data, the Macro-F1 scores are 0.946824, 0.953914 and 0.963753 when the labeled data are 60, 600 and 3000, respectively. Compared with the network who only trained by few labeled images, this method improves by up to 0.135362. When using only 3000 labeled data, the difference between using nearly 36000 images is only 0.004919, which is comparable to those trained by a large number of labeled data.

This paper also improves the associate learning method and mean teacher method, using low-order features to improve the network, so that their Macro-F1 scores can be increased by 0.034001 and 0.027037, respectively



Comparation of the method in the dataset. Results are shown in Macro-F1.

Labeled data	60	600	3000	All
Fully supervised	0.811462	0.903772	0.916293	0.968572
Pseudo label based method	0.946824	0.953914	0.963753	-
Improved mean teacher	0.921264	0.949054	0.958234	-
Mean teacher[13] 0.894227	0.942917	0.953046	-
Improved associate learning	0.910867	0.948779	0.955216	-
Associate learning[14] 0.876863	0.942194	0.949838	-
