# CVPR24_FRCSyn_ADMIS
This is the official GitHub repository for our team's contribution (**ADMIS**) to 

# [2nd Edition FRCSyn: Face Recognition Challenge in the Era of Synthetic Data](https://frcsyn.github.io/CVPR2024.html)

![image](poster.jpeg)

* A summary paper will be published in the **[proceedings of the IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR) 2024](https://cvpr.thecvf.com/)**

### We use a latent diffusion model (LDM) based on [IDiff-Face](https://github.com/fdbtrs/IDiff-Face) to synthesize faces. The LDM is conditioned using identity embeddings as contexts, extracted from faces by a pretrained [ElasticFace](https://github.com/fdbtrs/ElasticFace) iResNet-101 model.
![image](pipeline.png)

* Codes for training the generative and recognition models will be released soon. 

* If you are interested in the data used in our competition, please contact **zzhizhou66@gmail.com**. We are currently collecting our experimental results and will soon make public the checkpoints of the recognition models trained with this dataset, along with their performance on common benchmarks such as LFW, CFP-FP, IJB-B, and more.
  
