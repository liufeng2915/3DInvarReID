# **Learning Clothing and Pose Invariant 3D Shape Representation for Long-Term Person Re-Identification**

International Conference on Computer Vision (ICCV 2023). [[Arxiv](https://arxiv.org/abs/2308.10658), [PDF](http://cvlab.cse.msu.edu/pdfs/Liu_Kim_Gu_Jain_Liu_ICCV2023.pdf), [Project](http://cvlab.cse.msu.edu/project-reid3dinvar.html)]

**[Feng Liu](https://liufeng2915.github.io/), [Minchul Kim](https://mckim.dev/), [Ziang Gu](https://scholar.google.com/citations?user=8tOJ80IAAAAJ&hl=en), [Anil Jain](https://www.cse.msu.edu/~jain/),  [Xiaoming Liu](http://www.cse.msu.edu/~liuxm/index2.html)**

Long-Term Person Re-Identification (LT-ReID) has become increasingly crucial in computer vision and biometrics. In this work, we aim to extend LT-ReID beyond pedestrian recognition to include a wider range of real-world human activities while still accounting for cloth-changing scenarios over large time gaps. This setting poses additional challenges due to the geometric misalignment and appearance ambiguity caused by the diversity of human pose and clothing. To address these challenges, we propose a new approach 3DInvarReID for (i) disentangling identity from non-identity components (pose, clothing shape, and texture) of 3D clothed humans, and (ii) reconstructing accurate 3D clothed body shapes and learning discriminative features of naked body shapes for person ReID in a joint manner. To better evaluate our study of LT-ReID, we collect a real-world dataset called CCDA, which contains a wide variety of human activities and clothing changes. Experimentally, we show the superior performance of our approach for person ReID.


## Prerequisites

This code is developed with

* Python 3.7
* Pytorch 1.8
* Cuda 11.1 

## Stage1: Pre-train the Joint Two-Layer Implicit Model

Please refer to [pretrain/README](pretrain/README.md) for the details.


## Stage2: Person Re-Identification
Please refer to [matching/README](matching/README.md) for the details.


## Citation

```bash
@inproceedings{liu2023learning,
  title={Learning clothing and pose invariant 3d shape representation for long-term person re-identification},
  author={Liu, Feng and Kim, Minchul and Gu, ZiAng and Jain, Anil and Liu, Xiaoming},
  booktitle={ICCV},
  year={2023}
}
```

## Acknowledgments

Here are some great resources we benefit from:

* [gDNA](https://github.com/xuchen-ethz/gdna) 3D clothed human model modeling.
* [Simple-CCReID](https://github.com/guxinqian/Simple-CCReID) person re-identification.

## License

[MIT License](LICENSE)

## Contact

For questions feel free to post here or drop an email to - liufeng6@msu.edu