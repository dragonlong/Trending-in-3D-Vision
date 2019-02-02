
## Trending in 3D Vision
Deep learning not only provides good optimization techniques on feaure engineering, but also gives us unimaginable possibility to combine our intutions into research that could explore this 3D world better. In the research field of 3D Vision, a lot of excellent research work is happening. It's not enough to simply divide them into stereo vision, multi-view, monucular based 3D, we actually want to sort out research topics that could both connect with and distinguish from 2D vision. People have done paper collections on 3D as [link1](https://github.com/flamato/3D-computer-vision-resources) [link2](https://github.com/imkaywu/awesome-3d-vision-list), but they don't usually give us a good bird view, or most-updated infos. Based on above reasons, we create this special collection by summerizing the general ideas behind some most recent papers in 3D vision.

### 1. Single Image 3D  & Unsupervised Monucular Video Depth
[[Zhang et al. NIPS 2018](http://genre.csail.mit.edu/papers/genre_nips.pdf)] Learning to Reconstruct Shapes from Unseen Classes [[Project](http://genre.csail.mit.edu/)] [[Code]()]

[[Wu et al. ECCV 2018](http://shapehd.csail.mit.edu/papers/shapehd_eccv.pdf)]  Learning Shape Priors for
Single-View 3D Completion and Reconstruction [[Project](http://shapehd.csail.mit.edu/)] [[Code]()]

[[Niu et al. CVPR 2018](https://kevinkaixu.net/papers/niu_cvpr18_im2struct.pdf)] Im2Struct: Recovering 3D Shape Structure from a Single RGB Image [[Code](https://github.com/chengjieniu/Im2Struct)]

[[Zou et al. CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zou_LayoutNet_Reconstructing_the_CVPR_2018_paper.pdf)] LayoutNet: Reconstructing the 3D Room Layout from a Single RGB Image  [[Code](https://github.com/zouchuhang/LayoutNet)] [[Video](https://www.youtube.com/watch?v=WDzYXRP6XDs&feature=youtu.be)]

-------------------------
[[Mahj. et al. CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Mahjourian_Unsupervised_Learning_of_CVPR_2018_paper.pdf)] Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints [[Project](https://sites.google.com/view/vid2depth)] [[Code](https://github.com/tensorflow/models/tree/master/research/vid2depth)]

[[Yang et al. CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_LEGO_Learning_Edge_CVPR_2018_paper.pdf)] LEGO: Learning Edge with Geometry all at Once by Watching Videos  [[Demo](https://www.youtube.com/watch?v=40-GAgdUwI0)] [[Code](https://github.com/zhenheny/LEGO)]

[[Zou et al. ECCV 2018](https://arxiv.org/abs/1809.01649)] DF-Net: Unsupervised Joint Learning of Depth and Flow using Cross-Task Consistency  [[Project](http://yuliang.vision/DF-Net/)] [[Code](https://github.com/vt-vl-lab/DF-Net)]

(`Note: more papers on 'Monucular Video Depth' in CVPR 2017, 2018, ECCV 2018, along with the Robust Vision Challenge`)




### 2. 3D Generative Model for Vision beyond Visiable
[[Eslami et al.](Science-Machine Learning)]Neural scene representation and rendering
[[Project](https://deepmind.com/blog/neural-scene-representation-and-rendering/)] Code[[tf-gqn](https://github.com/ogroth/tf-gqn)][[gqn-datasets](https://github.com/deepmind/gqn-datasets)]
[[Pytorch-qgn](https://github.com/iShohei220/torch-gqn)]'seems to run quite slow'

[[Tulsiani et al. ECCV 2018](https://arxiv.org/pdf/1807.10264.pdf)] Layer-structured 3D Scene Inference
via View Synthesis [[Project](https://shubhtuls.github.io/lsi/)] [[Code](https://github.com/google/layered-scene-inference)]

[[Rama. et al. ECCV 2018](https://arxiv.org/abs/1807.11010)] Sidekick Policy Learning for Active Visual Exploration  [[Project](http://vision.cs.utexas.edu/projects/sidekicks/)] [[Code](https://github.com/srama2512/sidekicks)]

[[Song et al. CVPR 2018](https://arxiv.org/abs/1712.04569)] Im2Pano3D: Extrapolating 360° Structure and Semantics Beyond the Field of View [[Project](http://im2pano3d.cs.princeton.edu/)] [[Code](https://github.com/shurans/im2pano3d/)]



### 3. Pose Estimation

#### 3.1 Scene Layout and Object Pose
[[Wang et al. ARXIV pre-print 2019](https://arxiv.org/abs/1901.04780)]DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion
[[Code](https://github.com/j96w/DenseFusion)]
[[Project](https://sites.google.com/view/densefusion)]
[[Video](https://www.youtube.com/watch?v=SsE5-FuK5jo)]

[[Zhao et al. ARXIV pre-print 2019](https://arxiv.org/abs/1812.01387)]Estimating 6D Pose From Localizing Designated Surface Keypoints
[[Code](https://github.com/sjtuytc/betapose)]

[[Huang et al. NIPS 2018](https://arxiv.org/pdf/1810.13049.pdf)] Cooperative Holistic Scene Understanding: Unifying3D Object, Layout, and Camera Pose Estimation [[Video](https://www.youtube.com/watch?v=kXCugGwnr68)]


[[Trem. et al. CoRL 2018](https://arxiv.org/pdf/1809.10790.pdf)] Deep Object Pose Estimation for Semantic Robotic
Grasping of Household Objects  [[Project](https://research.nvidia.com/publication/2018-09_Deep-Object-Pose)] [[Code](https://github.com/NVlabs/Deep_Object_Pose)]

[[Sundermeyer et al. ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Martin_Sundermeyer_Implicit_3D_Orientation_ECCV_2018_paper.pdf)]Implicit 3D Orientation Learning for
6D Object Detection from RGB Images(**Best paper Award**)
[[Code](https://github.com/DLR-RM/AugmentedAutoencoder)]
[[Supplement](https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-030-01231-1_43/MediaObjects/474211_1_En_43_MOESM1_ESM.pdf)]
[[Video](https://www.youtube.com/watch?v=jgb2eNNlPq4)]

[[Tulsiani et al. CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tulsiani_Factoring_Shape_Pose_CVPR_2018_paper.pdf)] Factoring Shape, Pose, and Layout from the 2D Image of a 3D Scene [[Project](https://shubhtuls.github.io/factored3d/)] [[Code](https://github.com/shubhtuls/factored3d)]

[[Tulsiani et al. CVPR 2018](https://arxiv.org/pdf/1801.03910.pdf)]Multi-view Consistency as Supervisory Signal  for Learning Shape and Pose Prediction
[[Project](https://shubhtuls.github.io/mvcSnP/)]
[[Code](https://github.com/shubhtuls/mvcSnP)]

[[Tekin et al. CVPR 2018](https://arxiv.org/pdf/1711.08848.pdf)] Real-Time Seamless Single Shot 6D Object Pose Prediction [[Code](https://github.com/Microsoft/singleshotpose)] [[Supp.](http://openaccess.thecvf.com/content_cvpr_2018/Supplemental/3117-supp.pdf)]

[[Li et. al. ECCV 2018](https://arxiv.org/abs/1804.00175)] DeepIM: Deep Iterative Matching for 6D Pose Estimation [[Code](https://github.com/liyi14/mx-DeepIM0)]

[[Qi et al. CVPR 2018](https://arxiv.org/pdf/1711.08488.pdf)] Frustum PointNets for 3D Object Detection from RGB-D Data [[Project](http://stanford.edu/~rqi/frustum-pointnets/)] [[Code](https://github.com/charlesq34/frustum-pointnets)]

#### 3.2 Body Pose
[[Guo et al. ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Michelle_Guo_Neural_Graph_Matching_ECCV_2018_paper.pdf)]  Neural Graph Matching Networks for Fewshot 3D Action Recognition (`Pose for action recognition`)

[[Groueix et al. ECCV 2018](https://arxiv.org/pdf/1806.05228.pdf)] 3D-CODED : 3D Correspondences by Deep Deformation  [[Project](http://imagine.enpc.fr/~groueixt/3D-CODED/)] [[Code](https://github.com/ThibaultGROUEIX/3D-CODED)]

[[Joo et al. CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Joo_Total_Capture_A_CVPR_2018_paper.pdf)] Total Capture: A 3D Deformation Model for Tracking Faces, Hands, and Bodies [[Project](http://www.cs.cmu.edu/~hanbyulj/totalcapture/)] [[Supp.](http://www.cs.cmu.edu/~hanbyulj/totalcapture/totalBody_camready_supp.pdf)]

[[Riza et al. CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Guler_DensePose_Dense_Human_CVPR_2018_paper.pdf)]
Dense Human Pose Estimation In The Wild [[Project](http://densepose.org/)] [[Code](https://github.com/facebookresearch/DensePose)]

[[Pavl. et al. CVPR 2018](https://arxiv.org/pdf/1805.04092.pdf)] Learning to Estimate 3D Human Pose and Shape from a Single Color Image [[Project](https://www.seas.upenn.edu/~pavlakos/projects/humanshape/)]

#### 3.3 Face Pose
[[Moniz et al. NIPS 2018](https://papers.nips.cc/paper/8181-unsupervised-depth-estimation-3d-face-rotation-and-replacement.pdf)] Unsupervised Depth Estimation,
3D Face Rotation and Replacement [[Code releasing soon](https://github.com/joelmoniz/DepthNets/)]

[[Feng et al. ECCV 2018](https://arxiv.org/pdf/1803.07835v1.pdf)] Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network [[Code](https://github.com/YadiraF/PRNet)] [[Video](https://www.youtube.com/watch?v=tXTgLSyIha8&feature=youtu.be)]

[[Genova et al. CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Genova_Unsupervised_Training_for_CVPR_2018_paper.pdf)] Unsupervised Training for 3D Morphable Model Regression  [[Project]()] [[Code](https://github.com/google/tf_mesh_renderer)]

[[Deng et al. CVPR 2018](https://arxiv.org/pdf/1712.04695.pdf)] UV-GAN: Adversarial Facial UV Map Completion for Pose-invariant Face Recognition

#### 3.4 Hand Pose
[[Tsoli et al. ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Aggeliki_Tsoli_Joint_3D_tracking_ECCV_2018_paper.pdf)] Joint 3D Tracking of a Deformable Object in Interaction with a Hand

[[Ge et al. CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ge_Hand_PointNet_3D_CVPR_2018_paper.pdf)] Hand PointNet: 3D Hand Pose Estimation using Point Sets  [[Project](https://sites.google.com/site/geliuhaontu/home/cvpr2018)] [[Code](https://sites.google.com/site/geliuhaontu/HandPointNet.zip?attredirects=0&d=1)] [[VIdeo](https://youtu.be/-eiZYOo8cWc)]

[[Baek et al. CVPR 2018](https://arxiv.org/abs/1805.04497)] Augmented skeleton space transfer for depth-based hand pose estimation

(`Note: more papers on 'Hand Pose Estimation' in CVPR 2018, ECCV 2018`)


### 4. Disentangle Representations in 3D
[[Yao et al. NIPS 2018](https://arxiv.org/pdf/1808.09351.pdf)] 3D-Aware Scene Manipulation via Inverse Graphics  [[Project](http://3dsdn.csail.mit.edu/)]

[[Smith et al. NIPS 2018](https://papers.nips.cc/paper/7883-multi-view-silhouette-and-depth-decomposition-for-high-resolution-3d-object-representation.pdf)] Multi-View Silhouette and Depth Decomposition for High Resolution 3D Object Representation  [[Project](https://sites.google.com/site/mvdnips2018)]

[[Zhu et al. NIPS 2018](https://papers.nips.cc/paper/7297-visual-object-networks-image-generation-with-disentangled-3d-representations.pdf)] Visual Object Networks: Image Generation with
Disentangled 3D Representation



### 5. Unsupervised Key Points Detection
[[Suwa. et al. NIPS 2018](https://arxiv.org/pdf/1807.03146.pdf)] Discovery of Latent 3D Keypoints via
End-to-end Geometric Reasoning  [[Project](https://keypointnet.github.io/)] [[Code](https://github.com/tensorflow/models/tree/master/research/keypointnet)]

[[Zhou et al. ECCV 2018](https://arxiv.org/pdf/1712.05765.pdf)] Unsupervised Domain Adaptation for 3D
Keypoint Estimation via View Consistency [[Code](https://github.com/xingyizhou/3DKeypoints-DA)]  [[Results](https://drive.google.com/file/d/1UtlL7moKtNoVGyqWGRn8_c_57dwiqlVm/view)]



### 6. Point Cloud(PCL) Processing
#### 6.1 Neural Network for 3D data
[[Tata. et al. CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0144.pdf)] Tangent Convolutions for Dense Prediction in 3D [[Code](https://github.com/tatarchm/tangent_conv)]

[[Su et al. CVPR 2018](https://arxiv.org/abs/1802.08275)] SPLATNet: Sparse Lattice Networks for Point Cloud Processing
[[Project](http://siyuanhuang.com/cooperative_parsing/main.html)] [[Code](https://github.com/NVlabs/splatnet)]

[[Qi et al. NIPS 2017](https://arxiv.org/abs/1706.02413)] PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space  [[Project](http://stanford.edu/~rqi/pointnet2/)] [[Code](https://github.com/charlesq34/pointnet2)]

[[Weiler et al. NIPS 2018](https://papers.nips.cc/paper/8239-3d-steerable-cnns-learning-rotationally-equivariant-features-in-volumetric-data.pdf)] 3D Steerable CNNs: Learning Rotationally
Equivariant Features in Volumetric Data  [[Project]()] [[Code](https://github.com/mariogeiger/se3cnn)]

[[Sung et al. NIPS 2018](https://papers.nips.cc/paper/7330-deep-functional-dictionaries-learning-consistent-semantic-structures-on-3d-models-from-functions.pdf)] Deep Functional Dictionaries: Learning Consistent Semantic Structures on 3D Models from Functions  [[Project]()] [[Code](https://github.com/mhsung/deep-functional-dictionaries)]


#### 6.2 3D Registration & Rendering
[[Nguy. et al. NIPS 2018](https://papers.nips.cc/paper/8014-rendernet-a-deep-convolutional-network-for-differentiable-rendering-from-3d-shapes.pdf)] RenderNet: A deep convolutional network for
differentiable rendering from 3D shapes

[[Kim et al. ECCV 2018](https://arxiv.org/pdf/1807.02587.pdf)] Fast and Accurate Point Cloud Registration using Trees of Gaussian Mixtures  [[Project](https://research.nvidia.com/publication/2018-09_HGMM-Registration)] [[Video](https://www.youtube.com/watch?v=Bczht9CspiY)]

[[Sung et al. Siggraph Asia 2017](https://arxiv.org/abs/1708.01841)] ComplementMe: Weakly-Supervised Component Suggestions for 3D Modeling  [[Project](https://mhsung.github.io/complement-me.html)] [[Code](https://github.com/mhsung/complement-me)]

[[Zhu et al. SIGGRAPH Asia 2018]()] SCORES: Shape Composition with Recursive Substructure Priors  [[Project](https://kevinkaixu.net/projects/scores.html)] [[Code](https://kevinkaixu.net/projects/scores.html#code)]

[[Este. et al. 3DV 2018](https://vision.in.tum.de/_media/spezial/bib/estellers2018.pdf)] Robust Fitting of Subdivision Surfaces for Smooth Shape Analysis  [[Project]()] [[Code](https://bitbucket.org/ginie/subdivision_surfaces_3dv2018/src/master/)]

[[Kato et al. CVPR 2018](https://arxiv.org/abs/1711.07566)] Neural 3D Mesh Renderer  [[Project](http://hiroharu-kato.com/projects_en/neural_renderer.html)] [[Code](https://github.com/hiroharu-kato/neural_renderer)]




### 7. SLAM today
#### 7.1 3D Reconstruction & SLAM
[[Shi et al. ECCV 2018](https://arxiv.org/pdf/1803.08407)] PlaneMatch: Patch Coplanarity Prediction for
Robust RGB-D Reconstruction  [[Project](http://www.yifeishi.net/planematch.html)] [[Code](https://github.com/yifeishi/PlaneMatch)]

[[Bloesch et al. CVPR 2018](https://arxiv.org/abs/1804.00874)] CodeSLAM — Learning a Compact, Optimisable Representation for Dense Visual SLAM [[Project](http://www.imperial.ac.uk/dyson-robotics-lab/projects/codeslam/)] [[Video](https://www.youtube.com/watch?v=PbSggzaZWAQ&t=1s)]

[[Berg. et al. ICRA 2018](https://arxiv.org/abs/1710.02081)] Online Photometric Calibration of Auto Exposure Video for Realtime Visual Odometry and SLAM [[Project](https://vision.in.tum.de/research/vslam/photometric-calibration)] [[Code](https://vision.in.tum.de/research/vslam/photometric-calibration)]

#### 7.2 Object-aware SLAM
[[Rünz et al. ISMAR 2018](https://arxiv.org/pdf/1804.09194.pdf)] MaskFusion: Real-Time Recognition, Tracking and Reconstruction of Multiple Moving Objects [[Project](http://visual.cs.ucl.ac.uk/pubs/maskfusion/index.html)] [[Code](https://github.com/martinruenz/maskfusion)] [[Demo](http://visual.cs.ucl.ac.uk/pubs/maskfusion/MaskFusion.webm)]

[[McCo. et al. 3DV 2018](https://www.doc.ic.ac.uk/~sleutene/publications/fusion_plusplus_3dv_camera_ready.pdf)] Fusion++: Volumetric Object-Level SLAM [[Video](https://www.youtube.com/watch?v=2luKNC03x4k&feature=youtu.be)]

[[Zhou et al. ECCV 2018](https://arxiv.org/pdf/1808.01900.pdf)] DeepTAM: Deep Tracking and Mapping  [[Project](https://lmb.informatik.uni-freiburg.de/people/zhouh/deeptam/)] [[Code](https://github.com/lmb-freiburg/deeptam)]

#### 7.3 3D Photography
[[Hedman et al. SIGGRAPH](http://visual.cs.ucl.ac.uk/pubs/instant3d/instant3d_siggraph_2018.pdf)] Instant 3D Photography  [[Project](http://visual.cs.ucl.ac.uk/pubs/instant3d/)] [[Code](http://visual.cs.ucl.ac.uk/pubs/instant3d/implementation_details.pdf)]

[[Chen et al. ECCV 2018](http://gychen.org/PS-FCN/)] PS-FCN: A Flexible Learning Framework for Photometric Stereo [[Project](http://gychen.org/PS-FCN/)] [[Code](https://github.com/guanyingc/PS-FCN)]

## Misc:
### Workshops
[Bridge to CVPR 3D workshop](https://bridgesto3d.github.io/#schedule)

### Dataset 2018
[[Trem. et al. CVPR 2018 Workshop](https://research.nvidia.com/publication/2018-06_Falling-Things)] Falling Things: A Synthetic Dataset for 3D Object Detection and Pose Estimation [[Dataset](https://drive.google.com/open?id=1y4h9T6D9rf6dAmsRwEtfzJdcghCnI_01)]

[[Sun et al. CVPR 2018](https://arxiv.org/pdf/1804.04610.pdf)] Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling [[Dataset](https://github.com/xingyuansun/pix3d)]

[[Muel. et al. CVPR 2018](https://arxiv.org/abs/1712.01057)] GANerated Hands Dataset  [[Dataset](http://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/GANeratedDataset.htm)]

SUMO challenge dataset [[Dataset](https://sumochallenge.org/)] [[Code](https://github.com/facebookresearch/sumo-challenge)]
### More resources
Amesome 3D machine learning collection [here](https://github.com/timzhang642/3D-Machine-Learning)
