## Trending in 3D Vision
I first got fascinated by the beauty of 3D vision since 2015. After that, so many new and wonderful ideas, works have been brought into this field, and it seems so hard to catch up with this fast-evolving area today. This leads to the major motivation behind this paper reading list: to get a sense of current SOTA methods, and an overview of the research trending in the field of 3D vision, mainly with deep learning.

From this list, you may say, various applications, multiple modalities of data, powerful neural backbones are the major working horses, or the boom of neural radiance field and differentiable rendering inspire a lot of new methods and tasks, or you want to point out that self-supervision, data-efficient learning are the critical keys. Different people may have different opinions, but this list is about existing possibilities in 3D vision, to which you may say 'wow, this is even possible', or 'aha, I never imagined such a method'.

Note that this repo started as a self-collected paper list based on my own appetite, which may reflect some bias. Some may not be precisely categorized, for which you can raise an issue, or send a pull request.

### [1. SLAM with Deep Learning](#content)
  [[Chen et al. (ARXIV '22)]( https://arxiv.org/pdf/2203.01087.pdf )] Vision-based Large-scale 3D Semantic Mapping for Autonomous Driving Applications

  [[Avraham et al. (ARXIV '22)]( https://arxiv.org/abs/2206.01916 )]  Nerfels: Renderable Neural Codes for Improved Camera Pose Estimation

  [[Hughe et al. (ARXIV '22)]( https://arxiv.org/pdf/2201.13360.pdf )]  Hydra: A Real-time Spatial Perception Engine for 3D Scene Graph Construction and Optimization
  [[Video]( https://youtu.be/qZg2lSeTuvM )]

  [[Zhu at al. (**CVPR '22**)]( https://arxiv.org/pdf/2112.12130.pdf )] NICE-SLAM: Neural Implicit Scalable Encoding for SLAM
  [[Project]( https://pengsongyou.github.io/nice-slam )]
  [[Code]( https://github.com/cvg/nice-slam )]
  [[Video](https://youtu.be/V5hYTz5os0M )]

  [[Teed et al. (NeurIPS '21)]( https://arxiv.org/pdf/2108.10869.pdf )] DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras
  [[Code]( https://github.com/princeton-vl/DROID-SLAM )]
  [[Video]( https://www.youtube.com/watch?v=GG78CSlSHSA )]

  [[Yang et al. (3DV '21)]( https://arxiv.org/abs/2111.07418 )] TANDEM: Tracking and Dense Mapping in Real-time using Deep Multi-view Stereo
  [[Project]( https://vision.in.tum.de/research/vslam/tandem )]
  [[Code]( https://github.com/tum-vision/tandem )]
  [[Video](https://youtu.be/L4C8Q6Gvl1w)]

  [[Lin et al. (ARXIV '21)]( https://arxiv.org/pdf/2109.07982.pdf )] R3LIVE: A Robust, Real-time, RGB-colored, LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package
  [[Code]( https://github.com/hku-mars/r3live )]
  [[Video]( https://youtu.be/j5fT8NE5fdg )]

  [[Duzceker et al. (CVPR '21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Duzceker_DeepVideoMVS_Multi-View_Stereo_on_Video_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.pdf)] DeepVideoMVS: Multi-View Stereo on Video with Recurrent Spatio-Temporal Fusion
  [[Code]( https://github.com/ardaduz/deep-video-mvs )]
  [[Video]( https://www.youtube.com/watch?v=ikpotjxwcp4 )]

  [[Teed at al. (CVPR '21)]( https://arxiv.org/pdf/2103.12032.pdf )] Tangent Space Backpropagation for 3D Transformation Groups
  [[Code]( https://github.com/princeton-vl/lietorch )]

  [[Sun et al. (CVPR '21)]( https://arxiv.org/pdf/2104.00681.pdf )]  NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video
  [[Project]( http://zju3dv.github.io/neuralrecon/ )]
  [[Code]( https://github.com/zju3dv/NeuralRecon/ )]

  [[Murthy J. et al. (ICRA '20)](https://arxiv.org/pdf/1910.10672.pdf)] ∇SLAM: Automagically differentiable SLAM
  [[Project]( https://github.com/gradslam/gradslam )]
  [[Code]( https://gradslam.github.io/ )]

  [[Schops et al. (CVPR '19)]( https://openaccess.thecvf.com/content_CVPR_2019/papers/Schops_BAD_SLAM_Bundle_Adjusted_Direct_RGB-D_SLAM_CVPR_2019_paper.pdf )] BAD SLAM: Bundle Adjusted Direct RGB-D SLAM
  [[Project]( https://www.eth3d.net/ )]
  [[Code]( https://github.com/ETH3D/badslam )]

### [2. Dynamic Human, Animals and Objects Reconstruction](#content)
#### Human avatars
  [[Su et al. (ARXIV '22)]( https://arxiv.org/pdf/2205.01666.pdf )] DANBO: Disentangled Articulated Neural Body
  Representations via Graph Neural Networks
  [[Project]( https://lemonatsu.github.io/danbo/ )]

  [[Jiang et al. (**CVPR '22**)]( https://arxiv.org/pdf/2201.12792.pdf )] SelfRecon: Self Reconstruction Your Digital Avatar from Monocular Video
  [[Project]( https://jby1993.github.io/SelfRecon/ )]
  [[Code]( https://github.com/jby1993/SelfReconCode )]

  [[Weng et al. (**CVPR '22**)]( https://arxiv.org/abs/2201.04127 )] HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video
  [[Project]( https://grail.cs.washington.edu/projects/humannerf/ )]
  [[Code]( https://github.com/chungyiweng/humannerf )]
  [[Video]( https://youtu.be/GM-RoZEymmw )]

  [[Yu et al. (CVPR '21)](https://arxiv.org/pdf/2105.01859.pdf )] Function4D: Real-time Human Volumetric Capture from Very Sparse Consumer RGBD Sensors
  [[Project]( http://www.liuyebin.com/Function4D/Function4D.html )]
  [[Data]( https://github.com/ytrock/THuman2.0-Dataset )]
  [[Video]( http://www.liuyebin.com/Function4D/assets/supp_video.mp4 )]

#### Animals capture
  [[Yang et al. (**CVPR '22**)]( https://banmo-www.github.io/banmo-2-14.pdf )] BANMo: Building Animatable 3D Neural Models from Many Casual Videos
  [[Project]( https://banmo-www.github.io/ )]
  [[Code]( https://github.com/facebookresearch/banmo )]
  [[Video]( https://youtu.be/1NUa-yvFGA0 )]

  [[Wu et al. (NeurIPS '21)]( https://arxiv.org/abs/2107.10844 )] DOVE: Learning Deformable 3D Objects by Watching Videos
  [[Project]( https://dove3d.github.io/ )]
  [[Video]( https://youtu.be/_FsADb0XmpY )]

#### Human-object interaction
  [[Jiang et al. (**CVPR '22**)]( https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_NeuralHOFusion_Neural_Volumetric_Rendering_Under_Human-Object_Interactions_CVPR_2022_paper.pdf)] NeuralHOFusion: Neural Volumetric Rendering under Human-object Interactions
  [[Project]( https://nowheretrix.github.io/neuralfusion/ )]
  [[Video]( https://youtu.be/Stvks4rZMF0 )]

  [[Hasson et al. (CVPR '21)]( https://arxiv.org/abs/2108.07044 )] Towards unconstrained joint hand-object reconstruction from RGB videos
  [[Project](https://hassony2.github.io/homan.html )]
  [[Code]( https://github.com/hassony2/homan )]

#### Scene-level 3D dynamics
  [[Grauman et al. (**CVPR '22**)]( https://arxiv.org/abs/2110.07058 )] Ego4D: Around the World in 3,000 Hours of Egocentric Video
  [[Project]( https://ego4d-data.org/ )]
  [[Code]( https://github.com/EGO4D )]

  [[Li et al. (**CVPR '22**)]( https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Neural_3D_Video_Synthesis_From_Multi-View_Video_CVPR_2022_paper.pdf )] Neural 3D Video Synthesis From Multi-View Video
  [[Project]( https://neural-3d-video.github.io/ )]
  [[Video]( https://neural-3d-video.github.io/resources/video.mp4 )]
  [[Data]( https://github.com/facebookresearch/Neural_3D_Video )]

  [[Zhang et al. (SIGGRAPH '21)]( https://arxiv.org/abs/2108.01166 )]  Consistent Depth of Moving Objects in Video
  [[Project]( https://dynamic-video-depth.github.io/ )]
  [[Code]( https://github.com/google/dynamic-video-depth )]
  [[Video]( https://dynamic-video-depth.github.io/#video )]

  [[Zeed et al. (CVPR '21)]( https://arxiv.org/abs/2012.00726 )] RAFT-3D: Scene Flow using Rigid-Motion Embeddings
  [[Code]( https://github.com/princeton-vl/RAFT-3D )]

  [[Lu et al. (CVPR '21)]( https://arxiv.org/pdf/2105.06993.pdf )] Omnimatte: Associating Objects and Their Effects in Video
  [[Project]( https://omnimatte.github.io/ )]
  [[Code]( https://omnimatte.github.io/#code )]
  [[Video]( https://omnimatte.github.io/#video )]

### [3. Self-Supervised 3D Representations Learning](#content)
[[Hasselgren et al. (ARXIV '22)]( https://arxiv.org/abs/2206.03380 )] Shape, Light & Material Decomposition from Images using Monte Carlo Rendering and Denoising

[[Gkioxari et al. (ARXIV '22)]( https://drive.google.com/file/d/1E6xSbUzuu6soAA-jkaGCFl97LZ8SVRvr/view )] Learning 3D Object Shape and Layout without 3D Supervision
[[Project]( https://gkioxari.github.io/usl/ )]
[[Video]( https://youtu.be/PKhGIiMuRJU )]

[[Boss et al. (ARXIV '22)]( https://arxiv.org/pdf/2205.15768.pdf)]  SAMURAI: Shape And Material from Unconstrained Real-world Arbitrary Image collections
[[Project]( https://markboss.me/publication/2022-samurai/ )]
[[Video]( https://youtu.be/LlYuGDjXp-8 )]

[[Wei et al. (SIGGRAPH '22)]( https://arxiv.org/pdf/2205.02961.pdf )] Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search
[[Project]( https://colin97.github.io/CoACD/ )]
[[Code]( https://github.com/SarahWeiii/CoACD )]
[[Video]( https://www.youtube.com/watch?v=r12O0z0723s )]

[[Vicini et al. (SIGGRAPH '22)]( http://rgl.s3.eu-central-1.amazonaws.com/media/papers/Vicini2022sdf_1.pdf )] Differentiable Signed Distance Function Rendering
[[Project]( http://rgl.epfl.ch/publications/Vicini2022SDF )]
[[Video]( http://rgl.s3.eu-central-1.amazonaws.com/media/papers/Vicini2022sdf.mp4 )]

[[Or-El et al. (**CVPR '22**)]( https://arxiv.org/abs/2112.11427 )] StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation
[[Project]( https://stylesdf.github.io/ )]
[[Code](https://github.com/royorel/StyleSDF)]
[[Demo]( https://colab.research.google.com/github/royorel/StyleSDF/blob/main/StyleSDF_demo.ipynb )]

[[Girdhar et al. (**CVPR '22**)]( https://arxiv.org/abs/2201.08377 )] Omnivore: A Single Model for Many Visual Modalities
[[Project]( https://facebookresearch.github.io/omnivore )]
[[Code](https://github.com/facebookresearch/omnivore )]



[[Noguchi et al. (**CVPR '22**)]( https://openaccess.thecvf.com/content/CVPR2022/papers/Noguchi_Watch_It_Move_Unsupervised_Discovery_of_3D_Joints_for_Re-Posing_CVPR_2022_paper.pdf )] Watch It Move: Unsupervised Discovery of 3D Joints for Re-Posing of Articulated Objects
[[Project]( https://nvlabs.github.io/watch-it-move/ )]
[[Code]( https://github.com/NVlabs/watch-it-move )]
[[Video]( https://youtu.be/oRnnuCVV89o )]

[[Gong et al. (**CVPR '22**)]( https://arxiv.org/pdf/2203.15625.pdf )] PoseTriplet: Co-evolving 3D Human Pose Estimation, Imitation, and Hallucination under Self-supervision
[[Code]( https://github.com/garfield-kh/posetriplet )]

[[Wu et al. (**CVPR '22**)]( https://arxiv.org/abs/2112.02306 )] Toward Practical Monocular Indoor Depth Estimation
[[Project]( https://distdepth.github.io/ )]
[[Code]( https://github.com/facebookresearch/DistDepth )]
[[Video]( https://youtu.be/s9JdoR1xbz8 )]
[[Data](https://drive.google.com/file/d/1KfDFyTg9-1w1oJB4oT-DUjKC6LG0enwb/view?usp=sharing)]


[[Wei et al. (**CVPR '22**)](https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_Self-Supervised_Neural_Articulated_Shape_and_Appearance_Models_CVPR_2022_paper.pdf )] Self-supervised Neural Articulated Shape and Appearance Models
[[Project]( https://weify627.github.io/nasam/ )]
[[Video]( https://youtu.be/0YbhTxALi8M )]

[[Chan et al. (**CVPR '22**)](https://arxiv.org/pdf/2112.07945.pdf)] EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks
[[Project]( https://nvlabs.github.io/eg3d/ )]
[[Code]( https://github.com/NVlabs/eg3d )]
[[Video]( https://www.youtube.com/watch?v=cXxEwI7QbKg )]

  [[Rombach et al. (ICCV '21)]( https://arxiv.org/pdf/2104.07652.pdf)]
    Geometry-Free View Synthesis: Transformers and no 3D Priors
   Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations
  [[Project]( https://compvis.github.io/geometry-free-view-synthesis/ )]
  [[Code]( https://github.com/CompVis/geometry-free-view-synthesis )]
  [[Video]( https://github.com/CompVis/geometry-free-view-synthesis/blob/master/assets/acid_long.mp4 )]

  [[Harley et al. (CVPR '21)]( https://openaccess.thecvf.com/content/CVPR2021/papers/Harley_Track_Check_Repeat_An_EM_Approach_to_Unsupervised_Tracking_CVPR_2021_paper.pdf )] Track, Check, Repeat: An EM Approach to Unsupervised Tracking
  [[Project]( http://www.cs.cmu.edu/~aharley/em_cvpr21/ )]
  [[Code]( https://github.com/aharley/track_check_repeat )]

  [[Watson et al. (CVPR '21)]( https://openaccess.thecvf.com/content/CVPR2021/papers/Watson_The_Temporal_Opportunist_Self-Supervised_Multi-Frame_Monocular_Depth_CVPR_2021_paper.pdf )] The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth
  [[Code]( https://github.com/nianticlabs/manydepth )]
  [[Video]( https://storage.googleapis.com/niantic-lon-static/research/manydepth/manydepth_cvpr_cc.mp4 )]

  [[Nicolet et al. (SIGGRAPH Asia '21)]( http://rgl.s3.eu-central-1.amazonaws.com/media/papers/Nicolet2021Large.pdf )] Large Steps in Inverse Rendering of Geometry
  [[Project]( https://rgl.epfl.ch/publications/Nicolet2021Large )]
  [[Code]( https://github.com/rgl-epfl/cholespy )]
  [[Video]( https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Nicolet2021Large_1.mp4 )]

  [[Wu et al. (CVPR '20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Unsupervised_Learning_of_Probably_Symmetric_Deformable_3D_Objects_From_Images_CVPR_2020_paper.pdf )]   Unsupervised Learning of Probably Symmetric Deformable 3D Objects From Images in the Wild
  [[Project]( https://elliottwu.com/projects/20_unsup3d/ )]
  [[Code]( https://github.com/elliottwu/unsup3d )]
  [[Video]( https://youtu.be/p3KB3eIQw24 )]


### [4. Breakthroughs in Deep Implicit Functions](#content)

#### Topology-aware
  [[Palafox et al. (**CVPR '22**)]( https://arxiv.org/pdf/2201.08141.pdf )] SPAMs: Structured Implicit Parametric Models
  [[Project]( https://pablopalafox.github.io/spams/ )]
  [[Video]( https://www.youtube.com/watch?v=ChdjHNGgrzI )]

  [[Park et al. (SIGGRAPH Asia '21)]( https://arxiv.org/pdf/2106.13228.pdf )] A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields
  [[Project]( https://hypernerf.github.io/ )]
  [[Code]( https://github.com/google/hypernerf )]
  [[Video]( https://youtu.be/qzgdE_ghkaI )]

#### Additional priors
  [[Guo et al. (**CVPR '22**)]( https://arxiv.org/abs/2205.02836 )] Neural 3D Scene Reconstruction with the Manhattan-world Assumption
  [[Project]( https://zju3dv.github.io/manhattan_sdf/)]
  [[Code]( https://github.com/zju3dv/manhattan_sdf )]
  [[Video]( https://www.youtube.com/watch?v=oEE7mK0YQtc )]

#### Faster, memory-efficient
[[Chen et al. (ARXIV '22)]( https://arxiv.org/pdf/2203.09517.pdf )] TensoRF: Tensorial Radiance Fields
[[Project]( https://apchenstu.github.io/TensoRF/ )]
[[Code]( https://github.com/apchenstu/TensoRF )]

[[Müller et al. (ARXIV '22)]( https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf )] Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
[[Project]( https://nvlabs.github.io/instant-ngp/)]
[[Code]( https://nvlabs.github.io/instant-ngp/ )]
[[Video]( https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.mp4 )]

[[Schwarz et al. (ARXIV '22)]( https://arxiv.org/pdf/2206.07695.pdf )] VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids

[[Takikawa et al. (Siggraph '22)]( https://drive.google.com/file/d/1GTFPwQ3oe0etRJKP35oyRhHpsydTE_AR/view )] Variable Bitrate Neural Fields
[[Project]( https://nv-tlabs.github.io/vqad/ )]
[[Video]( https://www.youtube.com/watch?v=Lh0CoTRNFBA )]

[[Sun et al. (**CVPR '22**)]( https://arxiv.org/pdf/2111.11215.pdf )] Direct Voxel Grid Optimization Super-fast Convergence for Radiance Fields Reconstruction
[[Project]( https://sunset1995.github.io/dvgo/ )]
[[Code]( https://github.com/sunset1995/DirectVoxGO )]
[[Video]( https://youtu.be/gLmujfjRVGw )]
[[DVGOv2](https://arxiv.org/abs/2206.05085)]

[[Yu et al. (**CVPR '22**)]( https://arxiv.org/abs/2112.05131 )] Plenoxels: Radiance Fields without Neural Networks
[[Project]( https://alexyu.net/plenoxels/ )]
[[Code]( https://github.com/sxyu/svox2 )]
[[Video]( https://www.youtube.com/watch?v=KCDd7UFO1d0&t=6s )]

[[Xu et al. (**CVPR '22**)]( https://arxiv.org/abs/2201.08845 )] Point-NeRF: Point-based Neural Radiance Fields
[[Project]( https://xharlie.github.io/projects/project_sites/pointnerf/index.html )]
[[Code]( https://github.com/Xharlie/pointnerf )]

[[Deng et al. (**CVPR '22**)]( https://arxiv.org/abs/2107.02791 )] Depth-Supervised NeRF: Fewer Views and Faster Training for Free
[[Project]( https://www.cs.cmu.edu/~dsnerf/ )]
[[Code](https://github.com/dunbar12138/DSNeRF )]
[[Video]( https://youtu.be/84LFxCo7ogk )]

[[Takikawa et al. (CVPR '21)]( https://arxiv.org/pdf/2101.10994.pdf )] Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Shapes
[[Project]( https://nv-tlabs.github.io/nglod/ )]
[[Code]( https://github.com/nv-tlabs/nglod )]
[[Video]( https://youtu.be/0cJZn_hV2Ms )]

[[Garbin et al. (CVPR '21)]( https://arxiv.org/abs/2103.10380 )] FastNeRF: High-Fidelity Neural Rendering at 200FPS
[[Project]( https://microsoft.github.io/FastNeRF/ )]
[[Video]( https://youtu.be/JS5H-Usiphg )]

#### Dynamic
[[Fang et al. (ARXIV '22)]( https://arxiv.org/abs/2205.15285 )] TiNeuVox: Fast Dynamic Radiance Fields with Time-Aware Neural Voxels
[[Project]( https://jaminfong.cn/tineuvox/ )]
[[Code]( https://github.com/hustvl/TiNeuVox )]
[[Video]( https://youtu.be/sROLfK_VkCk )]

[[Wang et al. (**CVPR '22**)]( https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Fourier_PlenOctrees_for_Dynamic_Radiance_Field_Rendering_in_Real-Time_CVPR_2022_paper.pdf )] Fourier PlenOctrees for Dynamic Radiance Field Rendering in Real-time
[[Project]( https://aoliao12138.github.io/FPO/ )]
[[Video]( https://youtu.be/XZSuQQOY6Ls )]

[[Gao et al. (ICCV '21)]( https://arxiv.org/pdf/2105.06468.pdf )] Dynamic View Synthesis from Dynamic Monocular Video
[[Project]( https://free-view-video.github.io/ )]
[[Code](https://github.com/gaochen315/DynamicNeRF )]
[[Video]( https://youtu.be/j8CUzIR0f8M )]

[[Li et al. (CVPR '21)]( https://arxiv.org/abs/2011.13084 )] Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes
[[Project]( https://www.cs.cornell.edu/~zl548/NSFF// )]
[[Code](https://github.com/zhengqili/Neural-Scene-Flow-Fields )]
[[Video](https://www.cs.cornell.edu/~zl548/NSFF//overview.mp4 )]

#### Editable
[[Zhang et al. (ARXIV '22)]( https://arxiv.org/abs/2206.06360 )] ARF: Artistic Radiance Fields
[[Project]( https://www.cs.cornell.edu/projects/arf/ )]
[[Code & Data]( https://github.com/Kai-46/ARF-svox2 )]

[[Kobayashi et al. (ARXIV '22)]( https://arxiv.org/pdf/2205.15585.pdf )] Decomposing NeRF for Editing via Feature Field Distillation
[[Project]( https://pfnet-research.github.io/distilled-feature-fields/ )]

[[Benaim et al. (ARXIV '22)]( https://arxiv.org/pdf/2206.02776.pdf )]  Volumetric Disentanglement for 3D Scene Manipulation
[[Project]( https://sagiebenaim.github.io/volumetric-disentanglement/ )]

[[Lazova et al. (**CVPR '22**)](https://arxiv.org/abs/2204.10850 )] Control-NeRF: Editable Feature Volumes for Scene Rendering and Manipulation

[[Yuan et al. (**CVPR '22**)]( https://arxiv.org/pdf/2205.04978.pdf )] NeRF-Editing: Geometry Editing of Neural Radiance Fields

#### Generalizable
[[Yu et al. (ARXIV '22)]( https://arxiv.org/pdf/2206.00665.pdf )] MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction
[[Project]( https://niujinshuchong.github.io/monosdf )]

[[Rebain et al. (**CVPR '22**)]( https://openaccess.thecvf.com/content/CVPR2022/papers/Rebain_LOLNerf_Learn_From_One_Look_CVPR_2022_paper.pdf )] LOLNeRF: Learn from One Look
[[Project]( https://ubc-vision.github.io/lolnerf/ )]

[[Chen et al. (ICCV '21)]( https://arxiv.org/abs/2103.15595 )] MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo
[[Project]( https://apchenstu.github.io/mvsnerf/ )]
[[Code]( https://github.com/apchenstu/mvsnerf )]
[[Video]( https://youtu.be/3M3edNiaGsA )]

[[Yu et al. (CVPR '21)]( https://arxiv.org/pdf/2012.02190.pdf )] Neural Radiance Fields from One or Few Images
[[Project]( https://alexyu.net/pixelnerf/)]
[[Code]( https://github.com/sxyu/pixel-nerf )]
[[Video]( https://youtu.be/voebZx7f32g )]

#### Large-scale
[[Tancik et al. (ARXIV '22)](https://arxiv.org/abs/2202.05263)] Block-NeRF: Scalable Large Scene Neural View Synthesis
[[Project]( https://waymo.com/research/block-nerf/)]
[[Video]( https://youtu.be/6lGMCAzBzOQ )]

[[Zhang et al. (**CVPR '22**)]( https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_NeRFusion_Fusing_Radiance_Fields_for_Large-Scale_Scene_Reconstruction_CVPR_2022_paper.pdf )] NeRFusion: Fusing Radiance Fields for Large-Scale Scene Reconstruction
[[Project]( https://jetd1.github.io/NeRFusion-Web/ )]

#### Sparse input
[[Long et al. (ARXIV '22)]( https://arxiv.org/pdf/2206.05737.pdf)] SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse views
[[Project]( https://www.xxlong.site/SparseNeuS/ )]

[[Suhail et al. (**CVPR '22**)]( https://arxiv.org/pdf/2112.09687.pdf )] Light Field Neural Rendering
[[Project]( https://light-field-neural-rendering.github.io/ )]
[[Code]( https://github.com/google-research/google-research/tree/master/light_field_neural_rendering )]

[[Niemeyer et al. (**CVPR '22**)]( https://arxiv.org/abs/2112.00724 )] RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs
[[Project]( https://m-niemeyer.github.io/regnerf )]
[[Code]( https://github.com/google-research/google-research/tree/master/regnerf )]
[[Video](https://www.youtube.com/watch?v=QyyyvA4-Kwc )]

#### Datasets
[[Downs et al. (ARXIV '22)]( https://arxiv.org/pdf/2204.11918.pdf )] Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Item
[[Blog]( https://ai.googleblog.com/2022/06/scanned-objects-by-google-research.html )]
[[Data]( https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research )]

### [5. Frontiers on 3D Point Cloud Learning](#content)

[[Wiersma et al. (SIGGRAPH '22)]( https://rubenwiersma.nl/assets/pdf/DeltaConv.pdf )] DeltaConv: Anisotropic Operators for Geometric Deep Learning on Point Clouds
[[Project]( https://rubenwiersma.nl/deltaconv )]
[[Code]( https://github.com/rubenwiersma/deltaconv )]
[[Supp.]( https://rubenwiersma.nl/assets/pdf/DeltaConv_supplement.pdf )]

[[Ran et al. (**CVPR '22**)]( https://arxiv.org/pdf/2205.05740.pdf )] Surface Representation for Point Clouds
[[Code]( https://github.com/hancyran/RepSurf )]

[[Mittal et al. (**CVPR '22**)]( https://arxiv.org/pdf/2203.09516.pdf )] AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation
[[Project]( https://yccyenchicheng.github.io/AutoSDF/ )]
[[Code]( https://github.com/yccyenchicheng/AutoSDF/ )]

[[Chen et al. (**CVPR '22**)](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_The_Devil_Is_in_the_Pose_Ambiguity-Free_3D_Rotation-Invariant_Learning_CVPR_2022_paper.pdf)] The Devil is in the Pose: Ambiguity-free 3D Rotation-invariant
Learning via Pose-aware Convolution

[[Jakab et al. (CVPR '21)]( https://arxiv.org/abs/2104.11224 )] KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control
[[Project]( https://tomasjakab.github.io/KeypointDeformer/ )]
[[Code]( https://github.com/tomasjakab/keypoint_deformer/ )]
[[Video]( https://youtu.be/GdDX1ZFh1k0 )]


### [6. 3D Object Detection, Pose Estimation](#content)
[[Yang et al. (**CVPR '22**)](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_ArtiBoost_Boosting_Articulated_3D_Hand-Object_Pose_Estimation_via_Online_Exploration_CVPR_2022_paper.pdf)] ArtiBoost: Boosting Articulated 3D Hand-Object Pose Estimation via
Online Exploration and Synthesis

[[Yin et al. (**CVPR '22**)]( https://arxiv.org/pdf/2203.15765.pdf )]  FisherMatch: Semi-Supervised Rotation Regression via Entropy-based Filtering
[[Project]( https://yd-yin.github.io/FisherMatch/ )]
[[Code]( https://github.com/yd-yin/FisherMatch )]

[[Sun et al. (**CVPR '22**)]( https://arxiv.org/pdf/2205.12257.pdf )] OnePose: One-Shot Object Pose Estimation without CAD Models
[[Project]( https://zju3dv.github.io/onepose/ )]
[[CodeSoon]( https://github.com/zju3dv/OnePose)]
[[Supp]( https://zju3dv.github.io/onepose/files/onepose_supp.pdf )]

[[Deng et al. (NeurIPS '21)]( https://arxiv.org/pdf/2112.07787.pdf )] Revisiting 3D Object Detection From an Egocentric Perspective

[[Li et al. (NeurIPS '21)]( https://arxiv.org/abs/2111.00190 )]  Leveraging SE(3) Equivariance for Self-Supervised Category-Level Object Pose Estimation
[[Project]( https://dragonlong.github.io/equi-pose/ )]
[[Code](https://github.com/dragonlong/equi-pose)]

[[Lu et al. (ICCV '21)]( https://openaccess.thecvf.com/content/ICCV2021/papers/Lu_Geometry_Uncertainty_Projection_Network_for_Monocular_3D_Object_Detection_ICCV_2021_paper.pdf )] Geometry Uncertainty Projection Network for Monocular 3D Object Detection
[[Code]( https://github.com/SuperMHP/GUPNet )]

[[Ahmadyan et al. (CVPR '21)]( https://openaccess.thecvf.com/content/CVPR2021/papers/Ahmadyan_Objectron_A_Large_Scale_Dataset_of_Object-Centric_Videos_in_the_CVPR_2021_paper.pdf )] Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild With Pose Annotations
[[Project]( https://github.com/google-research-datasets/Objectron/#tutorials )]
[[Code]( https://github.com/google-research-datasets/Objectron )]

[[Murphy et al. (ICML '21)]( https://arxiv.org/abs/2106.05965 )] Implicit-PDF: Non-Parametric Representation of Probability Distributions on the Rotation Manifold
[[Project]( https://implicit-pdf.github.io/ )]
[[Code]( https://github.com/google-research/google-research/tree/master/implicit_pdf )]
[[Video]( https://youtu.be/Y-MlRRy0xJA )]
[[Data](https://www.tensorflow.org/datasets/catalog/symmetric_solids)]


### [7. Neural Motion, Deformation Generation](#content)
  [[Kim et al. (ARXIV '22)]( https://arxiv.org/pdf/2202.04307.pdf )] Conditional Motion In-betweening
  [[Project]( https://jihoonerd.github.io/Conditional-Motion-In-Betweening/ )]

  [[He et al. (ARXIV '22)]( https://arxiv.org/pdf/2206.03287.pdf )] NeMF: Neural Motion Fields for Kinematic Animation

  [[Ianina et al. (**CVPR '22**)]( https://nsarafianos.github.io/assets/bodymap/bodymap.pdf )] BodyMap: Learning Full-Body Dense Correspondence Map
  [[Project]( https://nsarafianos.github.io/bodymap )]
  [[Supp]( https://nsarafianos.github.io/assets/bodymap/bodymap_suppl.pdf )]

  [[Muralikrishnan et al. (**CVPR '22**)]( https://sanjeevmk.github.io/glass_webpage/resources/glass_fullRes.pdf )] GLASS: Geometric Latent Augmentation for Shape Spaces
  [[Project]( https://sanjeevmk.github.io/glass_webpage/ )]
  [[Code]( https://github.com/sanjeevmk/glass/ )]
  [[Video]( https://sanjeevmk.github.io/glass_webpage/video/glass_dist.mp4 )]

  [[Taheri et al. (**CVPR '22**)]( https://arxiv.org/abs/2112.11454 )] GOAL: Generating 4D Whole-Body Motion for Hand-Object Grasping
  [[Project]( https://goal.is.tue.mpg.de/ )]
  [[Video]( https://youtu.be/A7b8DYovDZY )]
  [[Code]( https://github.com/otaheri/GOAL )]

  [[AIGERMAN et al. (SIGGRAPH '22)]( https://arxiv.org/pdf/2205.02904.pdf )]  Neural Jacobian Fields: Learning Intrinsic Mappings of Arbitrary　Meshes

  [[Raab et al. (Siggraph '22)]( https://arxiv.org/abs/2206.08010 )] MoDi: Unconditional Motion Synthesis from Diverse Data
  [[Project(need fix)]( https://sigal-raab.github.io/MoDi )]
  [[Video]( https://youtu.be/lRkdF8y3Du4)]

  [[Li et al. (SIGGRAPH '22)]( https://arxiv.org/abs/2205.02625 )] GANimator: Neural Motion Synthesis from a Single Sequence
  [[Project]( https://peizhuoli.github.io/ganimator/ )]
  [[Code](https://github.com/PeizhuoLi/ganimator )]
  [[Video]( https://youtu.be/OV9VoHMEeyI )]

  [[Wang et al. (NeurIPS '21)]( https://arxiv.org/abs/2106.11944 )]   MetaAvatar: Learning Animatable Clothed Human Models from Few Depth Images
  [[Project]( https://neuralbodies.github.io/metavatar/ )]
  [[Code](https://github.com/taconite/MetaAvatar-release )]
  [[Video](https://youtu.be/AwOwdKxuBXE )]

  [[Henter et al. (Siggraph Asia '20)]( http://kth.diva-portal.org/smash/get/diva2:1471598/FULLTEXT01.pdf )] MoGlow: Probabilistic and controllable motion synthesis using normalising flows
  [[Project]( https://simonalexanderson.github.io/MoGlow/ )]
  [[Code](https://github.com/simonalexanderson/MoGlow )]
  [[Video](https://youtu.be/pe-YTvavbtA )]

### [8. 3D Representations Learning for Robotics](#content)
  [[Driess et al. (ARXIV '22)]( https://arxiv.org/pdf/2206.01634.pdf )] Reinforcement Learning with Neural Radiance Fields
  [[Project]( https://dannydriess.github.io/nerf-rl/ )]
  [[Video]( https://dannydriess.github.io/nerf-rl/video.mp4 )]

  [[Gao et al. (**CVPR '22**)]( https://openaccess.thecvf.com/content/CVPR2022/html/Gao_ObjectFolder_2.0_A_Multisensory_Object_Dataset_for_Sim2Real_Transfer_CVPR_2022_paper.html )]  ObjectFolder 2.0: A Multisensory Object Dataset for Sim2Real Transfer
  [[Project]( https://ai.stanford.edu/~rhgao/objectfolder2.0/ )]
  [[Code]( https://github.com/rhgao/ObjectFolder )]
  [[Video]( https://youtu.be/e5aToT3LkRA )]

  [[Ortiz et al. (RSS '22)]( https://arxiv.org/abs/2204.02296 )] iSDF: Real-Time Neural Signed Distance Fields for Robot Perception
  [[Project]( https://joeaortiz.github.io/iSDF/ )]
  [[Code]( https://github.com/facebookresearch/iSDF )]
  [[Video]( https://youtu.be/mAKGl1wBSic )]

  [[Wi et al. (ICRA '22)]( https://arxiv.org/abs/2202.00868)] VIRDO: Visio-tactile Implicit Representations of Deformable Objects
  [[Project]( https://www.mmintlab.com/research/virdo-visio-tactile-implicit-representations-of-deformable-objects/ )]
  [[Code]( https://github.com/MMintLab/VIRDO )]

  [[Adamkiewicz et al. (ICRA '22)]( https://arxiv.org/pdf/2110.00168.pdf )] Vision-Only Robot Navigation in a Neural Radiance World
  [[Project]( https://mikh3x4.github.io/nerf-navigation/ )]
  [[Code]( https://github.com/mikh3x4/nerf-navigation )]
  [[Video]( https://youtu.be/5JjWpv9BaaE )]
  [[Data](https://drive.google.com/drive/folders/10_DWHIIetzeM2-1ziyZHujycWh-fNs29?usp=sharing)]

  [[Li et al. (CoRL '21)]( https://arxiv.org/abs/2107.04004 )] 3D Neural Scene Representations for Visuomotor Control
  [[Project]( https://3d-representation-learning.github.io/nerf-dy/)]
  [[Video]( https://youtu.be/ELPMiifELGc )]

  [[Ichnowski et al. (CoRL '21)]( https://arxiv.org/pdf/2110.14217.pdf )] Dex-NeRF: Using a Neural Radiance Field to Grasp Transparent Objects
  [[Project]( https://sites.google.com/view/dex-nerf )]
  [[Dataset]( https://github.com/BerkeleyAutomation/dex-nerf-datasets )]
  [[Video]( https://youtu.be/F9R6Nf1d7P4 )]


###  [9. Prompt Learning to 3D](#content)
  [[Tevet et al. (ARXIV '22)]( https://guytevet.github.io/motionclip-page/static/source/MotionCLIP.pdf )] MotionCLIP: Exposing Human Motion Generation to CLIP Space
  [[Project]( https://guytevet.github.io/motionclip-page/ )]
  [[Code]( https://github.com/GuyTevet/MotionCLIP )]

  [[Wang et al. (**CVPR '22**)]( https://openaccess.thecvf.com/content/CVPR2022/html/Wang_CLIP-NeRF_Text-and-Image_Driven_Manipulation_of_Neural_Radiance_Fields_CVPR_2022_paper.html )] CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields
  [[Project]( https://cassiepython.github.io/clipnerf/ )]
  [[Code]( https://github.com/cassiePython/CLIPNeRF )]
  [[Video]( https://cassiepython.github.io/clipnerf/images/video.mp4 )]

  [[Michel et al. (ARXIV '21)]( https://arxiv.org/abs/2112.03221 )] Text2Mesh: Text-Driven Neural Stylization for Meshes
  [[Project]( https://threedle.github.io/text2mesh/ )]
  [[Code]( https://github.com/threedle/text2mesh )]

#### Volume Rendering
  [[Sawhey et al. (SIGGRAPH '22)]( https://cs.dartmouth.edu/wjarosz/publications/sawhneyseyb22gridfree-small.pdf )]  Grid-free Monte Carlo for PDEs with spatially varying coefficients
  [[Project]( https://cs.dartmouth.edu/wjarosz/publications/sawhneyseyb22gridfree.html )]
  [[Code]( https://cs.dartmouth.edu/wjarosz/publications/sawhneyseyb22gridfree-reference-implementation.zip )]

### More resources
- [Amesome 3D machine learning collection](https://github.com/timzhang642/3D-Machine-Learning)

- [NeRF Fields in Visual Computing](https://neuralfields.cs.brown.edu/index.html)

- [Awesome-point-cloud-registration](https://github.com/wsunid/awesome-point-clouds-registration)

- [Awesome-equivariant-network](https://github.com/Chen-Cai-OSU/awesome-equivariant-network/blob/main/README.md#content)
