# 高光谱全色锐化中的 zero-shot 与 diffusion 方法综述

## 工作流摘要

- 主题：高光谱全色锐化中的 zero-shot 与 diffusion 方法综述
- 纳入论文数量：3

## 纳入论文

- A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf
- A novel pansharpening method based on cross stage partial network and transformer.pdf
- A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf

## 工作流步骤日志

- 步骤 1：已选中 3 篇论文。
- 步骤 2：已完成 GraphRAG 辅助检索（graph_mixed，证据 42 条）。
- 步骤 2：已加载图表摘要辅助发现（图表页 12 条）。
- 步骤 3：已完成多篇论文比较。
- 提示：多篇比较大模型输出非 JSON，已回退规则比较。
- 步骤 4：已生成综述提纲。
- 步骤 5：已导出 Markdown 报告。

## GraphRAG 辅助发现

- 检索模式：graph_mixed
- 检索问题：围绕“高光谱全色锐化中的 zero-shot 与 diffusion 方法综述”，结合以下论文：A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf；A novel pansharpening method based on cross stage partial network and transformer.pdf；A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf，从图谱中归纳任务类型、输入模态、核心方法、空间-光谱关系、退化/先验机制、数据集、评价指标和对用户模型的启示。
### 图谱主题簇
- 1. 图谱主题簇：PAN、pan-sharpening、IEEE：该图谱主题簇主要围绕 PAN, pan-sharpening, IEEE, IN, Remote Sens 展开。核心关系包括：1-D spectral features --related_to--> Spectral Features Spectral Features；167 bands --related_to--> SWIR；1918 --reports_metric--> SSIM。相关论文数量为 23 篇。
关键实体：PAN、pan-sharpening、IEEE、IN、Remote Sens、HSI
- 2. 图谱主题簇：s：该图谱主题簇主要围绕 s 展开。相关论文数量为 1 篇。
关键实体：s
- 3. 图谱主题簇：ARHS pansharpening、land use classification、location-sharing mechanism：该图谱主题簇主要围绕 ARHS pansharpening, land use classification, location-sharing mechanism, ARHS-CNN, HRHS images 展开。核心关系包括：ARHS pansharpening --addresses_task--> land use classification；ARHS pansharpening --addresses_task--> ob...
关键实体：ARHS pansharpening、land use classification、location-sharing mechanism、ARHS-CNN、HRHS images、He et al.
- 4. 图谱主题簇：C、P1、P2：该图谱主题簇主要围绕 C, P1, P2, Y, degrading matrices 展开。核心关系包括：Y --uses_degradation_model--> P1；Y --uses_degradation_model--> P2；Y --uses_method--> C。相关论文数量为 1 篇。
关键实体：C、P1、P2、Y、degrading matrices、separability assumption
### 关键关系
- IR-TenSR --addresses_task--> HYPERSPECTRAL AND MULTISPECTRAL IMAGE FUSION；证据：Xu et al. [59] proposed an IR-TenSR HSI-MSI fusion method
- Pyramid fully convolutional network --addresses_task--> HYPERSPECTRAL AND MULTISPECTRAL IMAGE FUSION；证据：Pyramid fully convolutional network for hyperspectral and multispectral image fusion
- superpixel-based weighted nuclear norm minimization --addresses_task--> HYPERSPECTRAL AND MULTISPECTRAL IMAGE FUSION；证据：Hyperspectral and multispectral image fusion via superpixel-based weighted nuclear norm minimization
- Spatial-spectral structured sparse low-rank representation --addresses_task--> HYPERSPECTRAL AND MULTISPECTRAL IMAGE FUSION；证据：Spatial-spectral structured sparse low-rank representation for hyperspectral image super-resolution
- NF-3DLogTNN --addresses_task--> HYPERSPECTRAL AND MULTISPECTRAL IMAGE FUSION；证据：NF-3DLogTNN: An effective hyperspectral and multispectral image fusion method
- LTTR --addresses_task--> HYPERSPECTRAL AND MULTISPECTRAL IMAGE FUSION；证据：Dian et al. [60] proposed a low tensor-train rank regularized HSI–MSI fusion method, called LTTR
### 代表性图谱证据
- 1. 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》页码 4：[61] establish the HSI-MSI fusion model in tensor format and leverage the properties of HR-HSI through TD techniques. For instance, Dian et al. [60] proposed a low tensor-train ran...
- 2. 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》页码 15：no. 5401016. [22] R. Dian, S. Li, and X. Kang, “Regularizing hyperspectral and multispec- tral image fusion by CNN denoiser,” IEEE Trans. Neural Netw. Learn. Syst., vol. 32, no. 3,...
- 3. 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》页码 15：h for hyperspectral image fusion,” IEEE Trans. Geosci. Remote Sens., vol. 60, 2022, Art. no. 5513417. [28] F. Zhou, R. Hang, Q. Liu, and X. Yuan, “Pyramid fully convolutional netwo...
- 4. 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》页码 15：Wang, and S. Li, “Hyperspectral and multi- spectral image fusion via superpixel-based weighted nuclear norm minimization,” IEEE Trans. Geosci. Remote Sens., vol. 61, 2023, Art. no....
- 5. 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》页码 15、16：r-resolution,” IEEE Trans. Geosci. Remote Sens., vol. 60, 2022, Art. no. 5529316. 5515417 IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 62, 2024 [60] R. Dian, S. Li, and...
- 6. 《Zero-Shot_Hyperspectral_Pansharpening_Using_Hysteresis-Based_Tuning_for_Spectral_Quality_Control.pdf》页码 19：- tral pansharpening based on improved deep image prior and residual reconstruction,” IEEE Trans. Geosci. Remote Sens., vol. 60, 2022, Art. no. 5520816. [63] P. Guan and E. Y . Lam...

## 图表与视觉内容分析

- 论文：A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf，第 1 页，Title, Abstract, Index Terms, Nomenclature, and Introduction of the paper 'A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion'.
类型：figure
摘要：The page contains the title, author information, abstract, index terms, nomenclature table, and the beginning of the introduction section. It introduces the problem of Hyperspectral image (HSI) and multispectral image (MSI) fusion and proposes a novel method called coupled tensor double-factor (CTDF) decomposition.
技术细节：The abstract explains that HSI-MSI fusion aims to generate a high spatial resolution HSI (HR-HSI) by merging HSI and MSI. The proposed CTDF method uses tensor double-factor (TDF) decomposition to represent a third-order HR-HSI as a fourth-order spatial factor and a third-order spectral factor connected through tensor contraction. The nomenclature table defines mathematical notations used in the paper, such as tensors, matrices, vectors, Frobenius norm, transpose, identity matrix, vectorization operator, and Kronecker product.
图片路径：data\processed\vision\paper_0e8b4b4ce8\page_001.png
- 论文：A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf，第 2 页，Fig. 1. Graphical representation of HR-HSI X ∈RH×W×S using different TDs, i.e., (a) CP decomposition with CP-rank r, (b) Tucker decomposition with Tucker-rank [r1,r2,r3], (c) TR decomposition with TR-rank [r1,r2,r3], and (d) TDF decomposition with TDF-rank [r1,r2], fourth-order spatial factor C and third-order spectral factor D.
类型：architecture
摘要：该图展示了高光谱图像（HR-HSI）在不同张量分解（TD）方法下的图形表示。包括：(a) CP分解，(b) Tucker分解，(c) TR分解，以及(d) 本文提出的TDF（Tensor Double-Factor）分解。图中还给出了空间因子、光谱因子、三阶核心张量和三阶对角张量的图例。
技术细节：图(a)展示了CP分解，具有CP秩r；图(b)展示了Tucker分解，具有Tucker秩[r1, r2, r3]；图(c)展示了TR分解，具有TR秩[r1, r2, r3]；图(d)展示了TDF分解，具有TDF秩[r1, r2]，包含四阶空间因子C和三阶光谱因子D。空间因子表示包含至少一个空间模式（高度模式、宽度模式）的张量，光谱因子表示包含光谱模式的张量，核心张量和对角张量不包含空间和光谱模式。
图片路径：data\processed\vision\paper_0e8b4b4ce8\page_002.png
- 论文：A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf，第 3 页，Fig. 2. TDF degradation model from the HR-HSI to both MSI and HSI. (Top) From the HR-HSI X to MSI Z spectral degradation. (Bottom) From the HR-HSI X to HSI Y spatial degradation. Fig. 3. Flowchart of the proposed CTDF method for HSI-MSI fusion.
类型：architecture
摘要：图2展示了TDF（Tensor Double-Factor）退化模型，描述了从高分辨率高光谱图像（HR-HSI）到多光谱图像（MSI）的光谱退化过程，以及到多光谱图像（HSI）的空间退化过程。图3展示了提出的CTDF（Coupled Tensor Double-Factor）方法的流程图，包括输入（HSI和MSI）、模型构建（TDF分解和CTDF模型公式化）、迭代更新空间因子和光谱因子，以及最终输出HR-HSI。
技术细节：图2中，HR-HSI X ∈ R^(H×W×S) 通过TDF分解为空间因子 C 和光谱因子 D。顶部路径显示通过矩阵 P1, P2 作用于空间因子 C 得到退化空间因子，再与退化光谱因子 D_hat 结合生成 MSI Z ∈ R^(H×W×s)（光谱退化）。底部路径显示通过矩阵 P3 作用于光谱因子 D 得到退化光谱因子，再与退化空间因子 C_hat 结合生成 HSI Y ∈ R^(h×w×S)（空间退化）。图3流程：输入HSI Y和MSI Z -> 通过TDF分解Y和Z -> 构建CTDF模型 -> 通过算法2迭代更新空间因子C和光谱因子D -> 输出HR-HSI X。
图片路径：data\processed\vision\paper_0e8b4b4ce8\page_003.png
- 论文：A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf，第 5 页，Preliminaries and Proposed TDF Decomposition
类型：figure
摘要：The page contains mathematical definitions and equations related to tensor operations, including diagonal tensor, k-mode product, tensor permutation, generalized tensor k-unfolding, and generalized tensor contraction. It also introduces the proposed TDF (Tensor Double-Factor) decomposition for a third-order HR-HSI, representing it as a fourth-order spatial factor and a third-order spectral factor.
技术细节：The text defines various tensor operations (Definitions 1-5) and presents the element-wise form of the TDF decomposition in Equation (7) and its tensor format representation in Equation (8) using generalized tensor contraction. The TDF decomposition represents a third-order HR-HSI X as X = C x_{4,1}^{1,3} D, where C is a fourth-order spatial factor and D is a third-order spectral factor.
图片路径：data\processed\vision\paper_0e8b4b4ce8\page_005.png
- 论文：A novel pansharpening method based on cross stage partial network and transformer.pdf，第 1 页，A novel pansharpening method based on cross stage partial network and transformer
类型：figure
摘要：This page is the first page of the paper, containing the title, authors, abstract, and introduction. It introduces a novel pansharpening method named GF-CSTNet, which combines Guided Filtering, Cross Stage Partial Network (CSPNet), and Transformer to leverage both local detail acquisition and global data procuring capabilities.
技术细节：The abstract describes the proposed GF-CSTNet method. It mentions using Guided Filtering (GF) to enhance remote sensing image data, combining CSPNet and Transformer structures, designing a Rep-Conv2Former method with a multi-scale convolution modulator block, constructing a reparameterization module to optimize inference speed, and devising a residual learning module incorporating attention. Experimental results on GaoFen-2 and WorldView-3 datasets are mentioned.
图片路径：data\processed\vision\paper_bcbeef25cc\page_001.png
- 论文：A novel pansharpening method based on cross stage partial network and transformer.pdf，第 2 页，Introduction and Related works
类型：figure
摘要：The page contains text introducing the background of pansharpening methods, including model-based methods and deep learning techniques. It also lists the main contributions of the proposed GF-CSTNet and begins the 'Related works' section discussing FusionNet.
技术细节：The text discusses various pansharpening methods like VO, Compressed Sensing, Bayesian-based fusion, PNN, PanNet, MSDCNN, FusionNet, SDRCNN, MMFN, and SSCAConv. It introduces the proposed GF-CSTNet combining Transformer and CSPNet, mentioning guided filtering, Rep-Conv2Former module, and a residual learning module with attention. Equation (1) for FusionNet is provided: MS_hat = MS_tilde + f_theta_FS(P^D - MS_tilde).
图片路径：data\processed\vision\paper_bcbeef25cc\page_002.png
- 论文：A novel pansharpening method based on cross stage partial network and transformer.pdf，第 3 页，Figure 1 depicts the overall design of the fusion network, and each module will be explained in more detail in the following sections.
类型：architecture
摘要：该图展示了所提出的全色锐化融合网络的整体架构设计。根据文本描述，该网络结合了CSPNet（跨阶段部分网络）和Transformer结构，并引入了引导滤波（Guided filtering）和残差学习模块。
技术细节：图中展示了融合网络的整体流程。输入通常包括上采样的多光谱图像（MS）和全色图像（PAN）。网络内部集成了CSPNet模块以增强特征提取能力和效率，通过分割特征图并使用跨阶段层结合结果。同时，网络采用了基于Transformer的注意力机制（如Rep-Conv2Former注意力块），利用多头自注意力（MHSA）计算输入输出间的隐式表示。此外，还包含引导滤波模块用于去噪和增强细节，以及残差学习模块用于提取边缘信息。
图片路径：data\processed\vision\paper_bcbeef25cc\page_003.png
- 论文：A novel pansharpening method based on cross stage partial network and transformer.pdf，第 4 页，Figure 1. Integrated network overall structure diagram.
类型：architecture
摘要：该图展示了全色锐化（pansharpening）集成网络的整体结构。输入包括全色图像（PAN）和多光谱图像（MS）。PAN经过复制得到$P^D$，MS经过上采样得到$\overline{MS}$。$\overline{MS}$和$P^D$输入到引导滤波器（Guided Filter）中，结合残差学习（Residual Learning）得到$\overline{MS}'$。然后，$P^D$与$\overline{MS}'$进行矩阵减法，得到$P^D - \overline{MS}'$。该差值图像输入到CSPNet中，得到特征$f_{\theta_0}(P^D - \overline{MS}')$。最后，将CSPNet的输出与$P^D - \overline{MS}'$进行矩阵加法，并与真实标签（GT）计算Loss。
技术细节：模型包含引导滤波（Guided Filter）、残差学习（Residual Learning）、CSPNet等模块。信息流从PAN和MS开始，经过上采样、滤波、特征提取（CSPNet）和残差连接，最终输出融合结果并计算损失。图例说明了圆圈减号代表矩阵减法，圆圈加号代表矩阵加法。
图片路径：data\processed\vision\paper_bcbeef25cc\page_004.png
- 论文：A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf，第 1 页，A Dual-Decoder-VAE-Based Latent Diffusion Model for PAN-Sharpening
类型：figure
摘要：This page contains the title, authors, abstract, and introduction of the paper. It introduces the problem of PAN-sharpening, discusses existing deep learning and diffusion model approaches, highlights the limitations of conventional VAEs in latent diffusion models for high bit-depth satellite images, and proposes a Dual-Decoder-VAE (DDV) and a DDV-based latent diffusion PAN-sharpening (DDV-LDP) model.
技术细节：The abstract mentions that recent diffusion-based PAN-sharpening methods require 25-50 denoising steps. The proposed DDV-LDP model achieves 0.06 dB and 0.98 dB higher PSNR values than the state-of-the-art method (UKnowDif-T) on the KOMPSAT-3A and WorldView-III datasets, respectively, with a 99.5% reduction in testing time.
图片路径：data\processed\vision\paper_b549d78781\page_001.png
- 论文：A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf，第 2 页，Fig. 1. Comparison of structures between the CVAE and our proposed DDV. The architectures of CVAE and DDV are depicted on the left and right sides, respectively.
类型：architecture
摘要：该图对比了传统变分自编码器（CVAE）与本文提出的双解码器变分自编码器（DDV）的结构。左侧展示了CVAE的基本结构，包含编码器（Encoder）和解码器（Decoder），输入低分辨率多光谱图像（$I_{ms}^L$），输出重建图像（$I_{ms}^{L,r}$）。右侧详细展示了DDV的架构，包含编码器（Enc）、潜在解码器（LatDec）和全色锐化解码器（PSDec）。
技术细节：DDV模型包含三个主要部分： 1. 编码器（Enc）：接收$VLR$ MS图像（$I_{ms}^L$）作为输入，通过一系列ResBlock和卷积层提取特征，生成潜在表示（$z^L$）。 2. 潜在解码器（LatDec）：接收潜在表示$z^L$，通过上采样和ResBlock生成辅助数据（$I_{ms}^{VL}$），用于支持PSDec。 3. 全色锐化解码器（PSDec）：接收$LR$ PAN图像（$I_{pan}^L$）和$VLR$ MS图像（$I_{ms}^{VL}$），通过Concat、Pixel Unshuffle、ResBlock和上采样等操作，结合双标准化层（STL）和去标准化层（DSTL），最终输出高分辨率多光谱图像（$I_{ms}^{L,r}$）。 图中还展示了特征图的尺寸变化（如H/2×W/2×128, H/4×W/4×128等）以及具体的卷积核大小和步长。
图片路径：data\processed\vision\paper_b549d78781\page_002.png
- 论文：A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf，第 3 页，Fig. 2. Training and inference process of DDV-LDP.
类型：architecture
摘要：该图展示了DDV-LDP（Dual-Decoder-VAE-Based Latent Diffusion Model for PAN-Sharpening）模型的训练和推理过程。左侧为训练过程，分为前向扩散过程（Forward Diffusion Process）和反向扩散过程（Reverse Diffusion Process）。右侧为推理过程（Inference），展示了如何从输入生成最终的高分辨率全色锐化图像。
技术细节：1. **训练过程（左侧）**： - **前向扩散**：输入图像 $I^L_{ms}$ 经过编码器（Enc）得到初始潜变量 $z^L_0$，随后逐步添加噪声得到 $z^L_t$。 - **反向扩散**：使用 U-Net Denoiser 预测噪声。输入包括带噪潜变量 $z^L_t$、经过 Pixel Unshuffle (PU) 处理的全色图像 $I^L_{pan}$ 和多光谱图像 $I^L_{ms}$ 的拼接特征。 2. **推理过程（右侧）**： - 输入高分辨率全色图像 $I^H_{pan}$ 和低分辨率多光谱图像 $I^L_{ms}$。 - $I^H_{pan}$ 经过标准化和 Pixel Unshuffle (PU) 处理，与 $I^L_{ms}$ 拼接后输入 U-Net Denoiser。 - U-Net 包含 STL 和 DSTL 模块，通过跳跃连接（Concatenation）传递特征。 - 去噪后的潜变量 $z^{H,r}$ 输入 Latent Decoder (LatDec) 和 PAN-Sharpening Decoder (PSDec)。 - LatDec 的输出注入到 PSDec 的中间层，最终结合双三次插值（Bicubic x4）的 $I^L_{ms}$ 生成高分辨率多光谱图像 $I^H_{ps}$。
图片路径：data\processed\vision\paper_b549d78781\page_003.png
- 论文：A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf，第 4 页，Fig. 3. Qualitative comparison between the SOTA methods and our DDV-LDP using the KOMPSAT-3A dataset (full resolution). MDCUN: noisy and blurred, PGCU: structural inaccuracies, UAPN: blurring and color artifacts, LGTEUN/SSMNET/PAN-Mamba: blurring and color bleeding, TMDIFF/DDIF/ UKnowDif-T: structural distortions, DDV-LDP (Ours): relatively better color and structures. Zoomed-in view for better visualization.
类型：result
摘要：The figure presents a visual comparison of PAN-sharpening results from various state-of-the-art (SOTA) methods against the proposed DDV-LDP method on the KOMPSAT-3A dataset. It includes the input Low-Resolution Multispectral (LR MS) image and High-Resolution Panchromatic (HR PAN) image, followed by the outputs of MDCUN, PGCU, UAPN, LGTEUN, SSMNET, PAN-Mamba, TMDIFF, DDIF, UKnowDif-T, and DDV-LDP. Each result has a zoomed-in region (highlighted by a yellow box) to show fine details, with red arrows pointing to specific areas of interest.
技术细节：The comparison highlights specific flaws in existing methods: MDCUN produces noisy and blurred results; PGCU has structural inaccuracies; UAPN shows blurring and color artifacts; LGTEUN, SSMNET, and PAN-Mamba exhibit blurring and color bleeding; TMDIFF, DDIF, and UKnowDif-T suffer from structural distortions. In contrast, the proposed DDV-LDP method demonstrates relatively better color fidelity and structural preservation, particularly in restoring the edges of the red roof in the zoomed-in view.
图片路径：data\processed\vision\paper_b549d78781\page_004.png

## 多篇论文比较

- 比较主题：高光谱全色锐化中的 zero-shot 与 diffusion 方法综述
- 纳入论文数量：3

### 一、单篇论文概括
| 论文 | 研究问题 | 研究对象 | 方法 | 主要结论 |
| --- | --- | --- | --- | --- |
| A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf | 研究如何解决高光谱-多光谱图像融合（HSI-MSI fusion）任务中的空间细节增强与光谱信息保持问题。 | 低空间分辨率高光谱图像（HSI）与高空间分辨率多光谱图像（MSI）。 | 耦合张量双因子模型（CTDF），利用张量分解/双因子表示建模高光谱与多光谱图像的空间-光谱结构。 | 实验主要围绕 SAM、PSNR、SSIM、ERGAS、RMSE 等指标评价融合质量；完整优劣结论需结合结果表进一步核查。 |
| A novel pansharpening method based on cross stage partial network and transformer.pdf | 研究如何解决多光谱/遥感全色锐化（pansharpening）任务中的空间细节增强与光谱信息保持问题。 | 全色图像（PAN）与低空间分辨率多光谱图像（MS/MSI）。 | 结合 cross-stage partial 网络与 Transformer/注意力结构进行全色锐化特征融合。 | 实验显示该方法在多个客观评价指标上取得较优结果。 |
| A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf | this objective function, the HR MS images are obtained. | DL-based PAN-sharpening methods are actively studied. | network (SSMNET) [4] proposes a spatial–spectral modulation | network (SSMNET) [4] proposes a spatial–spectral modulation |

### 二、面向遥感图像融合/全色锐化的比较
#### 任务类型比较
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》主要对应HSI-MSI fusion / 高光谱-多光谱融合。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》主要对应MSI pansharpening 或通用 pansharpening。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》主要对应MSI pansharpening 或通用 pansharpening。
#### 输入模态比较
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》涉及的输入模态：HSI/高光谱图像、MSI/多光谱图像。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》涉及的输入模态：PAN/全色图像、MSI/多光谱图像。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》涉及的输入模态：PAN/全色图像、MSI/多光谱图像。
#### 核心方法比较
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》：耦合张量双因子模型（CTDF），利用张量分解/双因子表示建模高光谱与多光谱图像的空间-光谱结构。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》：结合 cross-stage partial 网络与 Transformer/注意力结构进行全色锐化特征融合。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》：network (SSMNET) [4] proposes a spatial–spectral modulation
#### 空间信息建模比较
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》空间建模：围绕空间结构恢复建模。方法线索：耦合张量双因子模型（CTDF），利用张量分解/双因子表示建模高光谱与多光谱图像的空间-光谱结构。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》空间建模：利用 PAN/全色图像提供高空间分辨率细节；用 Transformer/注意力建模长程空间依赖或跨模态特征交互。方法线索：结合 cross-stage partial 网络与 Transformer/注意力结构进行全色锐化特征融合。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》空间建模：利用 PAN/全色图像提供高空间分辨率细节；用卷积或编码-解码结构提取局部空间结构。方法线索：network (SSMNET) [4] proposes a spatial–spectral modulation
#### 光谱信息建模比较
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》光谱建模：以 HSI/LRHS 的光谱信息约束融合结果；关注光谱保真、光谱一致性或 SAM 等失真约束；用张量/低秩结构表达空间-光谱相关性。方法线索：耦合张量双因子模型（CTDF），利用张量分解/双因子表示建模高光谱与多光谱图像的空间-光谱结构。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》光谱建模：围绕光谱信息保持建模。方法线索：结合 cross-stage partial 网络与 Transformer/注意力结构进行全色锐化特征融合。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》光谱建模：在潜空间或生成模型中维持光谱表达。方法线索：network (SSMNET) [4] proposes a spatial–spectral modulation
#### 退化模型/先验建模比较
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》涉及：张量/低秩先验。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》涉及：注意力机制。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》涉及：latent space 表示、VAE/autoencoder 表示学习、扩散模型。
#### 数据集与实验协议比较
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》数据集/实验协议：CAVE、Harvard、Pavia University、IKONOS、Chikusei、WorldView-3。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》数据集/实验协议：GaoFen-2、WorldView-3。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》数据集/实验协议：network (SSMNET) [4] proposes a spatial–spectral modulation。
#### 指标与主要结论比较
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》涉及的评价指标：SAM、PSNR、SSIM、ERGAS、RMSE。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》涉及的评价指标：原文未明确说明。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》涉及的评价指标：原文未明确说明。

### 三、综合判断
#### 共同主题
- 选中论文覆盖多光谱/遥感全色锐化、高光谱-多光谱融合等相关任务。
- 共同核心问题是提升空间分辨率，同时尽量保持光谱或辐射一致性。
- 这些论文适合用于比较不同融合任务之间的方法迁移关系。
#### 优势与局限
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》主要结论：实验主要围绕 SAM、PSNR、SSIM、ERGAS、RMSE 等指标评价融合质量；完整优劣结论需结合结果表进一步核查。；局限性：部分配置的运行时间显著增加，说明该类方法需要关注计算效率。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》主要结论：实验显示该方法在多个客观评价指标上取得较优结果。；局限性：ture is constructed to not only reduce complexity but also extract more feature information from different
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》主要结论：network (SSMNET) [4] proposes a spatial–spectral modulation；局限性：原文未明确说明具体局限；综述中可重点核查采样开销、训练稳定性以及真实退化场景下的泛化能力。
#### 对用户高光谱全色锐化研究的综合启示
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》提示：可借鉴其耦合张量双因子模型（CTDF），利用张量分解/双因子表示建模高光谱与多光谱图像的空间-光谱结构，用于设计更稳健的空间-光谱融合、条件调制或消融实验。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》提示：可借鉴其结合 cross-stage partial 网络与 Transformer/注意力结构进行全色锐化特征融合，用于设计更稳健的空间-光谱融合、条件调制或消融实验。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》提示：DL-based PAN-sharpening methods are actively studied。
#### 可整合的研究思路
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》提示：可借鉴其耦合张量双因子模型（CTDF），利用张量分解/双因子表示建模高光谱与多光谱图像的空间-光谱结构，用于设计更稳健的空间-光谱融合、条件调制或消融实验。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》提示：可借鉴其结合 cross-stage partial 网络与 Transformer/注意力结构进行全色锐化特征融合，用于设计更稳健的空间-光谱融合、条件调制或消融实验。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》提示：DL-based PAN-sharpening methods are actively studied。

## 综述提纲

- 主题：高光谱全色锐化中的 zero-shot 与 diffusion 方法综述
- 参考论文：A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf, A novel pansharpening method based on cross stage partial network and transformer.pdf, A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf

### 1. 研究背景与问题提出
- 围绕“高光谱全色锐化中的 zero-shot 与 diffusion 方法综述”梳理 PAN 与 LRHS/HSI 在空间分辨率和光谱分辨率上的互补性。
- 选中论文覆盖多光谱/遥感全色锐化、高光谱-多光谱融合等相关任务。
- 共同核心问题是提升空间分辨率，同时尽量保持光谱或辐射一致性。
- 这些论文适合用于比较不同融合任务之间的方法迁移关系。

### 2. 任务定义与输入模态
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》主要对应HSI-MSI fusion / 高光谱-多光谱融合。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》主要对应MSI pansharpening 或通用 pansharpening。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》主要对应MSI pansharpening 或通用 pansharpening。
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》涉及的输入模态：HSI/高光谱图像、MSI/多光谱图像。

### 3. 传统方法与模型驱动方法
- 梳理 CS、MRA、VO、Bayesian/model-based、tensor decomposition 和 degradation consistency 等传统或模型驱动路线。
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》涉及：张量/低秩先验。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》涉及：注意力机制。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》涉及：latent space 表示、VAE/autoencoder 表示学习、扩散模型。

### 4. 深度网络方法
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》：耦合张量双因子模型（CTDF），利用张量分解/双因子表示建模高光谱与多光谱图像的空间-光谱结构。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》：结合 cross-stage partial 网络与 Transformer/注意力结构进行全色锐化特征融合。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》：network (SSMNET) [4] proposes a spatial–spectral modulation

### 5. 空间信息建模
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》空间建模：围绕空间结构恢复建模。方法线索：耦合张量双因子模型（CTDF），利用张量分解/双因子表示建模高光谱与多光谱图像的空间-光谱结构。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》空间建模：利用 PAN/全色图像提供高空间分辨率细节；用 Transformer/注意力建模长程空间依赖或跨模态特征交互。方法线索：结合 cross-stage partial 网络与 Transformer/注意力结构进行全色锐化特征融合。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》空间建模：利用 PAN/全色图像提供高空间分辨率细节；用卷积或编码-解码结构提取局部空间结构。方法线索：network (SSMNET) [4] proposes a spatial–spectral modulation

### 6. 光谱信息保持
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》光谱建模：以 HSI/LRHS 的光谱信息约束融合结果；关注光谱保真、光谱一致性或 SAM 等失真约束；用张量/低秩结构表达空间-光谱相关性。方法线索：耦合张量双因子模型（CTDF），利用张量分解/双因子表示建模高光谱与多光谱图像的空间-光谱结构。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》光谱建模：围绕光谱信息保持建模。方法线索：结合 cross-stage partial 网络与 Transformer/注意力结构进行全色锐化特征融合。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》光谱建模：在潜空间或生成模型中维持光谱表达。方法线索：network (SSMNET) [4] proposes a spatial–spectral modulation

### 7. 退化模型与物理约束
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》涉及：张量/低秩先验。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》涉及：注意力机制。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》涉及：latent space 表示、VAE/autoencoder 表示学习、扩散模型。

### 8. 数据集、实验协议与评价指标
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》数据集/实验协议：CAVE、Harvard、Pavia University、IKONOS、Chikusei、WorldView-3。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》数据集/实验协议：GaoFen-2、WorldView-3。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》数据集/实验协议：network (SSMNET) [4] proposes a spatial–spectral modulation。
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》涉及的评价指标：SAM、PSNR、SSIM、ERGAS、RMSE。

### 9. 局限与后续研究方向
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》的局限：部分配置的运行时间显著增加，说明该类方法需要关注计算效率。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》的局限：ture is constructed to not only reduce complexity but also extract more feature information from different
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》的局限：原文未明确说明具体局限；综述中可重点核查采样开销、训练稳定性以及真实退化场景下的泛化能力。

### 10. 对用户当前模型的启示
- 《A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion.pdf》提示：可借鉴其耦合张量双因子模型（CTDF），利用张量分解/双因子表示建模高光谱与多光谱图像的空间-光谱结构，用于设计更稳健的空间-光谱融合、条件调制或消融实验。
- 《A novel pansharpening method based on cross stage partial network and transformer.pdf》提示：可借鉴其结合 cross-stage partial 网络与 Transformer/注意力结构进行全色锐化特征融合，用于设计更稳健的空间-光谱融合、条件调制或消融实验。
- 《A_Dual-Decoder-VAE-Based_Latent_Diffusion_Model_for_PAN-Sharpening.pdf》提示：DL-based PAN-sharpening methods are actively studied。
