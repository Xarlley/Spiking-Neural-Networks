# Spiking Transformer：2018-2025

## 1 神经形态注意力的简介

人工智能领域的两大强大范式——受大脑启发、具备高能效的脉冲神经网络（SNN）与拥有卓越上下文理解能力的Transformer架构——在此交汇，催生了一类极具潜力的新型模型：脉冲Transformer。本调研报告旨在全面回顾自2018年至今，在顶级学术会议上发表的已实现的脉冲Transformer架构，系统梳理其关键创新、追溯不同模型间的学术传承，并分析塑造这一快速发展领域的总体趋势。

## 2 奠基性架构

脉冲Transformer领域通过两项开创性工作正式确立，它们提出了在脉冲神经网络中实现自注意力的首批可行方法。尽管两种模型都实现了这一目标，但它们在关键架构细节——残差连接——上的不同处理方式，造成了设计理念上的根本分歧，并影响了此后该领域的所有研究。

### 2.1 Spikformer：首个脉冲视觉Transformer

发表于**ICLR 2023**的Spikformer是成功将视觉Transformer（ViT）架构转换到脉冲领域的开创性工作。 其主要贡献是引入了**脉冲自注意力（Spiking Self-Attention, SSA）**机制，为神经形态注意力提供了第一个功能蓝图。

[论文](https://openreview.net/forum?id=frE4fUwz_h) [代码仓库](https://github.com/ZK-Zhou/spikformer)

SSA的核心创新在于将注意力机制中的核心张量——查询（Q）、键（K）和值（V）——转换为二进制脉冲序列。这种适配使得注意力图可以使用基于脉冲的操作来计算。SSA中一个关键的设计选择是完全移除了softmax函数，这是传统Transformer的标准组件。 作者认为，由于基于脉冲的操作本质上是非负的，softmax提供的归一化是多余的，移除它可以显著降低计算复杂性而不会损害性能。 这一简化是一个关键突破，使得自注意力机制在SNN中变得易于处理。

对于用于训练深度网络的关键组件——残差连接，Spikformer采用了**脉冲元素级（Spike-Element-Wise, SEW）** 快捷连接。这一设计直接继承自早期关于脉冲残差网络（ResNet）的工作，涉及将不同层的脉冲序列进行直接的元素级相加。 尽管功能上可行，但这一选择很快成为该领域争论和改进的主要焦点。

### 2.2 Spike-driven Transformer：向纯粹脉冲驱动范式的转变

在**NeurIPS 2023**上提出的Spike-driven Transformer被视为对Spikformer的直接且更具原则性的改进。 这项工作确立了“脉冲驱动”范式，这是一套更严格的神经形态架构设计原则。该范式要求所有计算必须是事件驱动的（即，对于零值输入不执行任何计算），并且所有神经元间的通信必须完全通过二进制脉冲进行。遵循这些原则，所有密集的矩阵乘法都可以转换成稀疏、高能效的加法操作。

[论文](https://papers.neurips.cc/paper_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf) [代码仓库](https://github.com/BICLab/Spike-Driven-Transformer)

为实现这一点，Spike-driven Transformer引入了两项关键创新：

脉冲驱动自注意力（Spike-Driven Self-Attention, SDSA）： 该机制通过从注意力模块中移除所有乘法操作，进一步完善了Spikformer的SSA。SDSA不使用点积，而是通过元素级掩码（实现为脉冲序列间的哈达玛积）和稀疏的列向求和来计算注意力分数。这种新颖的公式使得注意力机制在计算上是线性的，并且效率极高。

膜电位快捷连接（Membrane Shortcut, MS）： Spike-driven Transformer最重要的贡献是其对残差连接的重新评估。作者指出了Spikformer使用的SEW快捷连接存在一个关键缺陷：两个二进制脉冲序列相加可能产生多比特的整数值（例如，1+1=2），而这些值并非二进制脉冲。这种“非脉冲计算”需要在后续层中进行整数乘法，违反了脉冲驱动原则，并损害了专为纯二进制信号设计的神经形态硬件的全部节能潜力。 为纠正这一点，他们提出了膜电位快捷连接，即残差连接作用于神经元的模拟膜电位上，在应用脉冲发放阈值之前。这确保了神经元的输出以及所有传输的信号都严格保持为二进制。

### 2.3 小节结语

从Spikformer的SEW快捷连接到Spike-driven Transformer的MS快捷连接的架构转向，不仅仅是一次增量改进；它标志着该领域设计哲学的一次根本性分裂。这一转变标志着从简单地将ANN组件适配到脉冲工作方式，转向一种更有原则的方法，即在神经形态系统的计算约束和机遇的指导下，从头开始重新构想Transformer架构。SEW快捷连接虽然有效，但它是脉冲ResNet谱系的遗留产物。用MS快捷连接取而代之，确立了“脉冲驱动”范式作为评估脉冲Transformer模型理论效率和硬件兼容性的一个新的、更严格的标准。这一迅速识别并纠正架构缺陷的进展，催生了两个不同的谱系：遵循Spikformer设计的“整数驱动”模型，以及遵循Spike-driven Transformer所定原则的真正“脉冲驱动”模型。这表明该领域在加强对受大脑启发的计算原则的严格遵守方面迅速成熟。

## 3 脉冲驱动谱系：扩展性能与通用性

在“脉冲驱动”范式确立之后，出现了一条清晰而直接的研究路线，它建立在Spike-driven Transformer的基础架构之上。这一谱系的特点是一系列继承性的改进，旨在扩展模型的性能、将其能力推广到更广泛的任务，并推动其效率以满足在资源受限硬件上的实际部署需求。

### 3.1 Meta-SpikeFormer (Spike-driven Transformer V2)：通用化架构

[论文](https://iclr.cc/virtual/2024/poster/19587) [代码仓库](https://github.com/BICLab/Spike-Driven-Transformer-V2)

该谱系的第一个主要演进是Meta-SpikeFormer，在**ICLR 2024**上发表，也被称为Spike-driven Transformer V2。 这项工作的主要目标是将原始的Spike-driven Transformer从一个专门的图像分类模型扩展为一个通用的SNN骨干网络。它明确地建立在其前身的SDSA和MS快捷连接机制之上，将它们作为更灵活的元架构的基础。

Meta-SpikeFormer的关键成就是，它成为首个同时支持图像分类、目标检测和语义分割这三大核心计算机视觉任务的直接训练SNN骨干网络。 通过在这三个领域均达到SNN的SOTA（state-of-the-art）水平，Meta-SpikeFormer证明了脉冲驱动方法不仅限于简单的分类任务，还可以作为更复杂视觉应用的强大且高效的基础。这标志着脉冲Transformer从一个利基模型向通用视觉骨干网络候选者迈出了关键一步。 

### 3.2 Quantized Spike-driven Transformer (QSD-Transformer)：提升硬件效率

在Meta-SpikeFormer已证明的通用性基础上，发表于**ICLR 2025**的**Quantized Spike-driven Transformer (QSD-Transformer)** 专注于硬件部署的实际挑战。 这项工作将量化技术引入脉冲驱动谱系，旨在通过使用低比特宽度的参数来显著减少模型的内存占用和计算需求。

[论文](https://arxiv.org/abs/2501.13492) [代码仓库](https://github.com/bollossom/QSD-Transformer)

在此过程中，研究人员发现了一个该架构特有的新挑战：一种他们称为“脉冲信息失真”（spike information distortion, SID）的性能下降问题，该问题源于量化脉冲驱动自注意力（Q-SDSA）模块中激活值的双峰分布。 为克服这一问题，他们提出了一种复杂的双层优化策略。在底层，设计了一个**信息增强型Leaky Integrate-and-Fire（IE-LIF）**神经元来校正信息分布。在顶层，采用了一种细粒度蒸馏（Fine-Grained Distillation, FGD）方案，以使量化模型的行为与其全精度对应模型对齐。这项工作表明，即使在量化的严格约束下，脉冲驱动架构的高性能也可以得以保持，为其在资源极其有限的边缘设备上的部署铺平了道路。

### 3.3 Scaling Spike-driven Transformer (E-SpikeFormer)：达到ANN级别的性能

该谱系的最新进展，被称为Scaling Spike-driven Transformer或E-SpikeFormer，于**2025年发表在IEEE T-PAMI**上。 这项工作，也被称为Spike-Driven-Transformer-V3，直接解决了SNN领域最重大的挑战之一：扩展模型以达到大规模参数量，从而缩小与ANN的性能差距。

[论文](https://arxiv.org/html/2411.16061v1) [代码仓库](https://github.com/biclab/spike-driven-transformer-v3)

E-SpikeFormer的核心贡献是**脉冲发放近似（Spike Firing Approximation, SFA）** 训练方法。该技术在训练阶段使用整数值激活，这可以在GPU上高效处理，然后在推理阶段将这些整数转换为二进制脉冲序列，以便在神经形态硬件上部署。 这种方法优化了脉冲神经元的发放模式，为非常深的网络带来了更稳定、更高效的训练。通过利用SFA，作者成功地将脉冲驱动架构扩展到1.73亿参数，并在ImageNet-1k基准测试中取得了86.2%的SOTA top-1准确率，极大地缩小了SNN与ANN之间长期存在的性能差距。该模型是Meta-SpikeFormer架构的明确升级，代表了在追求高性能、高能效AI道路上的一个重要里程碑。

## 4 统计一览

常被用于在实验部分作为Baseline的工作在表格中被加粗。

| 模型名称 | 会议与年份 | 论文链接 | 开源代码链接 | 描述与关键创新 | 继承/改进关系 |
|------------|--------------|------------|-----------------------|-------------------------------|-----------------------------------|
| Spiking Transformers for Event-Based Single Object Tracking (STNet) | CVPR 2022 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Spiking_Transformers_for_Event-Based_Single_Object_Tracking_CVPR_2022_paper.pdf) | [GitHub](https://github.com/Jee-King/CVPR2022_STNet) | 引入了用于事件驱动跟踪的脉冲Transformer，融合时间（SNN模块）和空间（Transformer）信息；动态调整SNN阈值，以在数据集如FE240hz和EED上实现更好的准确性。 | 早期模型；未提及直接前代，但基于事件数据的通用SNN和Transformer概念构建。 |
| **Spikformer: When Spiking Neural Network Meets Transformer** | ICLR 2023 | [Paper](https://openreview.net/pdf?id=frE4fUwz_h) | [GitHub](https://github.com/ZK-Zhou/spikformer) | 提出脉冲自注意力（SSA），使用脉冲形式的Query/Key/Value而不使用softmax，以实现稀疏、高效计算；在ImageNet上实现74.81%的准确率，能量消耗低。 | 基础模型；启发了后续许多模型如Spike-driven Transformer。 |
| Masked Spiking Transformer (MST) | ICCV 2023 | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Masked_Spiking_Transformer_ICCV_2023_paper.pdf) | [GitHub](https://github.com/bic-L/Masked-Spiking-Transformer) | 使用ANN-to-SNN转换结合随机脉冲掩码（RSM）来修剪脉冲，在75%掩码比率下降低能量26.8%，而不损失性能。 | 通过受突触失败启发的掩码，改进了Spikformer和STNet的硬件兼容性和能量问题。 |
| **Spike-driven Transformer** | NeurIPS 2023 | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf) | [GitHub](https://github.com/BICLab/Spike-Driven-Transformer) | 完全脉冲驱动设计，带有脉冲驱动自注意力（SDSA），仅使用加法；在ImageNet上实现77.1%，能量比香草注意力低87.2倍。 | 通过消除MAC操作并确保二进制脉冲通信，改进了Spikformer。 |
| **Spike-driven Transformer V2 (Meta-SpikeFormer)** | ICLR 2024 | [Paper](https://openreview.net/pdf?id=1SIBN5Xyw7) | [GitHub](https://github.com/BICLab/Spike-Driven-Transformer-V2) | 元架构，探索脉冲驱动自注意力变体；支持分类、检测和分割；在ImageNet上80.0%。 | Spike-driven Transformer（V1）的直接扩展，提升了多功能性和性能，优于基于Conv的SNN。 |
| Spikeformer: Training high-performance spiking neural network with transformer | Neurocomputing 2024(arxiv 2022)| [Paper](https://dl.acm.org/doi/10.1016/j.neucom.2024.127279) | [Github]() | 设计了卷积标记器 (CT) 模块，将时空注意力机制（STA）集成到Spikeformer中 | 出版较晚，在2022就已提出，属于直接源自Transformer的早期模型。 |
| DISTA: Denoising Spiking Transformer with Intrinsic Plasticity and Spatiotemporal Attention | ICLR 2024 | [Paper](https://openreview.net/pdf?id=mjDROBU93g) | N/A | 引入神经元级和网络级的时空注意力结合去噪；在CIFAR10上实现96.26%。 | 通过添加时间注意力和可塑性，构建于Spikformer之上，以克服仅空间限制。 |
| STFormer: Spatial Temporal Spiking Transformer | ICLR 2024 | [Paper](https://openreview.net/pdf?id=wPK65O4pqS) | N/A | 具有空间和时间核心用于特征提取，带有脉冲引导注意力；在CIFAR10-DVS上SOTA（83.1%）。 | 通过优化LIF神经元放置和时空处理，改进了Spikformer和Spike-driven。 |
| TIM: An Efficient Temporal Interaction Module for Spiking Transformer | IJCAI 2024 | [Paper](https://www.ijcai.org/proceedings/2024/0347.pdf) | [GitHub](https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/TIM) | 基于卷积的模块增强时间处理；集成到Spikformer骨干中，在神经形态数据集上SOTA。 | 通过最小参数解决时间数据限制，构建于Spikformer之上。 |
| Memory-Efficient Reversible Spiking Neural Networks (RevSFormer) | AAAI 2024 | [Paper](https://ojs.aaai.org/index.php/AAAI/article/download/29616/31147) | N/A | 可逆架构用于内存效率；RevSFormer变体在CIFAR上降低GPU内存3倍。 | 通过添加时间可逆性，改进了通用脉冲Transformer以实现更深模型。 |
| Complex Dynamic Neurons Improved Spiking Transformer Network (DyTr-SNN) | AAAI 2023 | [Paper](https://ojs.aaai.org/index.php/AAAI/article/download/25081/24853) | N/A | 通过复杂神经元动态增强ASR；降低错误率和计算。 | 通过整合生物动态，改进了先前的基于SNN的Transformer用于语音。 |
| QKFormer: Hierarchical Spiking Transformer using Q-K Attention | NeurIPS 2024 | [Forum](https://openreview.net/forum?id=AVd7DpiooC) | [GitHub](https://github.com/zhouchenlin2096/QKFormer) | 线性复杂度Q-K注意力结合分层结构；在ImageNet上85.65%。 | 通过多尺度脉冲和变形捷径，比Spikformer改进10.84%。 |
| Spiking Transformer with Experts Mixture | NeurIPS 2024 | [Forum](https://openreview.net/forum?id=WcIeEtY3AG) | N/A | 脉冲驱动专家混合（SEMM）用于稀疏计算；在神经形态数据集上改进。 | 通过专家路由用于头/通道级效率，构建于基础脉冲Transformer之上。 |
| Spiking Token Mixer (STMixer) | NeurIPS 2024 | [Abstract](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e8c20cafe841cba3e31a17488dc9c3f1-Abstract-Conference.html) | [GitHub](https://github.com/brain-intelligence-lab/STMixer_demo) | 事件驱动结构结合conv/FC层；在低时间步匹配脉冲Transformer。 | 比先前脉冲Transformer改进了异步硬件兼容性。 |
| Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention | CVPR 2025 | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Spiking_Transformer_Introducing_Accurate_Addition-Only_Spiking_Self-Attention_for_Transformer_CVPR_2025_paper.pdf) | N/A | A²OS²A机制结合混合神经元；在ImageNet上78.66%。 | 通过减少注意力中的信息损失，改进了Spikformer和Spike-driven。 |
| Spiking Transformer with Spatial-Temporal Attention (STAtten) | CVPR 2025 | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Spiking_Transformer_with_Spatial-Temporal_Attention_CVPR_2025_paper.pdf) | [GitHub](https://github.com/Intelligent-Computing-Lab-Yale/STAtten) | 块状时空注意力；在CIFAR和神经形态数据上提升性能。 | 通过整合时间依赖，改进了Spikformer和Spike-driven V1/V2。 |
| Rethinking Spiking Self-Attention: α-XNOR Similarity Calculation | CVPR 2025 | [Paper](https://ieeexplore.ieee.org/abstract/document/11095114) | N/A | α-XNOR用于脉冲中的更好相似性；增强多个架构。 | 通过修复脉冲查询/键中的点积问题，改进了Spikformer等。 |
| Spike2Former: Efficient Spiking Transformer for High-performance Image Segmentation | AAAI 2025 | [Paper](https://arxiv.org/abs/2412.14587) | [GitHub](https://github.com/biclab/spike2former) | 归一化整数神经元用于稳定性；在ADE20K/Cityscapes上SOTA mIoU。 | 改进了先前Transformer的转换方法用于分割任务。 |
| Spiking Transformer-CNN (Spike-TransCNN) | ICLR 2025 (reject) | [Paper](https://openreview.net/forum?id=zweyouirw7) | N/A (supp. mentioned) | 用于事件驱动检测的混合；在Gen1上更高mAP和更低能量。 | 首个CNN与脉冲Transformer的混合；改进了局部/全局特征融合。 |
| Quantized Spike-driven Transformer (QSD-Transformer) | ICLR 2025 | [Paper](https://arxiv.org/abs/2501.13492) | [Github](https://github.com/bollossom/QSD-Transformer)  | 提出了一种复杂的双层优化策略。在底层，设计了一个信息增强型Leaky Integrate-and-Fire（IE-LIF）神经元来校正信息分布。在顶层，采用了一种细粒度蒸馏（Fine-Grained Distillation, FGD）方案，以使量化模型的行为与其全精度对应模型对齐。 | 将量化技术引入脉冲驱动谱系，旨在通过使用低比特宽度的参数来显著减少模型的内存占用和计算需求。

#### 4.1 备注

Spike-driven Transformer V2 (Meta-SpikeFormer)，ICLR 2024，与**SpikFormer(Zhou et al., 2023)** 和**Spike-driven Transformer(Yao et al., 2023b)** 作了比较。

Spike2Former: Efficient Spiking Transformer for High-performance Image Segmentation，AAAI 2025，与**Spike-driven Transformer V2**，作了比较

Spiking Transformer-CNN (Spike-TransCNN)，在ICLR2025被拒，但审稿人提到的问题主要是行文问题和创新程度问题，肯定了本文在目标识别数据集上取得了优异的成果（“这似乎却决于模块堆栈的数量”）。这篇文章的实验部分没有与竞争对手——各种改进型spiking transformer比较，与较新模型的对比仅仅是一些YOLO。

Quantized Spike-driven Transformer (QSD-Transformer)，ICLR 2025，与**SpikFormer(Zhou et al., 2023)** 和**Spike-driven Transformer(Yao et al., 2023b)**及一些传统模型作了比较。