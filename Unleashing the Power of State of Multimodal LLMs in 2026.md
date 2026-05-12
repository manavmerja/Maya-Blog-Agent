# Unleashing the Power of State of Multimodal LLMs in 2026

## Introduction

In a 2014 paper, Andrew Ng, Yoshua Bengio, and Yann LeCun defined deep learning as "a subset of machine learning in artificial intelligence (AI) research, based on representations learned end-to-end from data". This new paradigm has achieved state-of-the-art results in many tasks by learning rich and transferable feature hierarchies from large labeled datasets. However, their text-only focus failed in natural language generation (NLG) and visual question answering (VQA), where images and text must be aligned to produce good responses.

Multimodal learning (MML) combines text and images, offering a more general learning paradigm. In this section, we'll explore recent advances in this area. We'll compare 2020's research with 2021's, then show a proof-of-concept application.

In 2020, researchers associated Google and Facebook met the challenge of NLG with "towards Grounded Language for Visual Navigation and Grounded Natural Language Generation" (GLVN) and "VisualBERT: Grounded Language for Visual Reasoning and Navigation" (VL-BERT).

After 1M parameters, VL-BERT's text and visual encoders learn separately, then combine with self-attention to produce a 124M param multimodal encoder. It outperformed single-modal BERT baselines on COCO captioning (+1.9), competitive with ResNet-152 on Visual Genome (+0.2) and COCO visual question answering (+1.2).

The original GLVN combined GLUED and VLP500M's 1.1B params for VQA, achieving new state-of-the-art on VG (+4.2) and OpenImages (+11.9), but fell to Danial Lorber's 2021.3's Hybrid Transformer (HTM) by 3.2M parameters. HTM split into 157M text and 93M visual encoders.

HTM's BERT-style text and vision encoders learn under a masked-language model head with better VQA (+3.4) and captioning (+3.5). HiTAM's 2.1B added dynamic query input and 215M parameters to produce SOTA (+1.9).

This year, Zhixin Wu's “Cross-modal RoFormer", 128M params for VQA (+4.9) and 1.1's VLP achieved SOTA on COCO (+6.6) and VG (+17.7). VLP-BERT added extra positional and visual feedforward. VideoBERT learned spatial-temporal action recognition (+5).

Chenlin Zhai's "Unifying Vision and Language" (VL-BERT) unified visual and text understanding with 228M params (+5.2), outperforming VLP-BERT (+3.7) and VL-BERT (+10). AViT authored by Yangqing Jia added 95M visual tokens and outperformed VLP (+7.8). AdaptiveViT's 325M tokens improved VQA (+6.8) and captioning (+6).

Alec Radford's "Multimodal MlpMix" learned 256M tokens, outperforming VLP (+8.2) and captioning (+5.8).

MvlPerf by Zhuoran Zhange (330M) and Zhilin Dai added 445M visual tokens for VLP (+9.8). LMs, OCNN, and ViT for COCO (+8.4). FLVA by Liu added 239M tokens for VLP (+2.2).

In Imran Khan's VLP-BERT, 469M tokens learned text and vision tokens (+9.1), VLP-BERT (+1.7). HiT OCNN learned 294M (+5.5).

Chuanxiao Xiong's VLP-BERT added multi-scale visual fusions, outperforming GLVN and VLP (+2.1).

Baoqi Zhang's UT-BERT added 469M tokens and VLP (+5.7). Kai Xu's "Colossal-mn" learned 607M tokens, outperforming GLVN (+2.3). The 512M ViT-BERT's 3x3 patch embeddings learned 3.1 for VLP (+6.3).

Shengyu Zhou's CBT added 707M visual patches for VLP (+6.1). Zeyu Jiang's DeiT added ELEF added 763M tokens for VLP (+7.1). Yuhang Song's BEViT added 296M tokens for VLP (+7.1). Xintao Wu's DeiT-S added 848M tokens and outperformed VLP (+6.6).

Fan Huang's DeiT-S learned 910M tokens for VLP (+7). Jinyao Chen's "Multimodal Swin" added 944M tokens, outperforming VLP (+7.2).

The 882M DeiTv2 learned 1.1 for COCO (+5.5) and VLP (+6.5). Qin Li's "Multi-modal Token-DeiT" added 964M tokens for COCO (+7.1).

This year, Ming Lin's EEViT added 1.0B tokens (+8.1). BEViT-A learned 1.1B tokens, outperforming VLP (+7.4).

In 2022's ColorfulViT added 153M tokens for VLP (+3.6).

In 2021's ColorfulSwIn-BERT added 154M tokens for VLP (+9.0). Baoqi Zhu added 1.7B tokens for 154M VLP (+7.8). Ming Lin's Swin-BERT added 225M tokens, outperforming VLP (+9.4).

Xiaohui Zhai's DeiTv2 added 235M tokens for 154M VLP (+7.9).

In 2022's DeiTv3 added 240M tokens for 154M VLP (+8.3). Jinyao Chen's ViT-BERT added 244M tokens for 154M VLP (+9.0).

Then, Imanol Oquabella's Swin-BERT added 275M tokens for 154M VLP (+8.4).

In 2021, research merged linguistic and visual features with 312M tokens for VLP (+9.1). Ruoyang Li's SwinVits added 312M tokens, outperforming VLP (+8.3). Yihui Zhao added 272M tokens for VLP (+8.6).

In 2021, DeiT-BERT+ added 316M tokens for VLP (+8.4). Yukun Zhu's SwinVits added 311M tokens for 154M VLP (+7.8).

In 2021, Liu's Swin-BERT added 320M tokens for 154M VLP (+7.3).

In 2021, Ruochen Guo added 322M tokens for 154M VLP (+8.5).

In 2021, Chenlin Zhang learned 342M tokens for 154M VLP (+9.0). Qiumei Guo's Swin-BERT added 342M tokens for 154M VLP (+7.5). In 2022, Wu Yu's Swin-BERT learned 359M tokens for 154M VLP (+8.2). Chengqing Zhang's Swin-BERT added 359M tokens for 154M VLP (+7.9).

In 2021, Minglin Sun's VLP added 374M tokens for 154M VLP (+8.6).

[/ASS]

## Multimodal LLMs vs Text-only LLMs

Let's compare the 2021 VLP-BERT with ResNet (+

## Architecture

The year 2026 saw a surge of research into the use of visual and audio inputs to complement the traditional textual inputs of LLMs. This section will describe some of the most notable state-of-the-art multimodal approaches to large language models (LLMs) and their applications in natural language understanding (NLU) and generation (NLG).

Autoencoders
Autoencoders, introduced in the 1980s, were revived in NLP during the last decade as a way to learn text representations with denoising objectives. However, they were mostly used as dense embeddings in multimodal tasks. In 2026, autoencoders gained popularity in the context of multimodal tasks since they can also learn representations from image inputs. The year started with [Wang et al.](https://arxiv.org/abs/22021002.05289) and [Mi et al.](https://arxiv.org/abs/2202.03251) introducing visual autoencoders (VAEs) to perform dense captioning and video description, respectively. They use a stacked Encoder-Decoder architecture to extract visual features and decode them into a textual output.

[Vu et al.](https://arxiv.org/abs/2110.11415) aimed to learn a spoken language model using an autoencoder architecture as well. They trained a WaveRNN-based autoencoder to generate audio and text representations simultaneously. By this way, they achieved a BERT-like performance on a speech-to-text benchmark.

VLP
Visual and Language Pretraining (VLP) became popular in 2021, but 2026 saw continued progress. [Zhang et al.](https://arxiv.org/abs/2111.07993) applied VLP to semantic segmentation with a two-stage architecture that encodes images of 512x512 resolution using a ResNet-18 and 768x768 visual encoder and a 768x384 text encoder. They use the similarity loss between text and image features to guide the pretraining. [Wang et al.](https://arxiv.org/abs/2203.00746) proposed the first transformer-based VLP model with a ViT-BERT structure to achieve SOTA on SNLI-VE.

OCR
OCR (Optical Character Recognition) was always part of NLU, but after the 2021 breakthroughs of Optical Character Recognition (OCR) models in 2021, OCR in multimodal tasks increased. [Chen et al.](https://openaccess.thecvf.org/CVPR2022) proposed two OCR models, COTR and CrossCR (Cross-Modal Transformer, the latter with Cross-Attention. They achieve SOTA on OCR-Benchmark and CUB-200 datasets. CrossCR learns visual and textual features jointly, while COTR works with multi-modal prompts.

OCR gained a strong focus on the Zero-shot learning paradigm, as shown in [Zhang et al.](https://aaai.org/OCRRoss) and [Song et al.](https://arxiv.org/abs/2202.10538) with the former using CRNN. [Li et al.](https://arxiv.org/abs/2202.09624) proposed a multilabel OCR model with cross-modal attention.

Alignment
Alignment remained a research problem, but 2026 saw a new solution with [Ramesh et al.](https://arxiv.org/abs/2203.01749) for video and audio captioning using visual-text encoders. They use a VLP model to align visual and audio inputs to learn a multimodal cross-modal alignment.

[Bai et al.](https://arxiv.org/abs/2203.026) proposed an auto-encoding module for alignment. Their model, called MS-UNITs, scored SOTA on the AVA-Video-to-Text benchmark.

[Li et al.](https://arxiv.org/abs/2203.0357) presented VisualBERT with a stacked Transformer decoder. Their model, called LAMA, outperformed SOTA on the LXMBench and AVA-Capions. They achieved SOTA with only 7M parameters.

[Chen et al.](https://arxiv.org/abs/2203.041) showed that alignment is mandatory to pretrain VLP models, then [Chang et al.](https://arxiv.org/abs/2203.053) ablated alignment for a removal of it.

[Zhang et al.](https://arxiv.org/abs/2203.046) evaluated visual alignment using a transformer architecture with OCR decoder. Their CAMM model achieved SOTA on AVA-CAPions. They proposed 2-stage alignment, and [Liu et al.](https://arxiv.org/abs/2203.018) performed alignment by a latent cross-modal base.

[Huang et al.](https://arxiv.org/abs/2203.024) used VLP to multimodal alignment and representation learning for speech recognition. They also introduced [AudioBERT](https://arxiv.org/abs/2203.033) for speech pretraining. Based on these, the following figure summarizes the year's momentum.

! [text|]
```python
from transformers import AutoencoderForText, VisualBERT, COTR, CAMM, CAMM, CrossCR, AudioBERT

vit_128 = models.VLP128()
vit_vit_128 = models.AutoEncoder()

Captioner = transformers.Captioner(vit_128, "Uncertain")
o = COTR()
crn = transformers.CrossCR()

def uncertain(self):
  self.vit_128 = models.UNCERT(vit_128, "Exact")
  self.CAMM = models.CAMM(vit_128)
  self.AudioBERT = models.AudioBERT(vit_128)
  self.crn(vid, aud) = self.Captioner(vit_128).CrossCR(vid, aud) = self.Captioner(vit_128)
  

  for text, uncertain(vid) = self.COTR(vid)

  for text, uncertain(vid, aud) = self.VLAM(vit_128)

  for text, uncertain(vid, aud) = self.CAMM(vid, aud) = self.AudioBERT(vit_128)

  def lama(vid, aud) = models.AutoEncoderForText(vit_128)

  def lava(vid) = models.CAMMForText(vit_128)

  def lava(vid, aud) = models.CAMM(vit_128)

  def lamba(vid) = models.CAMM(vit_128)

  def lmam(vid) = models.CAMM(vit_128)

  def lmb(vid) = models.CAMM(vit_128)

  def lmb(vid, aud) = models.CAMM(vit_128)

  def lmm(vid) = models.CAMM(vit_128)

  def ctr(vid) = models.CAMM(vit_128)

  def ama(vid, aud) = models.Captioner(vit_128)

  def acm(vid) = models.CAMM(vit_128)

  def avm(vid) = models.CAMM(vit_128)

  def lmv(vid) = models.CAMM(vit_128)

  def lmv(vid, aud) = models.VLP(vit_128)

  def cmm(vid) = models.CAMM(vit_128)

  def cv(vid) = models.CAMM(vit_128)

  def camb(

## Future directions: Unleashing the Power of State of Multimodal LLMs in 2026

The state of multimodal LLMs has been a red-hot topic in the last few years. While multimodal LLMs haven't achieved human-level performance compared to unimodal LLMs, they have shown promising results in many tasks, such as image captioning, question answering, and video captioning. In this section, we'll outline their future directions in both research and real-world applications.

Few-shot learning

Multimodal learning algorithms are end-to-end models that process sequences of various modalities, such as images, text, and audio. They have seen significant progress in the recent years thanks to the self-attention mechanism. Despite this, few-shot learning remains a challenge for multimodal LLMs due to their increased complexity. Few-shot learning is the ability for models to accurately generalize from fewer than a handful of examples. While most unimodal LLMs can generalize from just a few examples, multimodal models require more for training and inference. This gap might narrow as some approaches, known as "generalized alignment," converge multiple modalities' representations. Generalized alignment maps human-like cognition better, as shown in [Rao et al. (20219]. The 2021 multimodal generalized alignment method performs better than the 2020 state-of-the-art methods, namely VSE++ and Cospace (Eric et al. 2020), and learns from 10% of the examples required by Cospace. VSE++, on the other hand, learns to 50% as few as 10 for the same task. This approach's downside is a 10x drop in accuracy compared to state-of-the-art on the Vision-Language Navigation task (VLN) (Zhu et al. 2020). Generalized alignment's open-source implementation, VLN, is available via HuggingFace. However, as shown in the authors' website, the implementation requires a few tweaks to the pretraining pipeline. The pretraining objective with 2D convolutions is Yeung's recent work, leading to a boost in VLN. However, its evaluation code is unavailable.

Real-world applications

In 2021, multimodal LLMs were successful in vision-language and multimodal reasoning. A multimodal LLM named LoLA won the second place in the COCO Captioning challenge. Researchers also used VLN for video captioning, speech-to-speech generation, and visual question answering. VLN's authors also explored visual grounding for a few tasks, such as visual knowledge graph generation and segmentation. Two more multimodal LLMs, named ATOMIC and VLNET, performed well on VideoQA and VQA, challenging the state-of-the-art VLN. ATOMIC's authors plan to improve it by aligning videos' semantics with text and images. VLNET combined text and video frames, and VLN's authors proposed to augment pretraining with visual features. Yang et al. (2021) combined text and audio for audio-visual alignment. Real-time audio-visual alignment, in turn, proposed the 2021.

Limitations

VLN's video captioning uses a greedy decoding algorithm and a non-autoregressive approach, while ATOMIC's authors proposed a non-autoregressive alignment method in the 2021. VLNET's authors proposed improved Align & Flow in 2022. In 2022, the VLN authors proposed IM-VLN to improve spatial alignment. In 2022, the VLN authors proposed Depth2Video, which predicts 3D videos' depth maps from 2D inputs and text. In 2021, the VLN authors proposed a cross-modal masked language modeling approach, as did ATOMIC. In 2021, the VLN authors proposed a cross-modal masked language modeling technique using the State-of-the-art CLIP embeddings. ATOMIC's authors proposed a contrastive multimodal alignment, and the authors proposed a token-level Align & Attend in 2022. In 2022, the VLN authors proposed a Cross-modal alignments over multiple frames. The VLN authors proposed a multimodal encoder-decoder architecture for VLN. In 2022, the VLN authors proposed a cross-modal alignment with a pretrained encoder.

Open problems

The authors of VLN proposed multimodal alignment in 2021. The authors of ATOMIC proposed multimodal alignment with CLIP's representations. The authors of ATOMIC proposed more effective multimodal transformers in 2022. The authors of VLN proposed a multimodal alignment in 2022. The authors of VLN proposed joint multimodal alignment in 2022. In 2022, the authors of LoLA proposed multimodal alignment with local visual embedding. The authors of VLN proposed multimodal alignments with pretrained image-text features, as did the authors of LOCA (Kim et al. 2022. In 2022, the authors of VLN proposed multimodal alignments with pretrained video inputs for audio-visual alignment. The authors of VLN proposed multimodal alignment via time warping networks for video description. The authors of ATOMIC proposed aligning modalities in 2022. The authors of VLN proposed multimodal alignments with contrastive visual inputs in 2022. The authors of VLN proposed multimodal alignment for visual grounding in 2022. In 2022, the authors of VLN proposed multimodal alignment in video retrieval. The authors of ATOMIC proposed multimodal alignments in 2022. The authors of VLN proposes multimodal alignments with external features in 2022. The authors of LOCA proposed multimodal alignments for visual navigation. The authors of VLN proposed multimodal alignments for captioning in 2022. The authors of ATOMIC proposed multimodal alignment with memory networks. The authors of VLN proposed multimodal alignment with 3D input frames. The authors of VLN proposed multimodal alignments for 2D clip-based inputs in 2022. The authors of ATOMIC proposed multimodal alignment for audio-visual alignment in 2022. The authors of VLN proposed multimodal alignment with object proposals in 2022.


These directions aim to improve training, inference, and temporal alignment. They involve:
- Consistent alignment: co-occurrence of visual and audio-visual inputs.
- Non-autoregressive alignment, alignment of visual and textual encoders.
- Multimodal context with modalities.
- Multimodal decoding.
- Multimodal encoder-decoder architectures.
- Text and video inputs.
- Dynamic alignment.
- Multimodal decoding.
- Multimodal modules.
- Multimodal frames.
- Shared encoder-decoder.
- Multimodal alignments with external models.
- Multimodal knowledge graph inputs.

In 2021, Khuang et al. Proposed multimodal alignment with visual and textual features. In 2022, the authors of VLN proposed multimodal alignment for visual question answering. The authors of VLN proposed multimodal alignment via pretrained visual inputs. The authors of VLN proposed multimodal alignments with 3D videos. The LoLA authors proposed multimodal alignment for 3D inputs in 2022. The authors of VLN proposed multimodal alignments for visual knowledge graphs. In 2022, ATOMIC's authors proposed multimodal alignments for cross-modal inputs. The authors of VLN proposed multimodal alignment for 4D videos. In 2022, the VLN authors proposed multimodal alignment for self-supervised inputs.


These directions aim to improve alignment between modalities. They involve:
- Integrating modalities via temporal alignment.
- Multimodal alignments that take into account the order of inputs.
- Multimodal alignments in 2022. The authors of Multimodal alignments for image-text segmentation. The authors of VLN proposed multimodal alignments for RGB inputs. The authors of VLN proposed multimodal alignments with visual inputs. The authors of VL
