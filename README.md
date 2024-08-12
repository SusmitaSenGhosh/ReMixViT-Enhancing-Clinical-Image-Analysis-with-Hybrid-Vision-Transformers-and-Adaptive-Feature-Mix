# ReMixViT-Enhancing-Clinical-Image-Analysis-with-Hybrid-Vision-Transformers-and-Adaptive-Feature-Mix

# ReMixViT: Enhancing Clinical Image Analysis with Hybrid Vision Transformers and Adaptive Feature Mixing
This repository Keras implementation of the experimets conducted in '*ReMixViT: Enhancing Clinical Image Analysis with Hybrid Vision Transformers and Adaptive Feature Mixing**'. Codes are verified on python3.8.5 with tensorflow version '2.4.1'. Other dependencies are NumPy, cv2, sklearn, matplotlib, random, os, etc.

**Data Resources:**

1. Colorectal Histology: https://www.kaggle.com/kmader/colorectal-histology-mnist
2. ISIC18: https://challenge2018.isic-archive.com/task3/
3. CBIS_DDSM: https://www.tensorflow.org/datasets/catalog/curated_breast_imaging_ddsm
4. Chestxray: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
5. Fundus: https://github.com/deepdrdoc/DeepDRiD
6. PBC: https://data.mendeley.com/datasets/snkd93bnjr/1

**Data preparation:** For data preparation use data_prep.py.

**Training and Evaluation:**
1. ViT/ReViT/MixViT/ReMixViT/ResNet50/Res-ViT/Res-ReMixViT: Execute train_and_test_models.py
2. Res-ReMixViT+: Refer to train_and_test_models_aux.py
3. To plot ACSF curve for training and testing of ViT and ViT-R-MM use ViT_ViTRMM_graph.py.
4. For visual interpretation of ViT/ReMixViT, ResNet50/Res-ViT/Res-ReMixViT and Res-ReMixViT+ use view_gradients_and_attention_maps.py, view_gradients_and_attention_maps_hybrid.py and view_gradients_and_attention_maps_aux.py respectively.
