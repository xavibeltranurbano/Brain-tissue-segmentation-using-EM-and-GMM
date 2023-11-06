# Brain tissue segmentation using Expectation Maximization (EM) algorithm for Gaussian Mixture Models (GMM)

This repository contains the implementation and results of a novel approach to brain tissue segmentation, combining the Expectation Maximization (EM) algorithm with Gaussian Mixture Models (GMM). Developed at the University of Girona as part of the Erasmus Mundus Joint Master Degree in Medical Imaging and Applications (MAIA), this method aims to enhance the accuracy of medical diagnoses and treatment planning for neurological conditions.

## Introduction

Medical image segmentation is a critical step in the diagnosis and treatment of neurological diseases. Precise segmentation of brain tissues such as cerebrospinal fluid (CSF), white matter (WM), and gray matter (GM) from magnetic resonance (MR) images plays a pivotal role in quantitative brain analysis. Traditional techniques face challenges in accuracy and consistency, which this project aims to overcome through probabilistic clustering methods. By leveraging the EM algorithm in conjunction with GMM, we can assign membership weights to voxels and accurately segment brain tissues, potentially aiding in the identification and treatment of conditions like Alzheimer's and Parkinson's disease.

## Dataset

The segmentation process was conducted using a multimodal dataset comprising 5 MRI images, each with the following modalities:

- T1-weighted (T1)
- FLAIR-weighted (FLAIR)
- Ground Truth (GT)

These modalities offer complementary information about brain structures, essential for achieving a comprehensive segmentation. The T1 modality provides detailed anatomical information, FLAIR highlights fluid and lesion differentiation, and the Ground Truth data serve as a benchmark for assessing segmentation accuracy.

## Methodology

Our approach utilizes the EM algorithm in two primary steps:

1. **Expectation Step**: Here, we assign a membership weight to each voxel, indicating the likelihood of it belonging to a particular tissue cluster.
2. **Maximization Step**: In this step, we update the parameters of our Gaussian models — means, covariances, and mixture weights — based on the current membership assignments.

The model initializes with either random parameters or using KMeans clustering for a more informed starting point. The iterative process of expectation and maximization continues until convergence is achieved, indicated by the log-likelihood reaching a plateau.

## Results

The algorithm was rigorously tested across the dataset, with the performance measured using the Dice Score (DSC). The DSC is a statistical validation metric to quantify the similarity between the predicted segmentation and the ground truth. Our model demonstrated strong performance with DSC scores reaching 0.85, 0.80, and 0.81 for White Matter, Gray Matter, and Cerebrospinal Fluid, respectively.

The results were on par with other segmentation tools like K-Means and SPM, and in some cases, our EM-GMM approach outperformed these established methods. The table below summarizes the performance metrics obtained:

| Tissue Type | DSC Score |
|-------------|-----------|
| White Matter (WM) | 0.85 |
| Gray Matter (GM) | 0.80 |
| Cerebrospinal Fluid (CSF) | 0.81 |

For a more detailed analysis and visual representation of the segmentation results, please refer to the figures and tables included in this repository.

## Conclusion

In conclusion, the application of the Expectation Maximization algorithm with Gaussian Mixture Models has proven to be an effective strategy for the segmentation of brain tissues in MRI images. Our methodology not only aligns with the stringent requirements of medical image processing but also showcases the potential to facilitate early diagnosis and treatment planning for neurodegenerative diseases.

The multimodal dataset utilized in this study underscored the robustness of our approach, with the algorithm achieving Dice Score metrics indicative of high segmentation accuracy. Specifically, the scores of 0.85 for White Matter, 0.80 for Gray Matter, and 0.81 for Cerebrospinal Fluid reflect the precision of our model in distinguishing between the complex structures within the brain.

Moreover, when compared to traditional segmentation tools such as K-Means and Statistical Parametric Mapping (SPM), our EM-GMM approach demonstrated competitive, if not superior, performance. This suggests that the integration of probabilistic modeling with clustering algorithms can significantly enhance the outcome of tissue segmentation tasks.

We believe that the insights and findings from our work have substantial implications for the future of medical imaging and the treatment of neurological conditions. As we continue to refine our algorithms and validate our models across larger datasets, we remain committed to advancing the frontier of medical image analysis and improving patient care outcomes.

The journey of this project from concept to implementation has been thoroughly documented within this repository. We encourage researchers, clinicians, and developers to explore our codebase, engage with our findings, and contribute to the ongoing evolution of this promising field.
