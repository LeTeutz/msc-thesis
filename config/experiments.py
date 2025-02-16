from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class ExperimentConfig:
    name: str
    train_dataset: str
    test_dataset: str
    train_transform: Optional[str] = None
    test_transform: Optional[str] = None
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
    adaptation_method: Optional[str] = None

EXPERIMENTS = {
    "mnist_baseline": ExperimentConfig(
        name="MNIST 1K Baseline",
        train_dataset="mnist_100",
        test_dataset="mnist_100",
        train_transform="identity",
        test_transform="identity"
    ),

    # Basic blurs (7)
    # "mnist_light_gaussian": ExperimentConfig(
    #     name="MNIST 1K Light Gaussian",
    #     train_dataset="mnist_100",
    #     test_dataset="mnist_100",
    #     train_transform="light_gaussian",
    #     test_transform="identity"
    # ),
    # "mnist_medium_gaussian": ExperimentConfig(
    #     name="MNIST 1K Medium Gaussian",
    #     train_dataset="mnist_100",
    #     test_dataset="mnist_100",
    #     train_transform="medium_gaussian",
    #     test_transform="identity"
    # ),
    # "mnist_heavy_gaussian": ExperimentConfig(
    #     name="MNIST Heavy Gaussian",
    #     train_dataset="mnist_100",
    #     test_dataset="mnist_100",
    #     train_transform="heavy_gaussian",
    #     test_transform="identity"
    # ),
    # "mnist_very_heavy_gaussian": ExperimentConfig(
    #     name="MNIST Very Heavy Gaussian",
    #     train_dataset="mnist_100",
    #     test_dataset="mnist_100",
    #     train_transform="very_heavy_gaussian",
    #     test_transform="identity"
    # ),
    "mnist_median": ExperimentConfig(
        name="MNIST 1K Median",
        train_dataset="mnist_100",
        test_dataset="mnist_100",
        train_transform="median",
        test_transform="identity"
    ),
    "mnist_box": ExperimentConfig(
        name="MNIST 1K Box",
        train_dataset="mnist_100",
        test_dataset="mnist_100",
        train_transform="box",
        test_transform="identity"
    ),
    "mnist_motion": ExperimentConfig(
        name="MNIST 1K Motion",
        train_dataset="mnist_100",
        test_dataset="mnist_100",
        train_transform="motion",
        test_transform="identity"
    ),

    # # Blur + inversion combinations (6)
    # "mnist_light_gaussian_inv": ExperimentConfig(
    #     name="MNIST 1K Light Gaussian + Invert",
    #     train_dataset="mnist_100",
    #     test_dataset="mnist_100",
    #     train_transform="light_gaussian_inv",
    #     test_transform="identity"
    # ),
    # "mnist_medium_gaussian_inv": ExperimentConfig(
    #     name="MNIST 1K Medium Gaussian + Invert",
    #     train_dataset="mnist_100",
    #     test_dataset="mnist_100",
    #     train_transform="medium_gaussian_inv",
    #     test_transform="identity"
    # ),
    # "mnist_heavy_gaussian_inv": ExperimentConfig(
    #     name="MNIST 1K Heavy Gaussian + Invert",
    #     train_dataset="mnist_100",
    #     test_dataset="mnist_100",
    #     train_transform="heavy_gaussian_inv",
    #     test_transform="identity"
    # ),
    "mnist_median_inv": ExperimentConfig(
        name="MNIST 1K Median + Invert",
        train_dataset="mnist_100",
        test_dataset="mnist_100",
        train_transform="median_inv",
        test_transform="identity"
    ),
    "mnist_box_inv": ExperimentConfig(
        name="MNIST 1K Box + Invert",
        train_dataset="mnist_100",
        test_dataset="mnist_100",
        train_transform="box_inv",
        test_transform="identity"
    ),
    "mnist_motion_inv": ExperimentConfig(
        name="MNIST 1K Motion + Invert",
        train_dataset="mnist_100",
        test_dataset="mnist_100",
        train_transform="motion_inv",
        test_transform="identity"
    ),

    # # Double blur combinations (3)
    # "mnist_double_gaussian": ExperimentConfig(
    #     name="MNIST 1K Double Gaussian",
    #     train_dataset="mnist_100",
    #     test_dataset="mnist_100",
    #     train_transform="double_gaussian",
    #     test_transform="identity"
    # ),
    "mnist_gaussian_median": ExperimentConfig(
        name="MNIST 1K Gaussian + Median",
        train_dataset="mnist_100",
        test_dataset="mnist_100",
        train_transform="gaussian_median",
        test_transform="identity"
    ),
    # "mnist_double_gaussian_inv": ExperimentConfig(
    #     name="MNIST 1K Double Gaussian + Invert",
    #     train_dataset="mnist_100",
    #     test_dataset="mnist_100",
    #     train_transform="double_gaussian_inv",
    #     test_transform="identity"
    # )
}

# EXPERIMENTS = {

    # "mnist_baseline": ExperimentConfig(
    #     name="MNIST-5K -> MNIST-5K",
    #     train_dataset="mnist_100",
    #     test_dataset="mnist_100",
    #     train_transform="identity",
    #     test_transform="identity"
    # ),

# # MNIST-5K Single Transformations
# "mnist_100_noise": ExperimentConfig(
#     name="MNIST-5K Noise -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="noise",
#     test_transform="identity"
# ),
# "mnist_100_contour": ExperimentConfig(
#     name="MNIST-5K Contour -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="contour",
#     test_transform="identity"
# ),
# "mnist_100_contrast": ExperimentConfig(
#     name="MNIST-5K Contrast -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="contrast",
#     test_transform="identity"
# ),
# "mnist_100_blur": ExperimentConfig(
#     name="MNIST-5K Blur -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="blur",
#     test_transform="identity"
# ),
# "mnist_100_invert": ExperimentConfig(
#     name="MNIST-5K Invert -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="invert",
#     test_transform="identity"
# ),

# # MNIST-5K Pairs
# "mnist_100_color_contour": ExperimentConfig(
#     name="MNIST-5K Color+Contour -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="color_contour",
#     test_transform="identity"
# ),
# "mnist_100_color_contrast": ExperimentConfig(
#     name="MNIST-5K Color+Contrast -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="color_contrast",
#     test_transform="identity"
# ),
# "mnist_100_color_noise": ExperimentConfig(
#     name="MNIST-5K Color+Noise -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="color_noise",
#     test_transform="identity"
# ),
# "mnist_100_contour_noise": ExperimentConfig(
#     name="MNIST-5K Contour+Noise -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="contour_noise",
#     test_transform="identity"
# ),
# "mnist_100_contrast_noise": ExperimentConfig(
#     name="MNIST-5K Contrast+Noise -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="contrast_noise",
#     test_transform="identity"
# ),

# # MNIST-5K Triples
# "mnist_100_color_contour_noise": ExperimentConfig(
#     name="MNIST-5K Color+Contour+Noise -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="color_contour_noise",
#     test_transform="identity"
# ),
# "mnist_100_color_contrast_noise": ExperimentConfig(
#     name="MNIST-5K Color+Contrast+Noise -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="color_contrast_noise",
#     test_transform="identity"
# ),
# "mnist_100_contour_contrast_noise": ExperimentConfig(
#     name="MNIST-5K Contour+Contrast+Noise -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="contour_contrast_noise",
#     test_transform="identity"
# ),

# # MNIST-5K Quadruple
# "mnist_100_color_contour_contrast_noise": ExperimentConfig(
#     name="MNIST-5K Color+Contour+Contrast+Noise -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="color_contour_contrast_noise",
#     test_transform="identity"
# ),

# # MNIST-5K All transforms
# "mnist_100_all_transforms": ExperimentConfig(
#     name="MNIST-5K All Transforms -> MNIST-5K",
#     train_dataset="mnist_100",
#     test_dataset="mnist_100",
#     train_transform="all_transforms",
#     test_transform="identity"
# ),

#     # MNIST-1k Single Transformations
# "mnist_1k_noise": ExperimentConfig(
#     name="MNIST-1k Noise -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="noise",
#     test_transform="identity"
# ),
# "mnist_1k_contour": ExperimentConfig(
#     name="MNIST-1k Contour -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="contour",
#     test_transform="identity"
# ),
# "mnist_1k_contrast": ExperimentConfig(
#     name="MNIST-1k Contrast -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="contrast",
#     test_transform="identity"
# ),
# "mnist_1k_blur": ExperimentConfig(
#     name="MNIST-1k Blur -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="blur",
#     test_transform="identity"
# ),
# "mnist_1k_invert": ExperimentConfig(
#     name="MNIST-1k Invert -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="invert",
#     test_transform="identity"
# ),

# # MNIST-1k Pairs
# "mnist_1k_color_contour": ExperimentConfig(
#     name="MNIST-1k Color+Contour -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="color_contour",
#     test_transform="identity"
# ),
# "mnist_1k_color_contrast": ExperimentConfig(
#     name="MNIST-1k Color+Contrast -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="color_contrast",
#     test_transform="identity"
# ),
# "mnist_1k_color_noise": ExperimentConfig(
#     name="MNIST-1k Color+Noise -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="color_noise",
#     test_transform="identity"
# ),
# "mnist_1k_contour_noise": ExperimentConfig(
#     name="MNIST-1k Contour+Noise -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="contour_noise",
#     test_transform="identity"
# ),
# "mnist_1k_contrast_noise": ExperimentConfig(
#     name="MNIST-1k Contrast+Noise -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="contrast_noise",
#     test_transform="identity"
# ),

# # MNIST-1k Triples
# "mnist_1k_color_contour_noise": ExperimentConfig(
#     name="MNIST-1k Color+Contour+Noise -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="color_contour_noise",
#     test_transform="identity"
# ),
# "mnist_1k_color_contrast_noise": ExperimentConfig(
#     name="MNIST-1k Color+Contrast+Noise -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="color_contrast_noise",
#     test_transform="identity"
# ),
# "mnist_1k_contour_contrast_noise": ExperimentConfig(
#     name="MNIST-1k Contour+Contrast+Noise -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="contour_contrast_noise",
#     test_transform="identity"
# ),

# # MNIST-1k Quadruple
# "mnist_1k_color_contour_contrast_noise": ExperimentConfig(
#     name="MNIST-1k Color+Contour+Contrast+Noise -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="color_contour_contrast_noise",
#     test_transform="identity"
# ),

# # MNIST-1k All transforms
# "mnist_1k_all_transforms": ExperimentConfig(
#     name="MNIST-1k All Transforms -> MNIST-1k",
#     train_dataset="mnist_1k",
#     test_dataset="mnist_1k",
#     train_transform="all_transforms",
#     test_transform="identity"
# ),
# }

# EXPERIMENTS = {

    # "emnist_baseline_t3a": ExperimentConfig(
    #     name="EMNIST -> EMNIST with T3A",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="t3a",
    #     epochs=2
    # ),

    # "mnist_baseline_memo": ExperimentConfig(
    #     name="MNIST -> MNIST with MEMO",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="memo",
    #     epochs=2
    # ),

    # "mnist_baseline_style": ExperimentConfig(
    #     name="MNIST -> MNIST with Test-Time Style Shift",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="style",
    #     epochs=2
    # ),




    # "mnist_baseline": ExperimentConfig(
    #     name="MNIST -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="identity",
    #     test_transform="identity"
    # ),
    # "mnist_color": ExperimentConfig(
    #     name="MNIST Color -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="colorize",
    #     test_transform="identity"
    # ),
    # "fashion_mnist_baseline": ExperimentConfig(
    #     name="Fashion-MNIST -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="identity",
    #     test_transform="identity"
    # ),
    # "fashion_mnist_color": ExperimentConfig(
    #     name="Fashion-MNIST Color -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="colorize",
    #     test_transform="identity"
    # ),
    # "emnist_baseline": ExperimentConfig(
    #     name="EMNIST -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="identity",
    #     test_transform="identity"
    # ),
    # "emnist_color": ExperimentConfig(
    #     name="EMNIST Color -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="colorize",
    #     test_transform="identity"
    # ),

    # # MNIST 
    # "mnist_noise": ExperimentConfig(
    #     name="MNIST Noise -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="noise",
    #     test_transform="identity"
    # ),
    # "mnist_contour": ExperimentConfig(
    #     name="MNIST Contour -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour",
    #     test_transform="identity"
    # ),
    # "mnist_contrast": ExperimentConfig(
    #     name="MNIST Contrast -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contrast",
    #     test_transform="identity"
    # ),
    # "mnist_blur": ExperimentConfig(
    #     name="MNIST Blur -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="blur",
    #     test_transform="identity"
    # ),
    # "mnist_invert": ExperimentConfig(
    #     name="MNIST Invert -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="invert",
    #     test_transform="identity"
    # ),

    # # Pairs
    # "mnist_color_contour": ExperimentConfig(
    #     name="MNIST Color+Contour -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour",
    #     test_transform="identity"
    # ),
    # "mnist_color_contrast": ExperimentConfig(
    #     name="MNIST Color+Contrast -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity"
    # ),
    # "mnist_color_noise": ExperimentConfig(
    #     name="MNIST Color+Noise -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_noise",
    #     test_transform="identity"
    # ),
    # "mnist_contour_noise": ExperimentConfig(
    #     name="MNIST Contour+Noise -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour_noise",
    #     test_transform="identity"
    # ),
    # "mnist_contrast_noise": ExperimentConfig(
    #     name="MNIST Contrast+Noise -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contrast_noise",
    #     test_transform="identity"
    # ),

    # # Triples
    # "mnist_color_contour_noise": ExperimentConfig(
    #     name="MNIST Color+Contour+Noise -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour_noise",
    #     test_transform="identity"
    # ),
    # "mnist_color_contrast_noise": ExperimentConfig(
    #     name="MNIST Color+Contrast+Noise -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contrast_noise",
    #     test_transform="identity"
    # ),
    # "mnist_contour_contrast_noise": ExperimentConfig(
    #     name="MNIST Contour+Contrast+Noise -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour_contrast_noise",
    #     test_transform="identity"
    # ),

    # # Quadruple
    # "mnist_color_contour_contrast_noise": ExperimentConfig(
    #     name="MNIST Color+Contour+Contrast+Noise -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour_contrast_noise",
    #     test_transform="identity"
    # ),

    # # All transforms
    # "mnist_all_transforms": ExperimentConfig(
    #     name="MNIST All Transforms -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="all_transforms",
    #     test_transform="identity"
    # ),


    # # Fashion MNIST 
    # "fmnist_noise": ExperimentConfig(
    #     name="Fashion-MNIST Noise -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="noise",
    #     test_transform="identity"
    # ),
    # "fmnist_contour": ExperimentConfig(
    #     name="Fashion-MNIST Contour -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="contour",
    #     test_transform="identity"
    # ),
    # "fmnist_contrast": ExperimentConfig(
    #     name="Fashion-MNIST Contrast -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="contrast",
    #     test_transform="identity"
    # ),
    # "fmnist_blur": ExperimentConfig(
    #     name="Fashion-MNIST Blur -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="blur",
    #     test_transform="identity"
    # ),
    # "fmnist_invert": ExperimentConfig(
    #     name="Fashion-MNIST Invert -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="invert",
    #     test_transform="identity"
    # ),

    # # Pairs
    # "fmnist_color_contour": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contour -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour",
    #     test_transform="identity"
    # ),
    # "fmnist_color_contrast": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity"
    # ),
    # "fmnist_color_noise": ExperimentConfig(
    #     name="Fashion-MNIST Color+Noise -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_noise",
    #     test_transform="identity"
    # ),
    # "fmnist_contour_noise": ExperimentConfig(
    #     name="Fashion-MNIST Contour+Noise -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="contour_noise",
    #     test_transform="identity"
    # ),
    # "fmnist_contrast_noise": ExperimentConfig(
    #     name="Fashion-MNIST Contrast+Noise -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="contrast_noise",
    #     test_transform="identity"
    # ),

    # # Triples
    # "fmnist_color_contour_noise": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contour+Noise -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour_noise",
    #     test_transform="identity"
    # ),

    # # ----------------------------------------
    # "fmnist_color_contrast_noise": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast+Noise -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contrast_noise",
    #     test_transform="identity"
    # ),
    # "fmnist_contour_contrast_noise": ExperimentConfig(
    #     name="Fashion-MNIST Contour+Contrast+Noise -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="contour_contrast_noise",
    #     test_transform="identity"
    # ),

    # # Quadruple
    # "fmnist_color_contour_contrast_noise": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contour+Contrast+Noise -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour_contrast_noise",
    #     test_transform="identity"
    # ),

    # # All transforms
    # "fmnist_all_transforms": ExperimentConfig(
    #     name="Fashion-MNIST All Transforms -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="all_transforms",
    #     test_transform="identity"
    # ),




    # # EMNIST 
    # "emnist_noise": ExperimentConfig(
    #     name="EMNIST Noise -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="noise",
    #     test_transform="identity"
    # ),
    # "emnist_contour": ExperimentConfig(
    #     name="EMNIST Contour -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="contour",
    #     test_transform="identity"
    # ),
    # "emnist_contrast": ExperimentConfig(
    #     name="EMNIST Contrast -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="contrast",
    #     test_transform="identity"
    # ),
    # "emnist_blur": ExperimentConfig(
    #     name="EMNIST Blur -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="blur",
    #     test_transform="identity"
    # ),
    # "emnist_invert": ExperimentConfig(
    #     name="EMNIST Invert -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="invert",
    #     test_transform="identity"
    # ),

    # # Pairs
    # "emnist_color_contour": ExperimentConfig(
    #     name="EMNIST Color+Contour -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour",
    #     test_transform="identity"
    # ),
    # "emnist_color_contrast": ExperimentConfig(
    #     name="EMNIST Color+Contrast -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contrast",
    #     test_transform="identity"
    # ),
    # "emnist_color_noise": ExperimentConfig(
    #     name="EMNIST Color+Noise -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_noise",
    #     test_transform="identity"
    # ),
    # "emnist_contour_noise": ExperimentConfig(
    #     name="EMNIST Contour+Noise -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="contour_noise",
    #     test_transform="identity"
    # ),
    # "emnist_contrast_noise": ExperimentConfig(
    #     name="EMNIST Contrast+Noise -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="contrast_noise",
    #     test_transform="identity"
    # ),

    # # Triples
    # "emnist_color_contour_noise": ExperimentConfig(
    #     name="EMNIST Color+Contour+Noise -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour_noise",
    #     test_transform="identity"
    # ),
    # "emnist_color_contrast_noise": ExperimentConfig(
    #     name="EMNIST Color+Contrast+Noise -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contrast_noise",
    #     test_transform="identity"
    # ),
    # "emnist_contour_contrast_noise": ExperimentConfig(
    #     name="EMNIST Contour+Contrast+Noise -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="contour_contrast_noise",
    #     test_transform="identity"
    # ),

    # # Quadruple
    # "emnist_color_contour_contrast_noise": ExperimentConfig(
    #     name="EMNIST Color+Contour+Contrast+Noise -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour_contrast_noise",
    #     test_transform="identity"
    # ),

    # # All transforms
    # "emnist_all_transforms": ExperimentConfig(
    #     name="EMNIST All Transforms -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="all_transforms",
    #     test_transform="identity"
    # ),

    # "mnist_color_contrast_contour": ExperimentConfig(
    #     name="MNIST Color+Contrast+Contour -> MNIST",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity"
    # ),

    # "fmnist_color_contrast_contour": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast+Contour -> Fashion-MNIST",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity"
    # ),

    # "emnist_color_contrast_contour": ExperimentConfig(
    #     name="EMNIST Color+Contrast+Contour -> EMNIST",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity"
    # ),

    # METHODS 
    ## MNIST + TENT
    # "mnist_contour_tent": ExperimentConfig(
    #     name="MNIST Contour -> MNIST with TENT",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),
    # "mnist_color_contour_tent": ExperimentConfig(
    #     name="MNIST Color+Contour -> MNIST with TENT",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),
    # "mnist_color_contrast_tent": ExperimentConfig(
    #     name="MNIST Color+Contrast -> MNIST with TENT",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),
    # "mnist_color_contrast_contour_tent": ExperimentConfig(
    #     name="MNIST Color+Contrast+Contour -> MNIST with TENT",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),

    ## MNIST + T3A
    # "mnist_baseline_t3a": ExperimentConfig(
    #     name="MNIST -> MNIST with T3A",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "mnist_color_t3a": ExperimentConfig(
    #     name="MNIST Color -> MNIST with T3A",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "mnist_contour_t3a": ExperimentConfig(
    #     name="MNIST Contour -> MNIST with T3A",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "mnist_color_contour_t3a": ExperimentConfig(
    #     name="MNIST Color+Contour -> MNIST with T3A",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "mnist_color_contrast_t3a": ExperimentConfig(
    #     name="MNIST Color+Contrast -> MNIST with T3A",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "mnist_color_contrast_contour_t3a": ExperimentConfig(
    #     name="MNIST Color+Contrast+Contour -> MNIST with T3A",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),

    # MNIST + SHOT 
    # "mnist_baseline_shot2": ExperimentConfig(
    #     name="MNIST -> MNIST with SHOT2",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "mnist_color_SHOT": ExperimentConfig(
    #     name="MNIST Color -> MNIST with SHOT2",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "mnist_contour_shot2": ExperimentConfig(
    #     name="MNIST Contour -> MNIST with SHOT2",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "mnist_color_contour_shot2": ExperimentConfig(
    #     name="MNIST Color+Contour -> MNIST with SHOT2",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "mnist_color_contrast_shot2": ExperimentConfig(
    #     name="MNIST Color+Contrast -> MNIST with SHOT2",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "mnist_color_contrast_contour_shot2": ExperimentConfig(
    #     name="MNIST Color+Contrast+Contour -> MNIST with SHOT2",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),

    # # MNIST + memo
    # "mnist_baseline_memo": ExperimentConfig(
    #     name="MNIST -> MNIST with MEMO",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "mnist_color_memo": ExperimentConfig(
    #     name="MNIST Color -> MNIST with MEMO",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "mnist_contour_memo": ExperimentConfig(
    #     name="MNIST Contour -> MNIST with MEMO",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "mnist_color_contour_memo": ExperimentConfig(
    #     name="MNIST Color+Contour -> MNIST with MEMO",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "mnist_color_contrast_memo": ExperimentConfig(
    #     name="MNIST Color+Contrast -> MNIST with MEMO",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "mnist_color_contrast_contour_memo": ExperimentConfig(
    #     name="MNIST Color+Contrast+Contour -> MNIST with MEMO",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),

    # MNIST + Style Shift
    # "mnist_baseline_style": ExperimentConfig(
    #     name="MNIST -> MNIST with Test-Time Style Shift",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="style"
    #     # epochs=2
    # ),
    # "mnist_color_style": ExperimentConfig(
    #     name="MNIST Color -> MNIST with Test-Time Style Shift",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "mnist_contour_style": ExperimentConfig(
    #     name="MNIST Contour -> MNIST with Test-Time Style Shift",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "mnist_color_contour_style": ExperimentConfig(
    #     name="MNIST Color+Contour -> MNIST with Test-Time Style Shift",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "mnist_color_contrast_style": ExperimentConfig(
    #     name="MNIST Color+Contrast -> MNIST with Test-Time Style Shift",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "mnist_color_contrast_contour_style": ExperimentConfig(
    #     name="MNIST Color+Contrast+Contour -> MNIST with Test-Time Style Shift",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),

    # Fashion-MNIST + TENT
    # "fmnist_contour_tent": ExperimentConfig(
    #     name="Fashion-MNIST Contour -> Fashion-MNIST with TENT",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),
    # "fmnist_color_contour_tent": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contour -> Fashion-MNIST with TENT",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),
    # "fmnist_color_contrast_tent": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast -> Fashion-MNIST with TENT",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),
    # "fmnist_color_contrast_contour_tent": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast+Contour -> Fashion-MNIST with TENT",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),

    # Fashion-MNIST + T3A
    # "fmnist_baseline_t3a": ExperimentConfig(
    #     name="Fashion-MNIST -> Fashion-MNIST with T3A",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "fmnist_color_t3a": ExperimentConfig(
    #     name="Fashion-MNIST Color -> Fashion-MNIST with T3A",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "fmnist_contour_t3a": ExperimentConfig(
    #     name="Fashion-MNIST Contour -> Fashion-MNIST with T3A",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "fmnist_color_contour_t3a": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contour -> Fashion-MNIST with T3A",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "fmnist_color_contrast_t3a": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast -> Fashion-MNIST with T3A",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "fmnist_color_contrast_contour_t3a": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast+Contour -> Fashion-MNIST with T3A",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),

    # # Fashion-MNIST + MEMO
    # "fmnist_baseline_memo": ExperimentConfig(
    #     name="Fashion-MNIST -> Fashion-MNIST with MEMO",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "fmnist_color_memo": ExperimentConfig(
    #     name="Fashion-MNIST Color -> Fashion-MNIST with MEMO",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "fmnist_contour_memo": ExperimentConfig(
    #     name="Fashion-MNIST Contour -> Fashion-MNIST with MEMO",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "fmnist_color_contour_memo": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contour -> Fashion-MNIST with MEMO",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "fmnist_color_contrast_memo": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast -> Fashion-MNIST with MEMO",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "fmnist_color_contrast_contour_memo": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast+Contour -> Fashion-MNIST with MEMO",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),

    # # Fashion-MNIST + SHOT
    # "fmnist_baseline_shot2": ExperimentConfig(
    #     name="Fashion-MNIST -> Fashion-MNIST with SHOT2",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "fmnist_color_shot": ExperimentConfig(
    #     name="Fashion-MNIST Color -> Fashion-MNIST with SHOT2",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "fmnist_contour_shot2": ExperimentConfig(
    #     name="Fashion-MNIST Contour -> Fashion-MNIST with SHOT2",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "fmnist_color_contour_shot2": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contour -> Fashion-MNIST with SHOT2",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "fmnist_color_contrast_shot2": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast -> Fashion-MNIST with SHOT2",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "fmnist_color_contrast_contour_shot2": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast+Contour -> Fashion-MNIST with SHOT2",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),

    # # Fashion-MNIST + Style Transfer
    # "fmnist_baseline_style": ExperimentConfig(
    #     name="Fashion-MNIST -> Fashion-MNIST with Test-Time Style Shift",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "fmnist_color_style": ExperimentConfig(
    #     name="Fashion-MNIST Color -> Fashion-MNIST with Test-Time Style Shift",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "fmnist_contour_style": ExperimentConfig(
    #     name="Fashion-MNIST Contour -> Fashion-MNIST with Test-Time Style Shift",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "fmnist_color_contour_style": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contour -> Fashion-MNIST with Test-Time Style Shift",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "fmnist_color_contrast_style": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast -> Fashion-MNIST with Test-Time Style Shift",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "fmnist_color_contrast_contour_style": ExperimentConfig(
    #     name="Fashion-MNIST Color+Contrast+Contour -> Fashion-MNIST with Test-Time Style Shift",
    #     train_dataset="fashion_mnist",
    #     test_dataset="fashion_mnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),



    # EMNIST + TENT
    # "emnist_contour_tent": ExperimentConfig(
    #     name="EMNIST Contour -> EMNIST with TENT",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),
    # "emnist_color_contour_tent": ExperimentConfig(
    #     name="EMNIST Color+Contour -> EMNIST with TENT",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),
    # "emnist_color_contrast_tent": ExperimentConfig(
    #     name="EMNIST Color+Contrast -> EMNIST with TENT",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),
    # "emnist_color_contrast_contour_tent": ExperimentConfig(
    #     name="EMNIST Color+Contrast+Contour -> EMNIST with TENT",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),

    # EMNIST + T3A
    # "emnist_baseline_t3a": ExperimentConfig(
    #     name="EMNIST -> EMNIST with T3A",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "emnist_color_t3a": ExperimentConfig(
    #     name="EMNIST Color -> EMNIST with T3A",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "emnist_contour_t3a": ExperimentConfig(
    #     name="EMNIST Contour -> EMNIST with T3A",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "emnist_color_contour_t3a": ExperimentConfig(
    #     name="EMNIST Color+Contour -> EMNIST with T3A",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "emnist_color_contrast_t3a": ExperimentConfig(
    #     name="EMNIST Color+Contrast -> EMNIST with T3A",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),
    # "emnist_color_contrast_contour_t3a": ExperimentConfig(
    #     name="EMNIST Color+Contrast+Contour -> EMNIST with T3A",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),

    # # EMNIST + MEMO
    # "emnist_baseline_memo": ExperimentConfig(
    #     name="EMNIST -> EMNIST with MEMO",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "emnist_color_memo": ExperimentConfig(
    #     name="EMNIST Color -> EMNIST with MEMO",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "emnist_contour_memo": ExperimentConfig(
    #     name="EMNIST Contour -> EMNIST with MEMO",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "emnist_color_contour_memo": ExperimentConfig(
    #     name="EMNIST Color+Contour -> EMNIST with MEMO",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "emnist_color_contrast_memo": ExperimentConfig(
    #     name="EMNIST Color+Contrast -> EMNIST with MEMO",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),
    # "emnist_color_contrast_contour_memo": ExperimentConfig(
    #     name="EMNIST Color+Contrast+Contour -> EMNIST with MEMO",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="memo"
    # ),

    # # EMNIST + SHOT
    # "emnist_baseline_shot2": ExperimentConfig(
    #     name="EMNIST -> EMNIST with SHOT2",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "emnist_color_shot2": ExperimentConfig(
    #     name="EMNIST Color -> EMNIST with SHOT2",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "emnist_contour_shot2": ExperimentConfig(
    #     name="EMNIST Contour -> EMNIST with SHOT2",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "emnist_color_contour_shot2": ExperimentConfig(
    #     name="EMNIST Color+Contour -> EMNIST with SHOT2",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "emnist_color_contrast_shot2": ExperimentConfig(
    #     name="EMNIST Color+Contrast -> EMNIST with SHOT2",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),
    # "emnist_color_contrast_contour_shot2": ExperimentConfig(
    #     name="EMNIST Color+Contrast+Contour -> EMNIST with SHOT2",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="shot2"
    # ),

    # EMNIST + Style Transfer
    # "emnist_baseline_style": ExperimentConfig(
    #     name="EMNIST -> EMNIST with Test-Time Style Shift",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="identity",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "emnist_color_style": ExperimentConfig(
    #     name="EMNIST Color -> EMNIST with Test-Time Style Shift",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="colorize",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "emnist_contour_style": ExperimentConfig(
    #     name="EMNIST Contour -> EMNIST with Style Transfer",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "emnist_color_contour_style": ExperimentConfig(
    #     name="EMNIST Color+Contour -> EMNIST with Style Transfer",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "emnist_color_contrast_style": ExperimentConfig(
    #     name="EMNIST Color+Contrast -> EMNIST with Style Transfer",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contrast",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
    # "emnist_color_contrast_contour_style": ExperimentConfig(
    #     name="EMNIST Color+Contrast+Contour -> EMNIST with Style Transfer",
    #     train_dataset="emnist",
    #     test_dataset="emnist",
    #     train_transform="color_contour_contrast",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),
# }



    # TEST TEST TEST
    # "mnist_contour_tent": ExperimentConfig(
    #     name="MNIST Contour -> MNIST with TENT",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="tent"
    # ),

    # "mnist_contour_t3a": ExperimentConfig(
    #     name="MNIST Contour -> MNIST with T3A",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="t3a"
    # ),

    # "mnist_contour_augtta": ExperimentConfig(
    #     name="MNIST Contour -> MNIST with AugTTA",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="augtta"
    # ),

    # "mnist_contour_shot": ExperimentConfig(
    #     name="MNIST Contour -> MNIST with SHOT",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="shot"
    # ),

    # "mnist_contour_style": ExperimentConfig(
    #     name="MNIST Contour -> MNIST with Style Transfer",
    #     train_dataset="mnist",
    #     test_dataset="mnist",
    #     train_transform="contour",
    #     test_transform="identity",
    #     adaptation_method="style"
    # ),