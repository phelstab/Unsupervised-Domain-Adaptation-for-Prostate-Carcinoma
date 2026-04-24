"""
3D ResNet implementation for medical image classification.

Based on Tencent MedicalNet:
https://github.com/Tencent/MedicalNet/blob/master/models/resnet.py

Reference:
    Chen et al. "Med3D: Transfer Learning for 3D Medical Image Analysis"
    https://arxiv.org/abs/1904.00625
"""

import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    """Basic residual block for 3D ResNet (used in ResNet-10/18)."""

    expansion = 1

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, use_batchnorm=True
    ):
        super().__init__()
        norm_layer = nn.BatchNorm3d if use_batchnorm else lambda ch: nn.Identity()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_batchnorm,
        )
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_batchnorm,
        )
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck3D(nn.Module):
    """Bottleneck residual block for 3D ResNet (used in ResNet-50)."""

    expansion = 4

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, use_batchnorm=True
    ):
        super().__init__()
        norm_layer = nn.BatchNorm3d if use_batchnorm else lambda ch: nn.Identity()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, bias=not use_batchnorm
        )
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_batchnorm,
        )
        self.bn2 = norm_layer(out_channels)
        self.conv3 = nn.Conv3d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=not use_batchnorm,
        )
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    """3D ResNet backbone for medical imaging."""

    def __init__(
        self,
        block,
        layers,
        num_channels=3,
        num_classes=6,
        dropout_rate=0.0,
        feature_dim=512,
        use_batchnorm=True,
        clinical_feature_dim=0,
        clinical_fusion="early",
    ):
        super().__init__()
        self.in_channels = 64
        self.dropout_rate = dropout_rate
        self.feature_dim = feature_dim
        self.use_batchnorm = use_batchnorm
        self.num_classes = num_classes
        self.clinical_feature_dim = int(clinical_feature_dim or 0)
        self.clinical_fusion = clinical_fusion
        self.use_clinical_branch = self.clinical_feature_dim > 0
        norm_layer = nn.BatchNorm3d if use_batchnorm else lambda ch: nn.Identity()

        self.conv1 = nn.Conv3d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=not use_batchnorm
        )
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        encoder_out_dim = 512 * block.expansion

        # Feature layer: NO dropout - keep features clean for domain alignment
        self.feature_layer = nn.Sequential(
            nn.Linear(encoder_out_dim, feature_dim), nn.ReLU()
        )

        # Image-only classifier: always present, used by DA losses and as
        # the sole classifier when no auxiliary modality is active.
        self.image_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(256, num_classes),
        )

        if self.use_clinical_branch:
            if self.clinical_fusion not in {"early", "late"}:
                raise ValueError(
                    f"Unsupported clinical_fusion '{self.clinical_fusion}'. "
                    "Choose from 'early' or 'late'."
                )

            self.clinical_encoder = nn.Sequential(
                nn.Linear(self.clinical_feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )

            if self.clinical_fusion == "early":
                # Fused classifier: receives concat(image_features, clinical_embedding).
                # Used for CE source loss and final inference.
                self.fused_classifier = nn.Sequential(
                    nn.Linear(feature_dim + 64, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    nn.Linear(256, num_classes),
                )
            else:
                self.clinical_classifier = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes),
                )
                self.late_fusion_classifier = nn.Linear(num_classes * 2, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            norm_layer = (
                nn.BatchNorm3d if self.use_batchnorm else lambda ch: nn.Identity()
            )
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=not self.use_batchnorm,
                ),
                norm_layer(out_channels * block.expansion),
            )

        layers = [
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                use_batchnorm=self.use_batchnorm,
            )
        ]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, use_batchnorm=self.use_batchnorm)
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, clinical_features=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        features = self.feature_layer(x)

        # Image-only logits: always available, used by prediction-level DA
        # losses (MCC, BNM, entropy) so that domain alignment is never
        # polluted by auxiliary-modality signal.
        image_classification = self.image_classifier(features)

        if not self.use_clinical_branch or clinical_features is None:
            classification = image_classification
        elif self.clinical_fusion == "early":
            clinical_embedding = self.clinical_encoder(clinical_features)
            fused_features = torch.cat([features, clinical_embedding], dim=1)
            classification = self.fused_classifier(fused_features)
        else:
            clinical_embedding = self.clinical_encoder(clinical_features)
            clinical_logits = self.clinical_classifier(clinical_embedding)
            fused_logits = torch.cat([image_classification, clinical_logits], dim=1)
            classification = self.late_fusion_classifier(fused_logits)

        return {
            "features": features,
            "classification": classification,
            "image_classification": image_classification,
        }


# class ISUPClassifier(nn.Module):
#     """Original simple CNN classifier (backbone='simple')."""

#     def __init__(self, num_channels=3, num_classes=6, dropout_rate=0.0, use_batchnorm=False):
#         super().__init__()

#         self.use_batchnorm = use_batchnorm
#         self.dropout_rate = dropout_rate

#         def conv_block(in_ch, out_ch, pool_type='max'):
#             layers = [nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)]
#             if use_batchnorm:
#                 layers.append(nn.BatchNorm3d(out_ch))
#             layers.append(nn.ReLU())
#             if pool_type == 'max':
#                 layers.append(nn.MaxPool3d(2))
#             elif pool_type == 'adaptive':
#                 layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
#             return layers

#         self.encoder = nn.Sequential(
#             *conv_block(num_channels, 32, 'max'),
#             *conv_block(32, 64, 'max'),
#             *conv_block(64, 128, 'max'),
#             *conv_block(128, 256, 'adaptive'),
#         )

#         feature_layers = [nn.Linear(256, 512), nn.ReLU()]
#         if dropout_rate > 0:
#             feature_layers.append(nn.Dropout(dropout_rate))
#         self.feature_layer = nn.Sequential(*feature_layers)

#         classifier_layers = [nn.Linear(512, 256), nn.ReLU()]
#         if dropout_rate > 0:
#             classifier_layers.append(nn.Dropout(dropout_rate))
#         classifier_layers.append(nn.Linear(256, num_classes))
#         self.classifier = nn.Sequential(*classifier_layers)

#     def forward(self, x):
#         batch_size = x.size(0)

#         encoded = self.encoder(x)
#         encoded = encoded.view(batch_size, -1)

#         features = self.feature_layer(encoded)
#         classification = self.classifier(features)

#         return {
#             'features': features,
#             'classification': classification
#         }


def create_model(
    backbone="resnet10",
    num_channels=3,
    num_classes=6,
    dropout_rate=0.0,
    use_batchnorm=False,
    clinical_feature_dim=0,
    clinical_fusion="early",
):
    """
    Factory function to create model with specified backbone.

    Args:
        backbone: 'resnet10', 'resnet18', 'resnet34', or 'resnet50'
        num_channels: Number of input channels (3 for T2W, ADC, DWI)
        num_classes: Number of output classes (2 for binary, 6 for ISUP)
        dropout_rate: Dropout rate for regularization
        use_batchnorm: Whether to use batch normalization
        clinical_feature_dim: Number of clinical/tabular input features
        clinical_fusion: Clinical fusion mode ('early' or 'late')

    Returns:
        Model instance
    """
    backbone = backbone.lower()

    # if backbone == 'simple':
    #     return ISUPClassifier(
    #         num_channels=num_channels,
    #         num_classes=num_classes,
    #         dropout_rate=dropout_rate,
    #         use_batchnorm=use_batchnorm
    #     )
    if backbone == "resnet10":
        return ResNet3D(
            block=BasicBlock3D,
            layers=[1, 1, 1, 1],
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            clinical_feature_dim=clinical_feature_dim,
            clinical_fusion=clinical_fusion,
        )
    elif backbone == "resnet18":
        return ResNet3D(
            block=BasicBlock3D,
            layers=[2, 2, 2, 2],
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            clinical_feature_dim=clinical_feature_dim,
            clinical_fusion=clinical_fusion,
        )
    elif backbone == "resnet34":
        return ResNet3D(
            block=BasicBlock3D,
            layers=[3, 4, 6, 3],
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            clinical_feature_dim=clinical_feature_dim,
            clinical_fusion=clinical_fusion,
        )
    elif backbone == "resnet50":
        return ResNet3D(
            block=Bottleneck3D,
            layers=[3, 4, 6, 3],
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            clinical_feature_dim=clinical_feature_dim,
            clinical_fusion=clinical_fusion,
        )
    else:
        raise ValueError(
            f"Unknown backbone: {backbone}. Choose from 'resnet10', 'resnet18', 'resnet34', 'resnet50'"
        )
