import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class ISUPLoss(nn.Module):
    def __init__(self, num_classes=6, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, predictions, targets):
        return self.ce_loss(predictions, targets)


class CORALLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, source_features, target_features):
        d = source_features.size(1)
        ns = source_features.size(0)
        nt = target_features.size(0)
        
        # Guard against batch_size=1 (division by zero in covariance)
        if ns <= 1 or nt <= 1:
            return torch.tensor(0.0, device=source_features.device, requires_grad=True)
        
        source_mean = source_features.mean(dim=0, keepdim=True)
        target_mean = target_features.mean(dim=0, keepdim=True)
        
        source_centered = source_features - source_mean
        target_centered = target_features - target_mean
        
        source_cov = (source_centered.t() @ source_centered) / (ns - 1)
        target_cov = (target_centered.t() @ target_centered) / (nt - 1)
        
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2 / (4 * d * d)
        
        return loss


class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits):
        """
        Compute entropy of predictions to encourage confident predictions.
        Lower entropy = more confident predictions.
        
        Args:
            logits: [batch_size, num_classes] raw model outputs
        
        Returns:
            Average entropy across the batch
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        return entropy


class GaussianKernel(nn.Module):
    """Gaussian Kernel Matrix - exact TLlib implementation.
    
    Source: https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/modules/kernels.py
    
    k(x1, x2) = exp(-||x1 - x2||^2 / (2 * sigma^2))
    
    When track_running_stats=True, sigma^2 = alpha * mean(||xi - xj||^2)
    
    Args:
        sigma: Fixed bandwidth. If None, computed adaptively.
        track_running_stats: If True, compute sigma from data. If False, use fixed sigma.
        alpha: Multiplier for adaptive sigma^2.
    """
    
    def __init__(self, sigma: float = None, track_running_stats: bool = True, alpha: float = 1.0):
        super().__init__()
        assert track_running_stats or sigma is not None, "Must provide sigma if track_running_stats=False"
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)
        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())
        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


def _update_index_matrix(batch_size: int, index_matrix: torch.Tensor = None, 
                         linear: bool = True) -> torch.Tensor:
    """
    Update the index_matrix which converts kernel_matrix to loss.
    Exact TLlib implementation.
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


class MMDLoss(nn.Module):
    """Multiple Kernel Maximum Mean Discrepancy (MK-MMD) loss.
    
    Exact TLlib implementation from:
    https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/dan.py
    
    Paper: Learning Transferable Features with Deep Adaptation Networks (ICML 2015)
    
    Args:
        kernel_alphas: Alpha multipliers for Gaussian kernels. Default matches TLlib example.
        linear: If True, use linear-time O(n) estimator. If False, use O(n²) estimator.
    """
    
    def __init__(self, kernel_alphas: list = None, linear: bool = False):
        super().__init__()
        if kernel_alphas is None:
            # TLlib example uses [0.5, 1.0, 2.0]
            kernel_alphas = [0.5, 1.0, 2.0]
        
        self.kernels = nn.ModuleList([
            GaussianKernel(sigma=None, track_running_stats=True, alpha=alpha) 
            for alpha in kernel_alphas
        ])
        self.index_matrix = None
        self.linear = linear
    
    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        # Handle different batch sizes (truncate to minimum)
        min_batch = min(z_s.size(0), z_t.size(0))
        if min_batch < 2:
            return torch.tensor(0.0, device=z_s.device, requires_grad=True)
        z_s = z_s[:min_batch]
        z_t = z_t[:min_batch]
        
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)
        
        # Add up the matrix of each kernel
        kernel_matrix = sum([kernel(features) for kernel in self.kernels])
        
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        
        return loss


class MCDLoss(nn.Module):
    """Maximum Classifier Discrepancy loss for domain adaptation.
    
    Paper: Maximum Classifier Discrepancy for Unsupervised Domain Adaptation (CVPR 2018)
    https://arxiv.org/abs/1712.02560
    
    https://raw.githubusercontent.com/ViLab-UCSD/UDABench_ECCV2024/refs/heads/master/UDA_trainer/mcd.py
    
    MCD uses two classifiers and measures discrepancy as L1 distance between
    their softmax outputs on target domain data.
    
    Args:
        feature_dim: Dimension of input features
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension in classifiers
        dropout_rate: Dropout rate in classifiers
    """
    
    def __init__(self, feature_dim=512, num_classes=2, hidden_dim=256, dropout_rate=0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        self.classifier1 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Forward pass through both classifiers.
        
        Args:
            features: [batch_size, feature_dim] feature vectors
        
        Returns:
            Tuple of (logits1, logits2) from both classifiers
        """
        return self.classifier1(features), self.classifier2(features)
    
    def discrepancy(self, logits1, logits2):
        """
        Compute L1 discrepancy between two classifier outputs.
        
        Args:
            logits1: [batch_size, num_classes] logits from classifier 1
            logits2: [batch_size, num_classes] logits from classifier 2
        
        Returns:
            Mean L1 distance between softmax outputs
        """
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)
        return torch.mean(torch.abs(probs1 - probs2))
    
    def classification_loss(self, logits1, logits2, labels):
        """
        Compute classification loss for both classifiers.
        
        Args:
            logits1: [batch_size, num_classes] logits from classifier 1
            logits2: [batch_size, num_classes] logits from classifier 2
            labels: [batch_size] ground truth labels
        
        Returns:
            Sum of cross-entropy losses from both classifiers
        """
        ce_loss = nn.CrossEntropyLoss()
        return ce_loss(logits1, labels) + ce_loss(logits2, labels)
    
    def entropy_loss(self, logits):
        """
        Compute entropy loss to encourage diverse predictions.
        Used as regularization in MCD training.
        
        Args:
            logits: [batch_size, num_classes] logits
        
        Returns:
            Negative entropy of mean prediction (encourages diversity)
        """
        probs = F.softmax(logits, dim=1)
        mean_probs = torch.mean(probs, dim=0)
        return -torch.mean(torch.log(mean_probs + 1e-6))
    
    def to(self, device):
        """Move module to device."""
        super().to(device)
        self.classifier1 = self.classifier1.to(device)
        self.classifier2 = self.classifier2.to(device)
        return self


class MCCLoss(nn.Module):
    """Minimum Class Confusion loss for domain adaptation.
    
    Paper: Minimum Class Confusion for Versatile Domain Adaptation (ECCV 2020)
    https://arxiv.org/abs/1912.03699
    
    Source: https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/self_training/mcc.py
    
    MCC minimizes class confusion in target predictions by computing a class
    confusion matrix weighted by prediction entropy, then minimizing off-diagonal
    elements (confusion between classes).
    
    Key insight: confident predictions (low entropy) are weighted higher because
    they are more reliable for estimating class confusion.
    
    Args:
        temperature: Temperature for softmax rescaling. Higher temperature = softer
                     probability distribution. Default 2.5 as in paper.
    
    Note:
        - Only uses target domain predictions (unsupervised)
        - Works on classifier outputs (logits), not features
        - Can be combined with other losses (DANN, CDAN, etc.) as regularizer
    """
    
    def __init__(self, temperature: float = 2.5):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Minimum Class Confusion loss.
        
        Args:
            logits: [batch_size, num_classes] unnormalized classifier predictions
                    on target domain
        
        Returns:
            MCC loss (scalar) - minimize this to reduce class confusion
        """
        batch_size, num_classes = logits.shape
        
        # Temperature-scaled softmax predictions
        predictions = F.softmax(logits / self.temperature, dim=1)  # [B, C]
        
        # Compute entropy weight for each sample
        # Low entropy (confident) samples get higher weight
        entropy = self._entropy(predictions)  # [B]
        entropy_weight = 1 + torch.exp(-entropy)  # Higher weight for low entropy
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(1)  # [B, 1]
        
        # Compute weighted class confusion matrix
        # C[i,j] = sum over samples of P(class i) * P(class j) * weight
        # Diagonal = correct predictions, Off-diagonal = confusion
        weighted_predictions = predictions * entropy_weight  # [B, C]
        class_confusion_matrix = torch.mm(weighted_predictions.t(), predictions)  # [C, C]
        
        # Normalize rows to get conditional probabilities
        class_confusion_matrix = class_confusion_matrix / (torch.sum(class_confusion_matrix, dim=1, keepdim=True) + 1e-8)
        
        # MCC loss = (sum of all elements - trace) / num_classes
        # = sum of off-diagonal elements / num_classes
        # Minimizing this reduces class confusion
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        
        return mcc_loss
    
    def _entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of predictions.
        
        Args:
            predictions: [batch_size, num_classes] probability distribution
        
        Returns:
            [batch_size] entropy for each sample
        """
        return -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)


class BNMLoss(nn.Module):
    """Batch Nuclear-norm Maximization loss for domain adaptation.
    
    Paper: Towards Discriminability and Diversity: Batch Nuclear-norm Maximization
           under Label Insufficient Situations (CVPR 2020)
    https://arxiv.org/abs/2003.12237
    
    Source: https://kevinmusgrave.github.io/pytorch-adapt/docs/layers/bnm_loss/
    
    BNM maximizes the nuclear norm of the batch prediction matrix to encourage
    both discriminability (confident predictions) and diversity (different 
    predictions for different samples).
    
    Nuclear norm = sum of singular values of the matrix.
    Maximizing nuclear norm encourages:
    1. Each row (sample) to have one-hot-like predictions (discriminability)
    2. Different samples to predict different classes (diversity)
    
    Note:
        - Only uses target domain predictions (unsupervised)
        - Works on classifier outputs (logits), not features
        - Returns negative nuclear norm (to minimize during training)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Batch Nuclear-norm Maximization loss.
        
        Args:
            logits: [batch_size, num_classes] unnormalized classifier predictions
                    on target domain
        
        Returns:
            BNM loss (scalar) - negative normalized nuclear norm
        """
        # Softmax to get probability predictions
        probs = F.softmax(logits, dim=1)
        
        # Compute nuclear norm (sum of singular values)
        # torch.linalg.norm with 'nuc' computes the nuclear norm
        nuclear_norm = torch.linalg.norm(probs, ord='nuc')
        
        # Return negative nuclear norm (normalized by batch size)
        # Negative because we want to maximize nuclear norm by minimizing loss
        return -nuclear_norm / probs.shape[0]


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for DANN.
    
    Source: https://github.com/tadeephuy/GradientReversal
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None


revgrad = GradientReversalFunction.apply


class GradientReversalLayer(nn.Module):
    """Wrapper module for gradient reversal function."""
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)
    
    def forward(self, x):
        return revgrad(x, self.alpha)
    
    def set_alpha(self, alpha):
        self.alpha = torch.tensor(alpha, requires_grad=False)


class DANNLoss(nn.Module):
    """Domain Adversarial Neural Network loss."""
    
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 2)
        )
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, source_features, target_features):
        """
        Compute DANN domain classification loss with gradient reversal.
        
        Args:
            source_features: [batch_size, feature_dim] source domain features
            target_features: [batch_size, feature_dim] target domain features
        
        Returns:
            Domain classification loss (to be minimized by discriminator,
            but gradient reversal causes feature extractor to maximize it)
        """
        features = torch.cat([source_features, target_features], dim=0)
        reversed_features = self.grl(features)
        
        domain_preds = self.domain_classifier(reversed_features)
        
        n_source = source_features.size(0)
        n_target = target_features.size(0)
        domain_labels = torch.cat([
            torch.zeros(n_source, dtype=torch.long, device=source_features.device),
            torch.ones(n_target, dtype=torch.long, device=target_features.device)
        ])
        
        return self.ce_loss(domain_preds, domain_labels)
    
    def set_alpha(self, alpha, device=None):
        """Update GRL alpha for scheduling."""
        if device is not None:
            self.grl.alpha = torch.tensor(alpha, requires_grad=False, device=device)
        else:
            self.grl.set_alpha(alpha)
    
    def to(self, device):
        """Move module to device."""
        super().to(device)
        self.domain_classifier = self.domain_classifier.to(device)
        self.grl.alpha = self.grl.alpha.to(device)
        return self


class DAARDALoss(nn.Module):
    """Asymmetrically-Relaxed Distribution Alignment (ARDA) loss.

    Implements the core divergence objectives from:
    "Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment"

    The loss is computed from source/target discriminator scores. During feature
    extractor updates, we minimize the divergence. During discriminator updates,
    we maximize it (equivalently minimize the negative divergence).
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        divergence: str = 'js_beta',
        relax: float = 1.0,
        grad_penalty: float = 0.0,
    ):
        super().__init__()

        if relax < 0:
            raise ValueError(f"relax must be >= 0, got {relax}")
        if grad_penalty < 0:
            raise ValueError(f"grad_penalty must be >= 0, got {grad_penalty}")

        divergence = divergence.lower()
        valid_divergences = {'js', 'js_beta', 'w_beta', 'js_sort'}
        if divergence not in valid_divergences:
            raise ValueError(
                f"Invalid divergence '{divergence}'. Must be one of {sorted(valid_divergences)}"
            )
        if divergence == 'js' and relax > 0:
            raise ValueError("Use 'js_beta' or set relax=0 when divergence='js'")

        self.divergence = divergence
        self.relax = relax
        self.grad_penalty = grad_penalty

        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _soft_relu(x: torch.Tensor) -> torch.Tensor:
        """Compute log(1 + exp(x)) stably."""
        return torch.log1p(torch.exp(-x.abs())) + torch.clamp(x, min=0.0)

    def _js_div(self, d_source: torch.Tensor, d_target: torch.Tensor) -> torch.Tensor:
        part1 = -self._soft_relu(-d_source).mean()
        part2 = -self._soft_relu(d_target).mean()
        return part1 + part2 + np.log(4.0)

    def _js_beta(self, d_source: torch.Tensor, d_target: torch.Tensor) -> torch.Tensor:
        part1 = -self._soft_relu(-d_source).mean()
        sigmoid_target = torch.exp(-self._soft_relu(-d_target))
        part2 = torch.log(self.relax + 2.0 - sigmoid_target).mean()
        return part1 + part2 - np.log(1.0 + self.relax)

    def _wasserstein_beta(self, d_source: torch.Tensor, d_target: torch.Tensor) -> torch.Tensor:
        part1 = -(1.0 + self.relax) * self._soft_relu(d_source).mean()
        part2 = self._soft_relu(d_target).mean()
        return part1 + part2

    def _js_sort(self, d_source: torch.Tensor, d_target: torch.Tensor) -> torch.Tensor:
        n = d_source.shape[0]
        n_selected = max(1, int(n // (1.0 + self.relax)))
        selected_source = torch.topk(d_source, n_selected, largest=False, sorted=False)[0]
        return self._js_div(selected_source, d_target)

    def _compute_divergence(self, d_source: torch.Tensor, d_target: torch.Tensor) -> torch.Tensor:
        if self.divergence == 'js':
            return self._js_div(d_source, d_target)
        if self.divergence == 'js_beta':
            return self._js_beta(d_source, d_target)
        if self.divergence == 'w_beta':
            return self._wasserstein_beta(d_source, d_target)
        if self.divergence == 'js_sort':
            return self._js_sort(d_source, d_target)
        raise RuntimeError(f"Unsupported divergence: {self.divergence}")

    def _discriminator_scores(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([source_features, target_features], dim=0)
        scores = self.domain_discriminator(features).squeeze(-1)
        n_source = source_features.shape[0]
        return scores[:n_source], scores[n_source:]

    def _gradient_penalty(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        n = min(source_features.shape[0], target_features.shape[0])
        if n == 0:
            return torch.tensor(0.0, device=source_features.device, requires_grad=True)
        source_features = source_features[:n]
        target_features = target_features[:n]

        alpha = torch.rand(n, 1, device=source_features.device)
        interpolated = alpha * source_features + (1.0 - alpha) * target_features
        interpolated.requires_grad_(True)

        scores = self.domain_discriminator(interpolated).squeeze(-1)
        grads = autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_norm = torch.sqrt(grads.square().sum(dim=1) + 1e-10)
        return (grad_norm - 1.0).square().mean()

    def feature_alignment_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Loss for feature extractor update (minimize divergence)."""
        d_source, d_target = self._discriminator_scores(source_features, target_features)
        return self._compute_divergence(d_source, d_target)

    def discriminator_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Loss for discriminator update (maximize divergence via negative sign)."""
        d_source, d_target = self._discriminator_scores(
            source_features.detach(), target_features.detach()
        )
        loss = -self._compute_divergence(d_source, d_target)

        if self.grad_penalty > 0:
            gp = self._gradient_penalty(
                source_features.detach(), target_features.detach()
            )
            loss = loss + self.grad_penalty * gp

        return loss
