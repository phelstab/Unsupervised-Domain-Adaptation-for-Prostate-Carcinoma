"""
Label Distribution Analyzer - ASCII visualization for dataset splits.

Usage:
    python -m cnn.training_setup.label_analyzer --source RUMC --target PCNN --binary
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from torch.utils.data import Subset


@dataclass
class LabelStats:
    """Statistics for a single data split."""
    name: str
    total: int
    class_counts: Dict[int, int]
    class_names: List[str]
    
    @property
    def class_percentages(self) -> Dict[int, float]:
        if self.total == 0:
            return {k: 0.0 for k in self.class_counts}
        return {k: (v / self.total) * 100 for k, v in self.class_counts.items()}


class LabelDistributionAnalyzer:
    """Analyzes and visualizes label distributions across dataset splits."""
    
    def __init__(self, binary_classification: bool = True):
        self.binary = binary_classification
        if binary_classification:
            self.class_names = ["Non-csPCa (0)", "csPCa (1)"]
        else:
            self.class_names = [f"ISUP {i}" for i in range(6)]
    
    def get_labels_from_subset(self, subset: Subset) -> np.ndarray:
        """Extract labels from a Subset, handling nested Subsets."""
        dataset = subset.dataset
        indices = subset.indices
        
        # Handle nested Subset (Subset of Subset)
        while isinstance(dataset, Subset):
            indices = [dataset.indices[i] for i in indices]
            dataset = dataset.dataset
        
        if hasattr(dataset, 'isup_labels'):
            return np.array([dataset.isup_labels[i] for i in indices])
        elif hasattr(dataset, 'datasets'):
            labels = []
            for idx in indices:
                cumulative = 0
                for ds in dataset.datasets:
                    if idx < cumulative + len(ds):
                        local_idx = idx - cumulative
                        labels.append(ds.isup_labels[local_idx])
                        break
                    cumulative += len(ds)
            return np.array(labels)
        else:
            raise ValueError(f"Cannot extract labels from {type(dataset)}")
    
    def compute_stats(self, subset: Subset, name: str) -> LabelStats:
        """Compute label statistics for a subset."""
        labels = self.get_labels_from_subset(subset)
        num_classes = 2 if self.binary else 6
        
        counts = {i: 0 for i in range(num_classes)}
        unique, cnts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, cnts):
            counts[int(u)] = int(c)
        
        return LabelStats(
            name=name,
            total=len(labels),
            class_counts=counts,
            class_names=self.class_names
        )
    
    def format_bar(self, percentage: float, width: int = 30, char: str = "#") -> str:
        """Create an ASCII bar for a percentage."""
        filled = int(percentage / 100 * width)
        empty = width - filled
        return char * filled + "." * empty
    
    def print_split_stats(self, stats: LabelStats, show_bar: bool = True) -> None:
        """Print statistics for a single split."""
        print(f"\n+{'-' * 50}+")
        print(f"| {stats.name:<48} |")
        print(f"+{'-' * 50}+")
        print(f"| {'Total samples:':<20} {stats.total:>27} |")
        print(f"+{'-' * 50}+")
        
        percentages = stats.class_percentages
        
        for class_id, count in stats.class_counts.items():
            pct = percentages[class_id]
            class_label = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            
            if show_bar:
                bar = self.format_bar(pct, width=20)
                print(f"| {class_label:<15} {count:>5} ({pct:>5.1f}%) {bar} |")
            else:
                print(f"| {class_label:<20} {count:>8} ({pct:>5.1f}%) |")
        
        print(f"+{'-' * 50}+")
    
    def print_comparison_table(self, all_stats: List[LabelStats]) -> None:
        """Print a comparison table of all splits."""
        if not all_stats:
            return
        
        num_classes = len(all_stats[0].class_counts)
        
        header = f"{'Split':<20}"
        for i in range(num_classes):
            label = self.class_names[i] if i < len(self.class_names) else f"C{i}"
            short_label = label.split()[0] if ' ' in label else label[:8]
            header += f" {short_label:>10}"
        header += f" {'Total':>8}"
        
        print("\n" + "=" * len(header))
        print("LABEL DISTRIBUTION COMPARISON")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        
        for stats in all_stats:
            row = f"{stats.name:<20}"
            for class_id in range(num_classes):
                count = stats.class_counts.get(class_id, 0)
                pct = stats.class_percentages.get(class_id, 0)
                row += f" {count:>4}({pct:>4.1f}%)"
            row += f" {stats.total:>8}"
            print(row)
        
        print("=" * len(header))
    
    def print_imbalance_warning(self, all_stats: List[LabelStats]) -> None:
        """Print warnings about class imbalance."""
        print("\n[!] IMBALANCE ANALYSIS:")
        print("-" * 40)
        
        for stats in all_stats:
            if stats.total == 0:
                print(f"  {stats.name}: [X] EMPTY SPLIT!")
                continue
            
            percentages = list(stats.class_percentages.values())
            max_pct = max(percentages)
            min_pct = min(percentages)
            
            if min_pct < 10:
                print(f"  {stats.name}: [!!] Severe imbalance (minority: {min_pct:.1f}%)")
            elif min_pct < 20:
                print(f"  {stats.name}: [!] Moderate imbalance (minority: {min_pct:.1f}%)")
            else:
                print(f"  {stats.name}: [OK] Acceptable ({min_pct:.1f}% - {max_pct:.1f}%)")


def analyze_splits(splits, binary_classification: bool = True, 
                   source_center: str = "Source", target_center: str = "Target") -> None:
    """
    Analyze and print label distributions for all splits.
    
    Args:
        splits: DataSplits object with source_train, source_val, source_test,
                target_train, target_val, target_test attributes
        binary_classification: Whether using binary classification
        source_center: Name of source center for display
        target_center: Name of target center for display
    """
    analyzer = LabelDistributionAnalyzer(binary_classification)
    
    split_configs = [
        (splits.source_train, f"{source_center} Train (Source)"),
        (splits.source_val, f"{source_center} Val (Source)"),
        (splits.source_test, f"{source_center} Test (Source)"),
        (splits.target_train, f"{target_center} Train (Target)"),
        (splits.target_val, f"{target_center} Val (Target)"),
        (splits.target_test, f"{target_center} Test (Target)"),
    ]
    
    all_stats = []
    
    print("\n" + "=" * 52)
    print("  LABEL DISTRIBUTION ANALYSIS")
    print("=" * 52)
    
    for subset, name in split_configs:
        if subset is None:
            continue
        stats = analyzer.compute_stats(subset, name)
        all_stats.append(stats)
        analyzer.print_split_stats(stats)
    
    analyzer.print_comparison_table(all_stats)
    analyzer.print_imbalance_warning(all_stats)
    
    return all_stats


def test_analyzer():
    """Test the analyzer with mock data."""
    print("\n" + "=" * 52)
    print("  RUNNING LABEL ANALYZER TEST")
    print("=" * 52)
    
    class MockDataset:
        def __init__(self, labels):
            self.isup_labels = np.array(labels)
        def __len__(self):
            return len(self.isup_labels)
    
    mock_source = MockDataset([0]*70 + [1]*30)
    mock_target = MockDataset([0]*45 + [1]*8)
    
    class MockSplits:
        source_train = Subset(mock_source, list(range(70)))
        source_val = Subset(mock_source, list(range(70, 85)))
        source_test = Subset(mock_source, list(range(85, 100)))
        target_train = Subset(mock_target, list(range(37)))
        target_val = Subset(mock_target, list(range(37, 45)))
        target_test = Subset(mock_target, list(range(45, 53)))
    
    stats = analyze_splits(MockSplits(), binary_classification=True,
                          source_center="MOCK_SRC", target_center="MOCK_TGT")
    
    assert len(stats) == 6, "Should have 6 splits"
    assert stats[0].total == 70, f"Source train should have 70 samples, got {stats[0].total}"
    assert stats[3].total == 37, f"Target train should have 37 samples, got {stats[3].total}"
    
    print("\n[OK] All tests passed!")
    return True


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Analyze label distributions")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--source", type=str, default="RUMC", help="Source center")
    parser.add_argument("--target", type=str, default="PCNN", help="Target center")
    parser.add_argument("--binary", action="store_true", help="Binary classification")
    parser.add_argument("--data-dir", type=str, default="input/images_preprocessed")
    
    args = parser.parse_args()
    
    if args.test:
        success = test_analyzer()
        sys.exit(0 if success else 1)
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data_generator import ISUPCenterDataset
    from uda_trainer import UDATrainer
    import torch
    
    print(f"\nAnalyzing: {args.source} -> {args.target}")
    print(f"Binary classification: {args.binary}")
    
    trainer = UDATrainer(
        source_center=args.source,
        target_center=args.target,
        workdir="workdir/label_analysis_temp",
        device=torch.device('cpu'),
        data_dir=args.data_dir,
        use_preprocessed=True,
        binary_classification=args.binary,
        da_method='none'
    )
    
    splits = trainer.create_datasets(source_size=-1, target_size=-1)
    analyze_splits(splits, args.binary, args.source, args.target)
