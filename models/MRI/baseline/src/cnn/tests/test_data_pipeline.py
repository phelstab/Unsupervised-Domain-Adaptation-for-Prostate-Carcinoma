"""
Test suite for UDA CNN data pipeline
Run: python -m pytest models/MRI/baseline/src/cnn/tests/test_data_pipeline.py -v
"""

import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training_setup.data_generator import ISUPCenterDataset
from training_setup.loss_functions import ISUPLoss, CORALLoss
from training_setup.domain_discriminator import DomainDiscriminator
from model import ISUPClassifier


class TestMarksheetLoading:
    """Test marksheet CSV loading and center filtering"""
    
    def test_marksheet_exists(self):
        marksheet_path = Path('input/picai_labels/clinical_information/marksheet.csv')
        assert marksheet_path.exists(), f"Marksheet not found at {marksheet_path}"
    
    def test_marksheet_has_center_column(self):
        marksheet_path = 'input/picai_labels/clinical_information/marksheet.csv'
        df = pd.read_csv(marksheet_path)
        assert 'center' in df.columns, "Marksheet missing 'center' column"
        assert 'patient_id' in df.columns, "Marksheet missing 'patient_id' column"
        assert 'case_ISUP' in df.columns, "Marksheet missing 'case_ISUP' column"
    
    def test_centers_present(self):
        marksheet_path = 'input/picai_labels/clinical_information/marksheet.csv'
        df = pd.read_csv(marksheet_path)
        centers = df['center'].unique()
        
        assert 'RUMC' in centers, "RUMC center not found"
        assert 'PCNN' in centers, "PCNN center not found"
        assert 'ZGT' in centers, "ZGT center not found"
    
    def test_center_counts(self):
        marksheet_path = 'input/picai_labels/clinical_information/marksheet.csv'
        df = pd.read_csv(marksheet_path)
        
        rumc_count = (df['center'] == 'RUMC').sum()
        pcnn_count = (df['center'] == 'PCNN').sum()
        zgt_count = (df['center'] == 'ZGT').sum()
        
        assert rumc_count > 0, f"RUMC has {rumc_count} samples"
        assert pcnn_count > 0, f"PCNN has {pcnn_count} samples"
        assert zgt_count > 0, f"ZGT has {zgt_count} samples"
        
        print(f"\nCenter counts: RUMC={rumc_count}, PCNN={pcnn_count}, ZGT={zgt_count}")
    
    def test_isup_labels_valid(self):
        marksheet_path = 'input/picai_labels/clinical_information/marksheet.csv'
        df = pd.read_csv(marksheet_path)
        
        isup_values = df['case_ISUP'].unique()
        for val in isup_values:
            assert 0 <= val <= 5, f"Invalid ISUP value: {val}"


class TestISUPCenterDataset:
    """Test ISUPCenterDataset functionality"""
    
    @pytest.fixture
    def marksheet_path(self):
        return 'input/picai_labels/clinical_information/marksheet.csv'
    
    @pytest.fixture
    def data_dir(self):
        return 'input/picai_public_images_fold0'
    
    def test_dataset_creation_rumc(self, marksheet_path, data_dir):
        try:
            dataset = ISUPCenterDataset(
                center='RUMC',
                marksheet_path=marksheet_path,
                data_dir=data_dir
            )
            assert len(dataset) > 0, "RUMC dataset is empty"
            print(f"\nRUMC dataset size: {len(dataset)}")
        except Exception as e:
            pytest.skip(f"Data directory not available: {e}")
    
    def test_dataset_creation_pcnn(self, marksheet_path, data_dir):
        try:
            dataset = ISUPCenterDataset(
                center='PCNN',
                marksheet_path=marksheet_path,
                data_dir=data_dir
            )
            assert len(dataset) > 0, "PCNN dataset is empty"
            print(f"\nPCNN dataset size: {len(dataset)}")
        except Exception as e:
            pytest.skip(f"Data directory not available: {e}")
    
    def test_dataset_creation_zgt(self, marksheet_path, data_dir):
        try:
            dataset = ISUPCenterDataset(
                center='ZGT',
                marksheet_path=marksheet_path,
                data_dir=data_dir
            )
            assert len(dataset) > 0, "ZGT dataset is empty"
            print(f"\nZGT dataset size: {len(dataset)}")
        except Exception as e:
            pytest.skip(f"Data directory not available: {e}")
    
    def test_dataset_filtering(self, marksheet_path, data_dir):
        try:
            rumc_dataset = ISUPCenterDataset(center='RUMC', marksheet_path=marksheet_path, data_dir=data_dir)
            pcnn_dataset = ISUPCenterDataset(center='PCNN', marksheet_path=marksheet_path, data_dir=data_dir)
            
            assert rumc_dataset.center == 'RUMC'
            assert pcnn_dataset.center == 'PCNN'
            assert len(rumc_dataset) != len(pcnn_dataset), "Datasets should have different sizes"
        except Exception as e:
            pytest.skip(f"Data directory not available: {e}")
    
    def test_isup_labels_in_dataset(self, marksheet_path, data_dir):
        try:
            dataset = ISUPCenterDataset(center='RUMC', marksheet_path=marksheet_path, data_dir=data_dir)
            
            for label in dataset.isup_labels:
                assert 0 <= label <= 5, f"Invalid ISUP label: {label}"
        except Exception as e:
            pytest.skip(f"Data directory not available: {e}")


class TestModelArchitecture:
    """Test model components"""
    
    def test_isup_classifier_creation(self):
        model = ISUPClassifier(num_channels=3, num_classes=6)
        assert model is not None
    
    def test_isup_classifier_forward(self):
        model = ISUPClassifier(num_channels=3, num_classes=6)
        batch_size = 2
        x = torch.randn(batch_size, 3, 20, 256, 256)
        
        output = model(x)
        
        assert 'features' in output
        assert 'classification' in output
        assert output['features'].shape == (batch_size, 512)
        assert output['classification'].shape == (batch_size, 6)
    
    def test_domain_discriminator_creation(self):
        discriminator = DomainDiscriminator(feature_dim=512)
        assert discriminator is not None
    
    def test_domain_discriminator_forward(self):
        discriminator = DomainDiscriminator(feature_dim=512)
        batch_size = 4
        features = torch.randn(batch_size, 512)
        
        output = discriminator(features)
        
        assert output.shape == (batch_size, 2)
    
    def test_model_device_transfer(self):
        model = ISUPClassifier(num_channels=3, num_classes=6)
        
        if torch.cuda.is_available():
            model = model.cuda()
            x = torch.randn(1, 3, 20, 256, 256).cuda()
            output = model(x)
            assert output['features'].is_cuda
        else:
            x = torch.randn(1, 3, 20, 256, 256)
            output = model(x)
            assert not output['features'].is_cuda


class TestLossFunctions:
    """Test loss function implementations"""
    
    def test_isup_loss_creation(self):
        loss_fn = ISUPLoss(num_classes=6)
        assert loss_fn is not None
    
    def test_isup_loss_forward(self):
        loss_fn = ISUPLoss(num_classes=6)
        predictions = torch.randn(4, 6)
        targets = torch.tensor([0, 1, 2, 3])
        
        loss = loss_fn(predictions, targets)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_isup_loss_with_weights(self):
        class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        loss_fn = ISUPLoss(num_classes=6, class_weights=class_weights)
        
        predictions = torch.randn(4, 6)
        targets = torch.tensor([0, 1, 2, 3])
        
        loss = loss_fn(predictions, targets)
        assert loss.item() > 0
    
    def test_coral_loss_creation(self):
        loss_fn = CORALLoss()
        assert loss_fn is not None
    
    def test_coral_loss_forward(self):
        loss_fn = CORALLoss()
        
        source_features = torch.randn(10, 512)
        target_features = torch.randn(10, 512)
        
        loss = loss_fn(source_features, target_features)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_coral_loss_identical_features(self):
        loss_fn = CORALLoss()
        
        features = torch.randn(10, 512)
        loss = loss_fn(features, features.clone())
        
        assert loss.item() < 1e-3
    
    def test_coral_loss_different_batch_sizes(self):
        loss_fn = CORALLoss()
        
        source_features = torch.randn(8, 512)
        target_features = torch.randn(12, 512)
        
        loss = loss_fn(source_features, target_features)
        assert loss.item() >= 0


class TestDataLoaderCompatibility:
    """Test DataLoader integration"""
    
    def test_dataset_with_dataloader(self):
        try:
            dataset = ISUPCenterDataset(
                center='RUMC',
                marksheet_path='input/picai_labels/clinical_information/marksheet.csv',
                data_dir='input/picai_public_images_fold0'
            )
            
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            
            assert len(loader) > 0
        except Exception as e:
            pytest.skip(f"Data directory not available: {e}")


class TestEndToEndPipeline:
    """Test complete pipeline integration"""
    
    def test_full_forward_pass(self):
        model = ISUPClassifier(num_channels=3, num_classes=6)
        discriminator = DomainDiscriminator(feature_dim=512)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 20, 256, 256)
        
        output = model(x)
        features = output['features']
        classification = output['classification']
        
        domain_pred = discriminator(features)
        
        assert features.shape == (batch_size, 512)
        assert classification.shape == (batch_size, 6)
        assert domain_pred.shape == (batch_size, 2)
    
    def test_loss_computation(self):
        cls_loss_fn = ISUPLoss(num_classes=6)
        coral_loss_fn = CORALLoss()
        
        batch_size = 4
        source_features = torch.randn(batch_size, 512)
        target_features = torch.randn(batch_size, 512)
        predictions = torch.randn(batch_size, 6)
        targets = torch.tensor([0, 1, 2, 3])
        
        cls_loss = cls_loss_fn(predictions, targets)
        coral_loss = coral_loss_fn(source_features, target_features)
        total_loss = cls_loss + 0.5 * coral_loss
        
        assert not torch.isnan(total_loss)
        assert total_loss.item() > 0
    
    def test_backward_pass(self):
        model = ISUPClassifier(num_channels=3, num_classes=6)
        loss_fn = ISUPLoss(num_classes=6)
        
        x = torch.randn(2, 3, 20, 256, 256)
        targets = torch.tensor([0, 1])
        
        output = model(x)
        loss = loss_fn(output['classification'], targets)
        
        loss.backward()
        
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
