"""
Data utilities for ML-Fragment-Optimizer.

This package provides comprehensive tools for loading, cleaning, processing,
and managing molecular datasets for machine learning applications.
"""

from .data_processing import (
    MoleculeDataLoader,
    MoleculeDataCleaner,
    DatasetSplitter,
    DataAugmenter,
    DoseResponseAnalyzer,
    calculate_dataset_statistics,
    impute_missing_values,
    batch_process,
    DatasetStatistics,
    SplitStrategy
)

from .dataset_loaders import (
    MoleculeNetLoader,
    ChEMBLLoader,
    ADMETLoader,
    load_benchmark_splits,
    get_dataset_info,
    DATASET_REGISTRY
)

from .assay_processing import (
    PlateReaderParser,
    AssayNormalizer,
    QualityController,
    ReplicateHandler,
    HitCaller,
    process_dose_response_plate,
    PlateQCMetrics,
    WellFormat
)

from .molecular_cleaning import (
    MolecularStandardizer,
    PAINSFilter,
    AggregatorFilter,
    ReactiveFilter,
    PropertyFilter,
    DuplicateRemover,
    ComprehensiveCleaner
)

from .format_converters import (
    MoleculeConverter,
    FileConverter,
    IdentifierConverter,
    BatchProcessor
)

__all__ = [
    # Data processing
    'MoleculeDataLoader',
    'MoleculeDataCleaner',
    'DatasetSplitter',
    'DataAugmenter',
    'DoseResponseAnalyzer',
    'calculate_dataset_statistics',
    'impute_missing_values',
    'batch_process',
    'DatasetStatistics',
    'SplitStrategy',

    # Dataset loaders
    'MoleculeNetLoader',
    'ChEMBLLoader',
    'ADMETLoader',
    'load_benchmark_splits',
    'get_dataset_info',
    'DATASET_REGISTRY',

    # Assay processing
    'PlateReaderParser',
    'AssayNormalizer',
    'QualityController',
    'ReplicateHandler',
    'HitCaller',
    'process_dose_response_plate',
    'PlateQCMetrics',
    'WellFormat',

    # Molecular cleaning
    'MolecularStandardizer',
    'PAINSFilter',
    'AggregatorFilter',
    'ReactiveFilter',
    'PropertyFilter',
    'DuplicateRemover',
    'ComprehensiveCleaner',

    # Format converters
    'MoleculeConverter',
    'FileConverter',
    'IdentifierConverter',
    'BatchProcessor',
]

__version__ = '0.1.0'
