# Changelog - ML-Fragment-Optimizer

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.1] - 2025-01-20

### Added

#### Error Handling & Validation
- **SMILES Validation**
  - Added `validate_smiles()` function in `predict_properties.py`
  - Added `validate_training_smiles()` function in `train_admet_model.py`
  - Comprehensive RDKit-based SMILES parsing validation
  - Detailed error reporting showing invalid molecules with indices
  - Graceful handling of empty/null SMILES strings

- **File I/O Validation**
  - Pre-flight file existence checks for all file operations
  - Permission error detection and handling
  - UTF-8 encoding support with encoding error detection
  - Parent directory creation with proper error handling
  - Support for comment lines (`#`) in SMILES files
  - Support for flexible column names (SMILES/smiles, ID/id)

- **Data Loading Enhancements**
  - Enhanced `load_molecules()` in `predict_properties.py`:
    - Support for `.smi`, `.txt`, `.sdf`, `.csv` formats
    - Automatic format detection based on file extension
    - Graceful SDF reading error handling
    - Returns valid SMILES with their indices for tracking

  - Enhanced `load_data()` in `train_admet_model.py`:
    - Empty file detection
    - Missing column detection with helpful messages
    - Automatic numeric conversion for property columns
    - Minimum dataset size warnings (< 10 molecules)
    - Property statistics logging (mean, std, range)

- **Batch Processing Resilience**
  - Graceful batch failure handling in predictions
  - Automatic fallback to individual predictions on batch failure
  - NaN marking for failed predictions (preserves valid data)
  - Detailed failure reporting with molecule index and error message

- **Model & Configuration Error Handling**
  - Config file validation in `train_admet_model.py`
  - Property list validation (rejects empty properties)
  - Featurizer initialization with helpful error messages
  - Model type validation with valid options listed
  - Training failure diagnostics with suggested fixes
  - Model loading validation in `predict_properties.py`

#### Documentation
- Added `IMPROVEMENTS.md` - Comprehensive documentation of all improvements
- Added `SESSION_SUMMARY.md` - High-level session summary
- Added `CHANGELOG.md` - This file

### Changed

#### Code Quality
- **Removed sys.path manipulation** from `predict_properties.py`
  - Eliminated development hack: `sys.path.insert(0, ...)`
  - Fixed import order and organization

#### Error Messages
- **All error messages now:**
  - Show available options when user provides invalid input
  - Suggest fixes for common errors
  - Display file paths and expected formats
  - Include diagnostic information

#### Logging
- Added progress bars for batch predictions
- Added property statistics logging during data loading
- Added detailed metrics display after training
- Improved warning messages for data quality issues

### Fixed
- Import errors caused by sys.path manipulation
- Silent failures on invalid SMILES
- Complete batch failure on single molecule error
- Cryptic error messages on file I/O errors
- Missing validation of file existence before processing
- Poor error messages for missing CSV columns

### Technical Details

#### Files Modified
```
src/ml_fragment_optimizer/cli/
├── predict_properties.py    (+150 lines)
└── train_admet_model.py     (+188 lines)
```

#### New Functions
- `validate_smiles()` - 24 lines
- `validate_training_smiles()` - 24 lines
- Enhanced `load_molecules()` - 140 lines (was 46, +94)
- Enhanced `load_data()` - 116 lines (was 26, +90)

#### Error Coverage
- File I/O errors (5 types)
- SMILES validation errors (3 types)
- Data loading errors (7 types)
- Model operation errors (4 types)
- Batch processing partial failures

---

## [0.1.0] - 2025-01-19

### Added
- Initial release of ML-Fragment-Optimizer
- Core ADMET prediction models
- Multi-task learning for ADMET properties
- Molecular featurization (Morgan, MACCS, RDKit, Avalon fingerprints)
- QSAR model building
- CLI applications:
  - `mlfrag-train` - Train ADMET models
  - `mlfrag-predict` - Batch prediction
  - `mlfrag-optimize` - Fragment optimization
  - `mlfrag-synthesis` - Synthesis planning
  - `mlfrag-active-learning` - Active learning loop
  - `mlfrag-benchmark` - Model benchmarking
- Configuration file support (YAML)
- Model save/load functionality
- Uncertainty quantification for predictions

### Supported Features
- **Model Types**: Random Forest, XGBoost, Gradient Boosting
- **Fingerprints**: Morgan (ECFP), MACCS, RDKit, Avalon, AtomPair
- **Input Formats**: CSV with SMILES column
- **Output Formats**: CSV with predictions
- **Properties**: Solubility, LogP, Clearance, and any numeric property

---

## Version History Summary

| Version | Date       | Key Changes                              |
|---------|------------|------------------------------------------|
| 0.1.1   | 2025-01-20 | Production-ready error handling         |
| 0.1.0   | 2025-01-19 | Initial release with core functionality |

---

## Migration Guide

### From 0.1.0 to 0.1.1

✅ **No breaking changes** - All existing scripts will continue to work.

**New capabilities:**
- Better error messages (no code changes needed)
- More file formats supported (`.smi`, `.txt`, `.sdf` in addition to `.csv`)
- Flexible column names (SMILES/smiles, ID/id)
- Graceful handling of partial failures

**Recommended updates:**
1. Update error handling in custom scripts to match new patterns
2. Leverage new file format support if applicable
3. Review log output for new diagnostic information

---

## Known Issues

### Non-Critical
- **NumPy 2.0 Compatibility Warning**: Harmless warning about NumPy 1.x/2.x compatibility
  - Impact: None (package works correctly)
  - Fix: Will be resolved when dependencies rebuild for NumPy 2.0

- **torch_geometric Warning**: Warning when torch_geometric not installed
  - Impact: Graph-based features disabled (not required for core functionality)
  - Fix: Install torch_geometric if graph features needed

---

## Upcoming Features (Roadmap)

### Version 0.2.0 (Planned)
- Graph Neural Network models (D-MPNN, MPNN)
- Transfer learning support
- Advanced uncertainty quantification (evidential deep learning)
- Multi-objective optimization
- Web interface for predictions

### Version 0.3.0 (Planned)
- Integration with commercial ADMET databases
- Automated hyperparameter tuning
- Model interpretability (SHAP values)
- Distributed training support

---

## Contributing

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for:
- Error handling philosophy
- Validation strategy
- Code style guidelines
- Testing recommendations

---

## License

MIT License - See LICENSE file for details

---

## Credits

- **Core Development**: ML-Fragment-Optimizer Team
- **Error Handling Improvements**: Session 2025-01-20
- **Dependencies**: RDKit, scikit-learn, XGBoost, PyTorch, pandas, loguru

---

## Links

- **Documentation**: README.md, IMPROVEMENTS.md
- **Issues**: GitHub Issues
- **Examples**: notebooks/01_quick_start.ipynb
