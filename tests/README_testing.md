# Testing Documentation

## 📁 Test Directory Structure

```
tests/
├── functional/                    # Functional tests
│   ├── test_positional_encoding.py      # Positional encoding tests
│   ├── test_multihead_attention.py     # Multi-head attention tests
│   └── test_transformer.py             # Complete Transformer tests
├── visualizations/               # Test visualizations (auto-generated)
│   ├── positional_encoding_visualization.png
│   ├── positional_encoding_3d_visualization.png
│   ├── multihead_attention_visualization.png
│   ├── multihead_attention_heads_visualization.png
│   ├── transformer_architecture.png
│   ├── transformer_structure.png
│   └── transformer_structure_screen.png
└── README_testing.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tests

#### Positional Encoding Tests
```bash
# Run positional encoding tests
python tests/functional/test_positional_encoding.py
```

#### Multi-Head Attention Tests
```bash
# Run multi-head attention tests
python tests/functional/test_multihead_attention.py
```

#### Complete Transformer Tests
```bash
# Run complete Transformer model tests
python tests/functional/test_transformer.py
```

#### Run All Tests
```bash
# Run all functional tests
python -m pytest tests/functional/
```

## 📋 Test Coverage

### Positional Encoding (`test_positional_encoding.py`)
- ✅ **Basic Functionality**: Forward pass and output shape validation
- ✅ **Uniqueness**: Ensures different positions have different encodings
- ✅ **Value Range**: Validates encoding values are within expected range
- ✅ **Consistency**: Tests encoding consistency across different inputs
- ✅ **Visualization**: Generates 2D and 3D visualization plots
- ✅ **Edge Cases**: Tests with different sequence lengths and model dimensions

### Multi-Head Attention (`test_multihead_attention.py`)
- ✅ **Forward Pass**: Tests attention mechanism computation
- ✅ **Output Shape**: Validates output tensor dimensions
- ✅ **Attention Weights**: Tests attention weight computation
- ✅ **Masking**: Tests causal and padding masks
- ✅ **Multi-Head**: Validates multiple attention heads
- ✅ **Visualization**: Generates attention weight heatmaps
- ✅ **Edge Cases**: Tests with different input sizes and configurations

### Complete Transformer (`test_transformer.py`)
- ✅ **Model Creation**: Tests Transformer model initialization
- ✅ **Forward Pass**: Tests complete model forward pass
- ✅ **Parameter Count**: Validates model parameter statistics
- ✅ **Model Saving**: Tests model checkpoint saving
- ✅ **Model Loading**: Tests model checkpoint loading
- ✅ **Architecture Visualization**: Generates model structure diagrams
- ✅ **Computational Cost**: Estimates FLOPs and memory usage

## 🎯 Test Features

### Visualization Generation
All tests automatically generate visualization plots:
- **Positional Encoding**: 2D and 3D plots showing encoding patterns
- **Attention Weights**: Heatmaps showing attention patterns
- **Model Architecture**: Diagrams showing model structure
- **Training Metrics**: Plots showing test results

### Comprehensive Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Visualization Tests**: Plot generation and validation
- **Edge Case Tests**: Boundary condition testing
- **Performance Tests**: Computational cost analysis

### Automated Validation
- **Shape Validation**: Tensor dimension checking
- **Value Validation**: Numerical range checking
- **Consistency Validation**: Cross-input consistency checking
- **Visualization Validation**: Plot generation verification

## 📊 Test Results

### Expected Outputs
- **Positional Encoding**: Unique encodings for each position
- **Multi-Head Attention**: Proper attention weight distributions
- **Transformer Model**: Successful forward pass with correct shapes
- **Visualizations**: Clear, informative plots saved to `tests/visualizations/`

### Performance Benchmarks
- **Model Size**: Parameter count validation
- **Computational Cost**: FLOPs estimation
- **Memory Usage**: Memory footprint analysis
- **Training Speed**: Epoch time measurement

## 🔧 Test Configuration

### Environment Requirements
- **Python**: 3.10+
- **PyTorch**: 2.2.2+
- **Matplotlib**: For visualization
- **NumPy**: For numerical computations

### Test Parameters
- **Batch Size**: Configurable batch sizes for testing
- **Sequence Length**: Various sequence lengths
- **Model Dimensions**: Different model configurations
- **Device**: CPU/GPU testing support

## 📈 Visualization Outputs

### Positional Encoding Visualizations
- **2D Plot**: Shows encoding patterns across positions and dimensions
- **3D Plot**: Interactive 3D visualization of encoding space
- **Heatmap**: Color-coded encoding values

### Attention Visualizations
- **Attention Heatmap**: Shows attention weights between tokens
- **Multi-Head Comparison**: Compares different attention heads
- **Layer Comparison**: Shows attention patterns across layers

### Model Architecture Visualizations
- **Structure Diagram**: Complete model architecture
- **Component Breakdown**: Individual component visualization
- **Parameter Statistics**: Model size and complexity analysis

## 🛠️ Running Custom Tests

### Creating New Tests
1. Create test file in `tests/functional/`
2. Follow naming convention: `test_*.py`
3. Include visualization generation
4. Add comprehensive assertions
5. Document test purpose and expected outputs

### Test Best Practices
- **Clear Naming**: Use descriptive test function names
- **Comprehensive Coverage**: Test all major functionality
- **Visualization**: Include relevant plots
- **Documentation**: Comment test purposes
- **Edge Cases**: Test boundary conditions

## 📝 Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed
- **Visualization Errors**: Check matplotlib backend configuration
- **Memory Issues**: Reduce batch size or model size for testing
- **Device Errors**: Ensure PyTorch is properly installed

### Debug Mode
```bash
# Run tests with verbose output
python -m pytest tests/functional/ -v

# Run specific test with debug output
python tests/functional/test_transformer.py --verbose
```

## 🎯 Future Enhancements

1. **Performance Tests**: Add timing and memory benchmarks
2. **Regression Tests**: Add tests for model performance regression
3. **Integration Tests**: Add end-to-end training tests
4. **Visualization Tests**: Add automated visualization validation
5. **Coverage Reports**: Add test coverage analysis

---

**Note**: All tests are designed to be self-contained and generate comprehensive visualizations for analysis.