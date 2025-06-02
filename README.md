# Dementia MRI Classification System

An AI-powered system for classifying brain MRI scans to detect different stages of dementia.

## Features

- Process and classify MRI images into four categories:
  - Non Demented
  - Mild Dementia
  - Moderate Dementia
  - Very Mild Dementia
- Train on large datasets (supports 86K+ images)
- Simple graphical user interface for single image or batch processing
- Visualization of prediction results with confidence scores

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The dataset should be organized as follows:

```
data/
├── Non_Demented/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Mild_Dementia/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Moderate_Dementia/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Very_Mild_Dementia/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Usage

### Training the Model

To train the model on your dataset:

```bash
python train.py --data_dir=path/to/dataset --epochs=20
```

For large datasets (like 86K images), use data generators:

```bash
python train.py --data_dir=path/to/dataset --epochs=20 --use_generators
```

### Running the Application

To run the graphical user interface:

```bash
python run_app.py
```

If you want to specify a different model file:

```bash
python run_app.py --model=path/to/model.h5
```

## System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB+ recommended for large datasets)
- NVIDIA GPU with CUDA support recommended for training
- 5GB free disk space

## License

This project is licensed under the MIT License - see the LICENSE file for details.