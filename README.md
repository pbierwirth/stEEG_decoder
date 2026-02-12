# stEEG_decoder: binary decoding for EEG/MEG data

**stEEG_decoder** is an optimized, lightweight `Python` package build on top of `Scikit-Learn`. It was designed primarily for quick decoding of EEG/MEG time-series data in the context of binary classfication problems. 

The package performs dimensionality reduction of the EEG data via Principal Component Analysis (PCA) and offers optional trial aggregation via the custom-made `subaverage` function to increase the signal-to-noise ratio. Both preprocessing steps have been shown to increase EEG-related classificiation perfromance substantialy (Grootswagers et al., 2016). A Support Vector Machine (SVM) with a linear kernel is used for classification, and the resulting weights are transformed to produce interpretable spatial activation maps (see Haufe et al., 2013).

Two aspects were especially important in the package's creation :

1. **Speed:** It uses **Numba-optimized** AUC computations to speed-up the computation of Temporal Generalization Matrices (TGM; King & Dehaene, 2014)

2. **Interpretability:** Instead of returning raw classifier weights (which are uninterpretable in backward decoding), it computes **Haufe-transformed activation maps** ($A = \Sigma_X W$; Haufe et al., 2013), allowing to visualize the source of neural information used for classification. 

This repository also includes example data from the ERPCore dataset N170 (face vs house) to test this package. Figure 1 illustrates the performance of single trial decoding using the default stEEG_decoder pipeline.  For more information see [**Example Usage**](#example-usage-of-the-package).

<p align="center">   <img src="images/img_decoding_results.png" width="800" height="359.4" title="Decoding Results">   <br>   <em>Figure 1: Temporal decoding with Haufe-transformed Activation Maps and Temporal-Generalization matrix.</em> </p>

---

## Central Features

- **Temporal Generalization:** Efficiently computes training vs. testing performance across all time points, while reducing standard package overhead for AUC calculation by using  `numba` **optimized functions**. 

- **Spatial Mapping:** Returns activation patterns that can be plotted as topographical maps. 

- **Scikit-Learn Compatibility**

  

---

## Getting Started

`stEEG_decoder` can be installed directly from this folder. This allows you to use `import stEEG_decoder` in any script (or Jupyter Notebook) on your PC. 

It is **strongly recommended** to run this pipeline in a clean environment (e.g., via Miniconda or Anaconda) to avoid conflicts with other packages.

Open your terminal (or Anaconda Prompt on Windows) and run:

**1. Create a new environment named 'neuro_decode' with Python 3.9**

  ```Bash
  conda create --name neuro_decode python=3.9
  ```

**2. Activate the environment**
  ```bash
  conda activate neuro_decode
  ```

(Note: If you don't use Conda, you can use standard Python venv: python -m venv venv_name and then activate it.)

**3. Open Terminal / Command Prompt**

Navigate to the unzipped folder (the one containing setup.py).

Mac/Linux:
```Bash
cd ~/Downloads/stEEG_decoder  # example path
```

Windows:
```Bash
cd C:\Users\YourName\Downloads\stEEG_decoder  # example path
```

**4. Install the Package**

Run this command. It will install stEEG_decoder along with all required libraries (MNE, Scikit-Learn, Numba, etc.).

```Bash
pip install .
```

Note for Developers: If you plan to modify the code, use this command instead:

   ```Bash
    pip install -e .
   ```
(The -e flag stands for "editable," meaning changes you make to the code will immediately be reflected without reinstalling.)

**5. Verify Installation**

Open Python and run:
```Python
import stEEG_decoder
print("Installation successful!")
```

---

## Example usage of the package

### Data Requirements

- **Input Data (`X`):** Must be a 3D NumPy array with shape `(n_channels x n_times x n_trials)`.
- **Labels (`y`):** Must be binary integers (`0` and `1`).

### Testing the package

To test the package, download the example data from the **stEEG_decoder** GitHub repository and run `example_wrapper.py`.  The example data provided in this repository is a preprocessed subset of the **ERP CORE** dataset (N170 component. Faces vs. Cars). 

**Source:**  ERP CORE - Compendium of Open Resources and Experiments (Luck, S. J. (2020) **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

For quick example code showing how to process single and multiple subjects, see below:

#### Running the Pipeline on a single subject

```python
import numpy as np
from tqdm import tqdm
from stEEG_decoder.pipeline import decode_temporal_generalization
from stEEG_decoder.cross_validation_eeg import CrossValidator


# Load data 
# X shape: (n_channels, n_times, n_trials)
# y shape: (n_trials,) 
X = ... 
y = ... 

# Initialize Cross-Validation with 5-folds
cv = CrossValidator(y, n_splits=5)

results = []

# Main Decoding Loop
for fold in tqdm(range(len(cv)), desc="Decoding Folds"):
    # Split Data
    x_train, x_test, y_train, y_test = cv.split_data(X, fold_idx=fold)
    
    # Run stEEG_decoder Pipeline
    # Returns: 'diagonal', 'tp_matrix' (TGM), and 'activation_map' (Haufe)
    res = decode_temporal_generalization(x_train, x_test, y_train, y_test)
    results.append(res)

# Aggregate Results
# Average the Temporal Generalization Matrix across folds
mean_tgm = np.mean([r['tp_matrix'] for r in results], axis=0)
# Average the Spatial Maps across folds
mean_activations = np.mean([r['activation_map'] for r in results], axis=0)
```

#### Running the Pipeline across subjects

```python

import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stEEG_decoder.pipeline import decode_temporal_generalization
from stEEG_decoder.cross_validation_eeg import CrossValidator


# Get all EEG data files
files = glob.glob('./eeg_example_data/*pkl')

# Initialize lists to hold the averaged subject data
group_diagonals = []
group_tp_matrices = []
group_activations = []
time = None 

# 3. Main Subject Loop
for file in tqdm(files, desc='Processing subjects'):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        
    x, y = data['x'], data['y']
    if time is None:
        time = data['time'] 
        
    cv = CrossValidator(y, n_splits=5)
    
    # lists for this specific subject's folds
    fold_diags, fold_tps, fold_acts = [], [], []
    
    # Inner Fold Loop
    for fold in tqdm(range(len(cv)), desc='Computing folds', leave=False):
        x_train, x_test, y_train, y_test = cv.split_data(x, fold_idx=fold)
        
        # Run pipeline
        res = decode_temporal_generalization(x_train, x_test, y_train, y_test)
        
        # Store fold results directly
        fold_diags.append(res['diagonal'])
        fold_tps.append(res['tp_matrix'])
        fold_acts.append(res['activation_map'])
        
    # Average across the 5 folds (axis=0) and store in the main group lists
    group_diagonals.append(np.mean(fold_diags, axis=0))
    group_tp_matrices.append(np.mean(fold_tps, axis=0))
    group_activations.append(np.mean(fold_acts, axis=0))

# Convert lists to 3D NumPy arrays for easy plotting
# Array shapes will be: (n_subjects, train_time, test_time)
diagonal = np.array(group_diagonals)
tp_matrix = np.array(group_tp_matrices)
activation = np.array(group_activations)
```

---

## Important Notes

1. This pipeline is optimized for **binary classification**. If you have multi-class data, you must convert them into binary pairs (One-vs-One) or subset your data before running the pipeline.
2. Ensure your input `X` is shaped `(n_channels, n_times, n_trials)`. If your data is `(trials, channels, time)`, simply transform the data via `X.transpose(1, 2, 0)`.

---

## Contact 

If you have any questions or encounter a bug feel free to contact me:

**Email:** philipp.bierwirth@uni-marburg.de

---

## References

> Grootswagers, T., Wardle, S. G. & Carlson, T. A. (2016). Decoding  Dynamic Brain Patterns from Evoked Responses: A Tutorial on Multivariate Pattern Analysis Applied to Time Series Neuroimaging Data. *Journal Of Cognitive Neuroscience*, *29*(4), 677–697. https://doi.org/10.1162/jocn_a_01068
>
> Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J., Blankertz,  B. & Bießmann, F. (2013). On the interpretation of weight vectors of linear models in multivariate neuroimaging. *NeuroImage*, *87*, 96–110. https://doi.org/10.1016/j.neuroimage.2013.10.067
>
> King, J. & Dehaene, S. (2014). Characterizing the dynamics of mental representations: the temporal generalization method. *Trends in Cognitive Sciences*, *18*(4), 203–210. https://doi.org/10.1016/j.tics.2014.01.002 
>
> Luck, S. J. (2020). ERP CORE (Compendium of Open Resources and Experiments) [Dataset]. In *UC Davis*. https://doi.org/10.18115/d5jw4r