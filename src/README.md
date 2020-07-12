# Code

## How to Use the Configuration File

Note: For `str` type parameters, quotation marks `'`, `"` are not required.

Note: For `list` type parameters, square brakets `[]` are not required.

### General Settings (under `[general]`)
> **`label`**
* Type: `str`
* Description: a label that will be used (along with the date) to name the simulation folder and the results csv file
* Required

### Training Settings (under `[training]`)
> **`plot_freq`**
* Type: `int`
* Description: Frequency of producing and saving loss and accuracy plots 
* Required (recommended: `50`)
> **`print_freq`**
* Type: `int`
* Description: Frequency of printing loss and time information
* Required (recommended: `1`)
> **`save_freq`**
* Type: `int`
* Description: Frequency of adding accuracy data to output csv file
* Required (recommended: `1`)
> **`random_seed`**
* Type: `int`
* Description: seed for random number generator, used to allow reproducible results
* Required
> **`target_radius`**
* Type: `float`
* Description: set a boundary for the target such that if abs(output - target) < target_radius, error = 0
> **`total_epochs`**
* Type: `int`
* Description: total number of training epochs
* Required (recommended: `700`)
> **`anchor_epoch`**
* Type: `int`
* Description: number of epochs *before* adding anchors 
* Required (recommended: `350`, i.e. anchors are added at the 351st epoch)

### Checkpoint Settings (under `[checkpoint]`)
In progress.

### Dataset Settings (under `[dataset]`)
> **`plaut`**
* Type: `str`
* Description: Filepath for the Plaut Dataset file
* Required (recommended: `../dataset/plaut_may07.csv`)
> **`anchor`**
* Type: `str`
* Description: Filepath for the Anchor Dataset file
* Required (recommended: `../dataset/anchors_may07.csv`)
> **`probe`**
* Type: `str`
* Description: Filepath for the Probe Dataset file
* Required (recommended: `../dataset/probes_may07.csv`)
> **`anchor_sets`**
* Type: `list`
* Description: indicates the sets of anchors used to train the model
* Required (example: `1, 2`)
> **`anc_freq`**
* Type: `float`
* Description: Base frequency of anchors that will be adjusted based on dilution
* Requird (recommended: `10`)
> **`track_plaut_types`**
* Type: `list`
* Description: word types to calculate accuracy for in the Plaut Dataset; accuracy for all word types will be calculated if left blank
* Note: To calculate average accuracy across all types, use `All`
* Optional (recommended: `HEC, HRI, HFE, LEC, LFRI, LFE`)
> **`track_anchor_types`**
* Type: `list`
* Description: word types to calculate accuracy for anchors; accuracy for all types anchors will be calculated if left blank
* Note: To calculate average accuracy across all types, use `All`
* Optional (recommended: `<blank>`)
> **`track_probe_types`**
* Type: `list`
* Description: word types to calculate accuracy for probes; accuracy for all types probes will be calculated if left blank
* Note: To calculate average accuracy across all types, use `All`
* Optional (recommended: `<blank>`)

### Optimizer
As many optimizers as desired can defined with the following format:
```
[optimX]
start_epoch = <value>
optimizer = <value>
learning_rate = <value>
momentum = <value>
weight_decay = <value>
```
where `X` is an positive integer (starting from 1) denoting the order of the optimizers to be used.

> **`start_epoch`** 
* Type: `int`
* Description: epoch number to start using the optimizer
* Note: `[optim1]` **must** have `start_epoch = 1` unless a checkpoint is loaded, in which case, the last used optimizer will continue to be used.
* Required
> **`optimizer`**
* Type: `str`
* Description: the optimizer type to be used
* Note: **Must** be either `SGD` or `Adam`
* Required
> **`learning_rate`**
* Type: `float`
* Description: the learning rate for the optimizer
* Required
> **`momentum`**
* Type: `float`
* Description: the momentum for the optimizer
* Required, but value will be ignored for Adam
> **`weight_decay`**
* Type: `float`
* Description: the weight decay for the optimizer
* Required

