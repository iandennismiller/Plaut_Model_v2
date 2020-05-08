# Dataset

## Format for Dataset Files
### General Requirements
**All** dataset files must consist of the following columns:
 > `orth`: containing the orthography of the words
 > - The orthography is **not** case sensitive

 > `phon`: containing the phonology of the words
 > - Must be in the format `/<phon>/` (e.g. `/As/`)
 > - The phonology **is** case sensitive

 > `type`: containing the word type

The Plaut and Anchor Dataset files have additional requirements, which are provided below.

### Plaut Dataset Files
In addition, Plaut Dataset files must consist of the following column:
> `freq`: containing the word frequency
> - frequency values must be either integers or floats

Optionally, Plaut Dataset files may consist of the following column:
> `log_freq`: containing a scaled frequency based on a natural logarithm
> - This frequency is calulated by `ln(F+2)` where `ln` denotes the natural logarithm and `F` denotes the frewuency of a word
> - If the `log_freq` column is not provided, the simulation code will automatically calculate this column

#### Sample Plaut Dataset File
`plaut_may07.csv`: Dataset file for Plaut Dataset

### Anchor Dataset Files
In addition, Anchor Dataset files must consist of the following column:
> `set`: containing an integer value to separate anchors by groups

#### Sample Anchor Dataset File
`anchor_may07.csv`: Dataset file for the anchors used in the Jan 2020 paper

### Probe Dataset Files

#### Sample Probe Dataset File
`probe_may07.csv`: Dataset file for the probes used in the Jan 2020 paper
