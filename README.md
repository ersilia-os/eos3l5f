# CLAMP contrastive molecule pretraining embedding

CLAMP is a model trained on molecule-assay pairs, using assay descriptions as text. The current model returns the CLAMP embedding, trained on ChEMBL data. Therefore, contrastive learning is used to learn chemical representations with awareness of a large-scale bioactivity dataset.

This model was incorporated on 2025-08-26.


## Information
### Identifiers
- **Ersilia Identifier:** `eos3l5f`
- **Slug:** `clamp`

### Domain
- **Task:** `Representation`
- **Subtask:** `Featurization`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `Embedding`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `768`
- **Output Consistency:** `Fixed`
- **Interpretation:** CLAMP embedding of the molecule

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| feat_000 | float |  | CLAMP feature 0 of the input molecule |
| feat_001 | float |  | CLAMP feature 1 of the input molecule |
| feat_002 | float |  | CLAMP feature 2 of the input molecule |
| feat_003 | float |  | CLAMP feature 3 of the input molecule |
| feat_004 | float |  | CLAMP feature 4 of the input molecule |
| feat_005 | float |  | CLAMP feature 5 of the input molecule |
| feat_006 | float |  | CLAMP feature 6 of the input molecule |
| feat_007 | float |  | CLAMP feature 7 of the input molecule |
| feat_008 | float |  | CLAMP feature 8 of the input molecule |
| feat_009 | float |  | CLAMP feature 9 of the input molecule |

_10 of 768 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos3l5f.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos3l5f.zip)

### Resource Consumption
- **Model Size (Mb):** `1273`
- **Environment Size (Mb):** `6840`


### References
- **Source Code**: [https://github.com/ml-jku/clamp](https://github.com/ml-jku/clamp)
- **Publication**: [https://arxiv.org/abs/2303.03363](https://arxiv.org/abs/2303.03363)
- **Publication Type:** `Preprint`
- **Publication Year:** `2025`
- **Ersilia Contributor:** [miquelduranfrigola](https://github.com/miquelduranfrigola)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [GPL-3.0-or-later](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos3l5f
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos3l5f
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
