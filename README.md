# MorphADMET

MorphADMET is a platform that integrates molecular structure and cell morphology data to predict ADMET properties using pretrained models.

## How to Run

Requires Python 3.12.

```bash
# List available prediction targets
python predict.py --list-targets

# Predict hERG inhibition for aspirin and save the result
python predict.py --smiles 'CC(=O)Oc1ccccc1C(=O)O' --target herg_inhibition --output result.csv
```

Use an available target and a new output filename for each prediction.

**Web application coming soon.**