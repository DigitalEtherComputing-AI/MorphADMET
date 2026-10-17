from standardization import standardize_structure
import model_loader


def predict(smiles, target):
    standardized = standardize_structure(smiles)
    if not standardized:
        raise ValueError('Invalid SMILES')
    value = model_loader.load(target.target_id).predict_smiles(standardized)
    binary = target.task_type == 'binary_classification'
    return dict(
        input_smiles=smiles, target_id=target.target_id, task_type=target.task_type,
        predicted_class_label=(target.positive_label if value >= .5 else target.negative_label)
        if binary else None,
        probability_positive=value if binary else None,
        positive_label=target.positive_label if binary else None,
        decision_threshold=.5 if binary else None,
        prediction_value=None if binary else value,
        prediction_unit=target.prediction_unit if not binary else None,
        value_interpretation=target.value_interpretation, model_warning='Reduced model selection; predictive accuracy has not been independently evaluated.')
