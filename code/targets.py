from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class ToxicityTarget:
    target_id: str
    label: str
    task_type: str
    positive_label: str | None = None
    negative_label: str | None = None
    prediction_unit: str | None = None
    value_interpretation: str | None = None

    @property
    def available(self) -> bool:
        return True


TARGETS: dict[str, ToxicityTarget] = {
    "herg_inhibition": ToxicityTarget(
        target_id="herg_inhibition",
        label="Heart (hERG inhibition - lab test)",
        task_type="binary_classification",
        positive_label="hERG inhibitor",
        negative_label="Non-inhibitor",
    ),
    "dictrank": ToxicityTarget(
        target_id="dictrank",
        label="Heart (DICTrank cardiotoxicity - clinical data)",
        task_type="binary_classification",
        positive_label="Cardiotoxicity concern",
        negative_label="No cardiotoxicity concern",
    ),
    "fda_dili": ToxicityTarget(
        target_id="fda_dili",
        label="Liver (FDA DILI - clinical data)",
        task_type="binary_classification",
        positive_label="DILI positive",
        negative_label="DILI negative",
    ),
    "fda_diril": ToxicityTarget(
        target_id="fda_diril",
        label="Kidney (FDA DIRIL - clinical data)",
        task_type="binary_classification",
        positive_label="Renal injury positive",
        negative_label="Renal injury negative",
    ),
    "ames_mutagenicity": ToxicityTarget(
        target_id="ames_mutagenicity",
        label="Ames mutagenicity (lab test)",
        task_type="binary_classification",
        positive_label="Mutagenic",
        negative_label="Non-mutagenic",
    ),
    "solubility": ToxicityTarget(
        target_id="solubility",
        label="Aqueous solubility (lab test)",
        task_type="regression",
        prediction_unit="log10(mol/L)",
        value_interpretation="Higher values indicate greater aqueous solubility.",
    ),
    "bbb_penetration": ToxicityTarget(
        target_id="bbb_penetration",
        label="BBB penetration (lab test)",
        task_type="binary_classification",
        positive_label="BBB penetrant",
        negative_label="Non-penetrant",
    ),
    "cyp1a2_inhibition": ToxicityTarget(
        target_id="cyp1a2_inhibition",
        label="CYP1A2 inhibition (lab test)",
        task_type="binary_classification",
        positive_label="Inhibitor",
        negative_label="Non-inhibitor",
    ),
    "cyp2c19_inhibition": ToxicityTarget(
        target_id="cyp2c19_inhibition",
        label="CYP2C19 inhibition (lab test)",
        task_type="binary_classification",
        positive_label="Inhibitor",
        negative_label="Non-inhibitor",
    ),
    "cyp2c9_inhibition": ToxicityTarget(
        target_id="cyp2c9_inhibition",
        label="CYP2C9 inhibition (lab test)",
        task_type="binary_classification",
        positive_label="Inhibitor",
        negative_label="Non-inhibitor",
    ),
    "cyp2d6_inhibition": ToxicityTarget(
        target_id="cyp2d6_inhibition",
        label="CYP2D6 inhibition (lab test)",
        task_type="binary_classification",
        positive_label="Inhibitor",
        negative_label="Non-inhibitor",
    ),
    "cyp3a4_inhibition": ToxicityTarget(
        target_id="cyp3a4_inhibition",
        label="CYP3A4 inhibition (lab test)",
        task_type="binary_classification",
        positive_label="Inhibitor",
        negative_label="Non-inhibitor",
    ),
}
