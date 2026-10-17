import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
import os
from pathlib import Path
import sys
import tempfile


@contextmanager
def private_runtime_output():
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    try:
        with tempfile.TemporaryFile(mode='w+') as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
    finally:
        for destination, original in zip((1, 2), saved):
            os.dup2(original, destination)
            os.close(original)


def main():
    parser = argparse.ArgumentParser(description='Local SMILES prediction to a private CSV file.')
    parser.add_argument('--list-targets', action='store_true')
    parser.add_argument('--smiles')
    parser.add_argument('--target', default='herg_inhibition')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not (root / 'code' / 'targets.py').is_file():
        parser.error('Prediction code is unavailable.')
    sys.path.insert(0, str(root / 'code'))
    from targets import TARGETS
    if args.list_targets:
        for target in TARGETS.values():
            if target.available:
                print(f'{target.target_id}\t{target.label}')
        return 0
    if not args.smiles or not args.smiles.strip() or args.output is None:
        parser.error('--smiles and --output are required')
    if args.target not in TARGETS or not TARGETS[args.target].available:
        parser.error('Unknown target; use --list-targets.')
    if args.output.exists():
        parser.error('Output exists; choose a new filename.')
    if not args.output.parent.is_dir():
        parser.error('Output parent directory must already exist.')
    try:
        with private_runtime_output():
            from predictor import predict
            result = predict(args.smiles.strip(), TARGETS[args.target])

        row = {key: result.get(key) for key in (
            'input_smiles', 'target_id', 'task_type', 'predicted_class_label',
            'probability_positive', 'positive_label', 'decision_threshold',
            'prediction_value', 'prediction_unit', 'value_interpretation', 'model_warning')}
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    except Exception:
        print('Prediction failed. Check the SMILES, private runtime installation, model assets and output permissions.',
              file=sys.stderr)
        return 1
    print(f'Saved: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
