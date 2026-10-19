"""Exporta y verifica la evaluación de un modelo EP2 guardado, sin entrenarlo."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, default=Path('output/jason_ep2'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    run = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    out = args.output_dir if args.output_dir else run / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / 'src'))
    from sign_mlp.data import prepare_sign_mnist_cnn_data
    from tensorflow import keras

    metrics = json.loads((run / 'reports/cnn_metrics.json').read_text(encoding='utf-8'))
    data = prepare_sign_mnist_cnn_data(root / 'data/raw', random_state=42)
    model_path = run / 'models/cnn_sign_language.keras'
    model = keras.models.load_model(model_path, compile=False)
    pred = np.argmax(model.predict(data['X_test'], batch_size=128, verbose=0), axis=1)
    truth = data['y_test']
    names = data['metadata']['class_names']
    labels = list(range(len(names)))
    report = classification_report(truth, pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(truth, pred, labels=labels)
    calculated = {'test_accuracy_check': float(accuracy_score(truth, pred)),
                  'test_precision_macro': report['macro avg']['precision'],
                  'test_recall_macro': report['macro avg']['recall'],
                  'test_f1_macro': report['macro avg']['f1-score']}
    for key, value in calculated.items():
        if not np.isclose(value, metrics[key], rtol=0, atol=1e-7):
            raise ValueError(f'La evaluación guardada y el modelo difieren en {key}: {value} / {metrics[key]}')
    assert int(cm.sum()) == metrics['test_images'] == len(truth)
    assert int(cm.trace()) == int(np.sum(truth == pred))

    def save_csv(filename, header, rows):
        with (out / filename).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    save_csv('cnn_predictions.csv', ['test_index_0_based', 'true_letter', 'predicted_letter', 'correct'],
             ((i, names[int(t)], names[int(p)], int(t == p)) for i, (t, p) in enumerate(zip(truth, pred))))
    save_csv('cnn_confusion_matrix.csv', ['real/predicted'] + names,
             ([name] + [int(v) for v in row] for name, row in zip(names, cm)))
    save_csv('cnn_per_class_metrics.csv', ['class', 'precision', 'recall', 'f1', 'support'],
             ((name, report[name]['precision'], report[name]['recall'], report[name]['f1-score'],
               int(report[name]['support'])) for name in names))
    errors = sorted([(names[i], names[j], int(cm[i,j])) for i in labels for j in labels
                     if i != j and cm[i,j] > 0], key=lambda row: (-row[2], row[0], row[1]))
    save_csv('cnn_top_confusions.csv', ['true_letter', 'predicted_letter', 'count'], errors)
    (out / 'cnn_classification_report_exact.txt').write_text(
        classification_report(truth, pred, labels=labels, target_names=names, digits=4, zero_division=0), encoding='utf-8')
    test = json.loads((root / 'reports/jenaro_integracion_test.json').read_text(encoding='utf-8'))
    training = json.loads((root / 'reports/jenaro_entrenamiento.json').read_text(encoding='utf-8'))
    selected = next(row for row in training['resultados_validacion'] if row['modelo'] == training['seleccionado'])
    fields = ['model','input_format','parameters','epochs_completed','fit_seconds','validation_accuracy',
              'test_accuracy','test_precision_macro','test_recall_macro','test_f1_macro','validation_test_gap_pp','timing_source']
    comparison = [
        ['MLP C_grande (EP1)','vector 784',selected['parametros'],selected['epocas'],selected['segundos_fit'],
         selected['val_accuracy'],test['test_accuracy'],test['test_precision_macro'],test['test_recall_macro'],
         test['test_f1_macro'],100*(selected['val_accuracy']-test['test_accuracy']),'reports/jenaro_entrenamiento.json'],
        ['CNN Jason (EP2)','tensor 28x28x1',metrics['parameters'],metrics['epochs_completed'],metrics['fit_seconds'],
         metrics['validation_accuracy'],calculated['test_accuracy_check'],calculated['test_precision_macro'],
         calculated['test_recall_macro'],calculated['test_f1_macro'],
         100*(metrics['validation_accuracy']-calculated['test_accuracy_check']),'output/jason_ep2/reports/cnn_metrics.json']]
    save_csv('mlp_cnn_comparison_exact.csv',fields,comparison)
    nbpath = run / '03_sign_language_mnist_cnn_ep2_ejecutado.ipynb'
    nb = json.loads(nbpath.read_text(encoding='utf-8'))
    code_cells = [c for c in nb['cells'] if c['cell_type']=='code']
    errors_nb = [o for c in code_cells for o in c.get('outputs',[]) if o['output_type']=='error']
    assert not errors_nb and all(c.get('execution_count') is not None for c in code_cells)
    assert (run/'reports/cnn_metrics.json').read_bytes()==(run/'reports/cnn_metrics_notebook.json').read_bytes()
    summary = {'responsable':'Jason Aguilera','modelo_reentrenado':False,'predicciones_verificadas':len(truth),
               'aciertos':int(cm.trace()),'errores':int(cm.sum()-cm.trace()),'metricas':calculated,
               'accuracy_delta_pp':100*(calculated['test_accuracy_check']-test['test_accuracy']),
               'f1_macro_delta':calculated['test_f1_macro']-test['test_f1_macro'],
               'brecha_validacion_test_pp':100*(metrics['validation_accuracy']-calculated['test_accuracy_check']),
               'top_confusions':[dict(real=a,prediccion=b,casos=n) for a,b,n in errors],
               'por_clase':{name:report[name] for name in names},
               'celdas_codigo_ejecutadas':len(code_cells),'errores_notebook':len(errors_nb),
               'json_metricas_coinciden':True,
               'modelo_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
               'test_sha256':hashlib.sha256((root/'data/raw/sign_mnist_test.csv').read_bytes()).hexdigest()}
    (out/'jason_evaluacion_verificada.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in summary.items() if key not in ('por_clase','top_confusions')},ensure_ascii=False,indent=2))
    print('PRINCIPALES CONFUSIONES:', errors[:8])


if __name__ == '__main__':
    main()
