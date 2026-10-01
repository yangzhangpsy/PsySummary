import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from scipy.integrate import quad

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/psysummary-test-matplotlib')

from PyQt5.QtCore import QEvent, QThread, Qt, pyqtSignal
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QApplication, QComboBox, QDockWidget, QLineEdit, QWidget
from PyQt5 import sip

from app.lib.cognitiveModelDialog import CognitiveModelDialog
from app.cognitiveModelSpec import (
    ACCURACY_CODING, LBA_MODEL, RATCLIFF_MODEL, RESPONSE_CODING, RDM_MODEL,
    cognitive_model_reference_text,
    make_model_specification,
    validate_model_specification, validate_model_data, model_result_parameters,
)
from app.cognitiveModels import (
    _wald_single_pdf, diffusion_cdf, diffusion_pdf, diffusion_quantile,
    fit_cognitive_model, fitted_accuracy_pdf, fitted_response_pdf,
    lba_cdf, lba_pdf, lba_quantile, lba_random,
    rdm_cdf, rdm_pdf, rdm_quantile,
)
from app.lib.draggablelistwidget import DraggableListWidget, MODEL_SPEC_ROLE
from app.lib.fitCognitiveModelThread import FitCognitiveModelThread
from app.fitCancellation import FitCancelled, raise_if_fit_cancelled
from app.lib.dotted_spinner import FRAME_INTERVAL_MS


class CognitiveModelNumericalTests(unittest.TestCase):
    def test_cognitive_fit_honors_cancellation_before_validation(self):
        """A requested cancellation exits without being converted into a fitting failure."""
        with self.assertRaises(FitCancelled):
            fit_cognitive_model(
                pd.DataFrame(), {}, cancel_check=lambda: True)

    def test_standalone_export_does_not_require_gui_cancellation_module(self):
        """Command-line exports load without copying the GUI-only cancellation helper."""
        from app.lib.scriptDock import OutputTextEdit

        with tempfile.TemporaryDirectory() as temporary_directory:
            script_path = os.path.join(temporary_directory, 'analysis.py')
            editor = OutputTextEdit()
            with patch(
                    'app.lib.scriptDock.QFileDialog.getSaveFileName',
                    return_value=(script_path, 'Python Files (*.py)')):
                editor.export()
            editor.close()

            self.assertTrue(os.path.isfile(
                os.path.join(temporary_directory, 'cognitiveModels.py')))
            self.assertFalse(os.path.exists(
                os.path.join(temporary_directory, 'fitCancellation.py')))

            probe = subprocess.run(
                [
                    sys.executable,
                    '-c',
                    'import importlib.abc, sys\n'
                    'class BlockApp(importlib.abc.MetaPathFinder):\n'
                    '    def find_spec(self, fullname, path=None, target=None):\n'
                    "        if fullname == 'app' or fullname.startswith('app.'):\n"
                    "            raise ModuleNotFoundError('blocked app package')\n"
                    '        return None\n'
                    'sys.meta_path.insert(0, BlockApp())\n'
                    'import cognitiveModels\n'
                    'cognitiveModels.raise_if_fit_cancelled(None)\n',
                ],
                cwd=temporary_directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_diffusion_boundary_densities_sum_to_one(self):
        parameters = dict(a=1.0, v=1.0, t0=0.2, z=0.5, s=1.0)
        lower = quad(lambda value: diffusion_pdf(value, 'lower', **parameters), 0.2, 8.0)[0]
        upper = quad(lambda value: diffusion_pdf(value, 'upper', **parameters), 0.2, 8.0)[0]
        self.assertAlmostEqual(lower + upper, 1.0, places=6)
        self.assertAlmostEqual(upper, 1.0 / (1.0 + math.exp(-1.0)), places=6)

    def test_lba_race_densities_sum_to_one(self):
        parameters = dict(A=0.5, b=1.0, t0=0.2, mean_v=[2.0, 1.0], sd_v=[1.0, 1.0])
        mass = sum(
            quad(lambda value: lba_pdf(value, response, **parameters), 0.2, 10.0, limit=200)[0]
            for response in (1, 2)
        )
        self.assertAlmostEqual(mass, 1.0, places=3)

    def test_lba_and_rdm_match_rtdists_reference_equations(self):
        times = np.array([0.3, 0.5, 1.0])
        expected_lba = np.array([0.014595309951, 2.365578031381, 0.106351748727])
        expected_rdm = np.array([2.050150052311, 1.106841665809, 0.080314304635])
        np.testing.assert_allclose(
            lba_pdf(times, 1, A=0.5, b=1.0, t0=0.2,
                    mean_v=[2.0, 1.0], sd_v=[1.0, 1.0]),
            expected_lba, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            rdm_pdf(times, 1, A=0.5, b=1.0, t0=0.2, v=[2.0, 1.0]),
            expected_rdm, rtol=1e-10, atol=1e-12)

    def test_models_match_rtdists_0_12_0_reference_grid(self):
        """Compare representative PDF, CDF, and quantile values generated by rtdists."""
        times = np.array([0.3, 0.5, 0.8, 1.2])
        ddm_cases = [
            (
                dict(a=1, v=1, t0=0.2, z=0.4, d=0, sz=0, sv=0, st0=0, s=1),
                {
                    'lower': [1.44574360865521, 0.39794185624245,
                              0.0768298761813114, 0.00873642995425235],
                    'upper': [2.16697119497287, 1.05066735898987,
                              0.208773619004941, 0.0237480569514612],
                },
            ),
            (
                dict(a=1.3, v=-0.7, t0=0.25, z=0.6, d=0.04,
                     sz=0.2, sv=0.5, st0=0.1, s=0.8),
                {
                    'lower': [0.000999314435384243, 1.477045523869939,
                              0.7398166164757115, 0.2706998221561852],
                    'upper': [0.007619822031239648, 0.2984190310594501,
                              0.2003152223105419, 0.08824359330453652],
                },
            ),
        ]
        for parameters, expected_by_response in ddm_cases:
            for response, expected in expected_by_response.items():
                np.testing.assert_allclose(
                    diffusion_pdf(times, response, **parameters), expected,
                    rtol=2e-7, atol=2e-6)

        self.assertAlmostEqual(
            diffusion_cdf(0.8, 'lower', **ddm_cases[0][0]), 0.3490043787490383, places=6)
        self.assertAlmostEqual(
            diffusion_cdf(0.8, 'upper', **ddm_cases[0][0]), 0.5984417143175156, places=6)
        self.assertAlmostEqual(
            diffusion_cdf(0.8, 'lower', **ddm_cases[1][0]), 0.4878444783361123, places=5)
        self.assertAlmostEqual(
            diffusion_cdf(0.8, 'upper', **ddm_cases[1][0]), 0.1120607773498064, places=5)
        self.assertAlmostEqual(
            diffusion_quantile(
                0.25, 'lower', interval=(0, 10), scale_probability=True, **ddm_cases[0][0]),
            0.272685749066332, delta=4e-4)
        self.assertAlmostEqual(
            diffusion_quantile(
                0.25, 'upper', interval=(0, 10), scale_probability=True, **ddm_cases[0][0]),
            0.326229266143871, delta=4e-4)

        race_cases = [
            (
                lba_pdf, lba_cdf, lba_quantile,
                dict(A=0.5, b=1, t0=0.2, mean_v=[2, 1], sd_v=[1, 1], st0=0),
                {
                    1: [0.014595309950927, 2.36557803138125,
                        0.340573888763559, 0.0422215572069633],
                    2: [0.000393390520297204, 0.847514712927402,
                        0.19829759899821, 0.0299886741090591],
                },
                {1: 0.653348024084254, 2: 0.234807610280719},
                {1: 0.443828976450131, 2: 0.482696283868811},
            ),
            (
                rdm_pdf, rdm_cdf, rdm_quantile,
                dict(A=0.5, b=1, t0=0.2, v=[2, 1], s=1, st0=0),
                {
                    1: [2.05015005231053, 1.10684166580932,
                        0.216585178372412, 0.0319635581779654],
                    2: [1.17225895712821, 0.578500227320573,
                        0.103062762327004, 0.0139334573815304],
                },
                {1: 0.606165514918286, 2: 0.327912097097907},
                {1: 0.333957945278325, 2: 0.328692288397977},
            ),
        ]
        for pdf, cdf, quantile, parameters, expected_pdf, expected_cdf, expected_quantile in race_cases:
            for response in (1, 2):
                np.testing.assert_allclose(
                    pdf(times, response, **parameters), expected_pdf[response],
                    rtol=2e-9, atol=1e-11)
                self.assertAlmostEqual(
                    cdf(0.8, response, **parameters), expected_cdf[response], places=6)
                self.assertAlmostEqual(
                    quantile(
                        0.25, response, interval=(0, 10),
                        scale_probability=True, **parameters),
                    expected_quantile[response], places=6)

    def test_rdm_matches_published_fixed_start_wald_density(self):
        rt = np.array([0.1, 0.3, 0.8, 1.5])
        b = 1.0
        v = 2.0
        expected = b * (2.0 * math.pi * rt ** 3) ** -0.5 * np.exp(
            -((v * rt - b) ** 2) / (2.0 * rt))
        np.testing.assert_allclose(_wald_single_pdf(rt, 0.0, b, v, 1.0), expected, rtol=1e-10)

    def test_rdm_race_densities_sum_to_one(self):
        parameters = dict(A=0.4, b=1.0, t0=0.25, v=[2.5, 1.5], s=1.0)
        mass = sum(
            quad(lambda value: rdm_pdf(value, response, **parameters), 0.25, 10.0, limit=200)[0]
            for response in (1, 2)
        )
        self.assertAlmostEqual(mass, 1.0, places=4)

    def test_all_fixed_model_evaluates_without_optimizer(self):
        dataframe = pd.DataFrame({
            'rt': [0.36, 0.42, 0.50, 0.61, 0.39, 0.55],
            'choice': ['left', 'right', 'right', 'left', 'right', 'left'],
            'correct': [1, 1, 0, 1, 1, 0],
        })
        specification = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'], minimum_rt=0.36,
            accuracy_variable='correct')
        specification['optimizer']['seed'] = 73
        specification['response_mapping'] = {'left': 'upper', 'right': 'lower'}
        for parameter in specification['parameters']:
            parameter['mode'] = 'fixed'
        fit = fit_cognitive_model(dataframe, specification)
        self.assertTrue(fit['converged'])
        self.assertIn('without optimization', fit['message'])
        self.assertEqual(fit['response_counts'], {'left': 3, 'right': 3})
        np.testing.assert_array_equal(fit['accuracy'], [1, 1, 0, 1, 1, 0])
        self.assertAlmostEqual(fit['accuracy_rate'], 4 / 6)
        self.assertEqual(fit['random_seed'], 73)
        self.assertTrue(np.isfinite(fit['log_likelihood']))
        self.assertEqual(len(fit['parameters']), len(specification['parameters']))
        self.assertEqual(fit['parameter_names'], [row['name'] for row in specification['parameters']])
        self.assertNotIn('conditional_fits', fit)

    def test_accuracy_and_response_coding_are_equivalent_for_fixed_parameters(self):
        """Boundary relabeling must preserve the trial likelihood under both codings."""
        frame = pd.DataFrame({
            'rt': [.35, .42, .5, .61, .7, .82],
            'choice': ['left', 'right', 'left', 'right', 'left', 'right'],
            'correct': [1, 1, 1, 1, 0, 0],
        })
        spec = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'],
            minimum_rt=.3, accuracy_variable='correct')
        for row in spec['parameters']:
            row['mode'] = 'fixed'
            if row['name'] == 'z':
                row['value'] = 0.65
            elif row['name'] == 'd':
                row['value'] = 0.04
        accuracy_spec = deepcopy(spec)
        accuracy_spec['boundary_coding'] = ACCURACY_CODING
        response_spec = deepcopy(spec)
        response_spec['boundary_coding'] = RESPONSE_CODING
        first = fit_cognitive_model(frame, accuracy_spec)
        second = fit_cognitive_model(frame, response_spec)

        np.testing.assert_allclose(first['parameters'], second['parameters'])
        self.assertAlmostEqual(first['log_likelihood'], second['log_likelihood'])
        self.assertAlmostEqual(first['accuracy_rate'], 2 / 3)
        self.assertAlmostEqual(second['accuracy_rate'], 2 / 3)
        np.testing.assert_allclose(first['correct_response_weights'], [.5, .5])
        self.assertEqual(len(first['parameters']), len(model_result_parameters(spec)))
        self.assertNotIn('conditional_fits', first)

    def test_partial_missing_accuracy_is_rejected_before_fitting(self):
        """Every retained binary trial needs valid accuracy under either coding."""
        frame = pd.DataFrame({'rt': [.35, .45, .5, .6], 'choice': ['left', 'right', 'left', 'right'],
                              'correct': [np.nan, 0, 1, 0]})
        spec = make_model_specification(RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'],
                                        minimum_rt=.3, accuracy_variable='correct')
        for row in spec['parameters']:
            row['mode'] = 'fixed'
        spec['optimizer']['seed'] = 73
        with self.assertRaisesRegex(ValueError, 'no missing values.*Filter Data'):
            fit_cognitive_model(frame, spec)

    def test_nonbinary_accuracy_is_rejected_with_filter_guidance(self):
        """Accuracy values are never guessed or silently recoded."""
        frame = pd.DataFrame({
            'rt': [.35, .45, .55, .65],
            'choice': ['left', 'right', 'left', 'right'],
            'correct': [1, 0, 2, 1],
        })
        spec = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'],
            minimum_rt=.3, accuracy_variable='correct')
        with self.assertRaisesRegex(ValueError, r'only 0 \(error\) and 1 \(correct\).*Filter Data'):
            fit_cognitive_model(frame, spec)

    def test_worker_reports_one_shared_parameter_set(self):
        """GUI result columns must contain one response-coded model parameter set."""
        frame = pd.DataFrame({'rt': [.35, .45, .5, .6], 'choice': ['left', 'right', 'left', 'right'],
                              'correct': [1, 0, 0, 1]})
        spec = make_model_specification(RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'],
                                        minimum_rt=.3, accuracy_variable='correct')
        for row in spec['parameters']:
            row['mode'] = 'fixed'
        outputs = []
        worker = FitCognitiveModelThread(frame, spec, [], [])
        worker.finished.connect(lambda *args: outputs.append(args))
        worker._process_model()
        values, names, _, _, records = outputs[0]
        self.assertEqual(len(values.iloc[0]), len(names))
        self.assertIn(' a (Fixed)', names[0])
        self.assertFalse(any('Correct response=' in name for name in names))
        self.assertEqual(len(records[0]['parameters']), len(spec['parameters']))
        self.assertNotIn('conditional_fits', records[0])

    def test_group_with_one_observed_boundary_fits_with_warning(self):
        """Globally binary data may contain groups with one observed response boundary."""
        frame = pd.DataFrame({
            'rt': [.35, .45, .55, .65],
            'choice': ['left', 'left', 'right', 'right'],
            'correct': [1, 1, 1, 1],
            'condition': ['A', 'A', 'B', 'B'],
        })
        spec = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'],
            minimum_rt=.3, accuracy_variable='correct')
        for row in spec['parameters']:
            row['mode'] = 'fixed'
        outputs = []
        worker = FitCognitiveModelThread(frame, spec, ['condition'], [])
        worker.finished.connect(lambda *args: outputs.append(args))
        worker._process_model()
        records = outputs[0][-1]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record['converged'] for record in records))
        self.assertTrue(all(record['fit_warnings'] for record in records))
        self.assertTrue(all('Only one response boundary' in record['message'] for record in records))

    def test_constrained_optimizer_moves_from_a_nonoptimal_start(self):
        dataframe = lba_random(
            250, A=0.5, b=1.0, t0=0.2,
            mean_v=[2.0, 1.0], sd_v=[1.0, 1.0], random_state=8102)
        specification = make_model_specification(
            LBA_MODEL, 'rt', 'response', [1, 2], minimum_rt=dataframe['rt'].min())
        for parameter in specification['parameters']:
            parameter['mode'] = 'fixed'
            if parameter['name'] in {'A', 'b', 't0', 'mean_v[1]', 'mean_v[2]'}:
                parameter['mode'] = 'free'
        initial = {
            parameter['name']: parameter['value'] for parameter in specification['parameters']
        }
        specification['optimizer'].update(starts=1, seed=73, max_iterations=1000)
        fit = fit_cognitive_model(dataframe, specification)
        fitted = dict(zip(fit['parameter_names'], fit['parameters']))
        self.assertTrue(fit['converged'])
        self.assertEqual(fit['optimizer_method'], 'SLSQP')
        self.assertGreater(
            max(abs(fitted[name] - initial[name])
                for name in ('A', 'b', 't0', 'mean_v[1]', 'mean_v[2]')),
            0.01,
        )

    def test_identifiability_constraints_are_enforced(self):
        specification = make_model_specification(
            RDM_MODEL, 'rt', 'choice', [1, 2], minimum_rt=0.3)
        next(parameter for parameter in specification['parameters'] if parameter['name'] == 's')['mode'] = 'free'
        with self.assertRaisesRegex(ValueError, 'must be Fixed'):
            validate_model_specification(specification, ['rt', 'choice'])

        specification = make_model_specification(
            LBA_MODEL, 'rt', 'choice', [1, 2], minimum_rt=0.3)
        for parameter in specification['parameters']:
            if parameter['name'].startswith('sd_v['):
                parameter['mode'] = 'free'
        with self.assertRaisesRegex(ValueError, 'At least one LBA sd_v'):
            validate_model_specification(specification, ['rt', 'choice'])

        specification = make_model_specification(
            LBA_MODEL, 'rt', 'choice', [1, 2], minimum_rt=0.3)
        invalid_sd = next(parameter for parameter in specification['parameters']
                          if parameter['name'] == 'sd_v[1]')
        invalid_sd.update(value=0, lower=-1)
        with self.assertRaisesRegex(ValueError, 'standard deviations must be greater than zero'):
            validate_model_specification(specification, ['rt', 'choice'])

        specification = make_model_specification(
            RDM_MODEL, 'rt', 'choice', [1, 2], minimum_rt=0.3)
        invalid_scale = next(parameter for parameter in specification['parameters']
                             if parameter['name'] == 's')
        invalid_scale.update(value=0, lower=-1)
        with self.assertRaisesRegex(ValueError, 'diffusion scale s must be greater than zero'):
            validate_model_specification(specification, ['rt', 'choice'])

    def test_model_reference_output_identifies_paper_package_and_runtime(self):
        expected_dois = {
            RATCLIFF_MODEL: '10.1162/neco.2008.12-06-420',
            LBA_MODEL: '10.1016/j.cogpsych.2007.12.002',
            RDM_MODEL: '10.3758/s13423-020-01719-6',
        }
        for model, doi in expected_dois.items():
            reference = cognitive_model_reference_text(model)
            self.assertIn(doi, reference)
            self.assertIn('rtdists', reference)
            self.assertIn('0.12-0', reference)
            self.assertIn('not invoked at runtime', reference)
            self.assertIn('SLSQP', reference)

    def test_fit_worker_emits_model_reference_once_per_analysis(self):
        dataframe = pd.DataFrame({
            'rt': [0.35, 0.42, 0.51, 0.63],
            'choice': ['left', 'right', 'left', 'right'],
            'correct': [1, 1, 0, 0],
        })
        specification = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'], minimum_rt=0.35,
            accuracy_variable='correct')
        for parameter in specification['parameters']:
            parameter['mode'] = 'fixed'
        messages = []
        worker = FitCognitiveModelThread(dataframe, specification, [], [])
        worker.fitStatus.connect(
            lambda _level, message, _show_time: messages.append(message))
        worker._process_model()
        reference_messages = [
            message for message in messages if message.startswith('Model reference:')
        ]
        self.assertEqual(len(reference_messages), 1)
        self.assertIn('10.1162/neco.2008.12-06-420', reference_messages[0])

    def test_invalid_draft_roundtrip_and_filter_preserve_parameters(self):
        """Saving a three-response draft must survive filtering and incomplete numeric input."""
        import ast
        frame = pd.DataFrame({
            'rt': [.4, .5, .6], 'choice': ['left', 'right', 'timeout'],
            'correct': [1, 1, 0],
        })
        dialog = CognitiveModelDialog(frame, RATCLIFF_MODEL, 'rt')
        dialog.response_combo.setCurrentText('choice')
        dialog.accuracy_combo.setCurrentText('correct')
        dialog.parameter_table.cellWidget(0, 2).setText('unfinished')
        self.assertIn('3 configured values', dialog.validation_label.text())
        with patch('app.lib.cognitiveModelDialog.QMessageBox.warning') as warning:
            dialog._accept_if_valid()
        warning.assert_not_called()
        self.assertEqual(dialog.result(), dialog.Accepted)
        draft = ast.literal_eval(repr(dialog.specification()))
        reopened = CognitiveModelDialog(frame.iloc[:2], RATCLIFF_MODEL, 'rt', draft)
        self.assertEqual(reopened.parameter_table.cellWidget(0, 2).text(), 'unfinished')
        self.assertEqual(reopened.specification()['response_values'], ['left', 'right'])
        with self.assertRaisesRegex(ValueError, "Parameter 'a'"):
            validate_model_data(reopened.specification(), frame.iloc[:2])
        reopened.parameter_table.cellWidget(0, 2).setText('1.2')
        validate_model_data(reopened.specification(), frame.iloc[:2])
        dialog.close()
        reopened.close()

    def test_current_data_and_each_group_are_validated_before_fit(self):
        """New response values and missing group responses must never be silently omitted."""
        frame = pd.DataFrame({
            'rt': [.4, .5, .6], 'choice': ['left', 'right', 'timeout'],
            'correct': [1, 1, 0],
        })
        spec = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'], accuracy_variable='correct')
        with self.assertRaisesRegex(ValueError, 'contains 3 values.*timeout'):
            fit_cognitive_model(frame, spec)
        frame = frame.iloc[:2].assign(condition=['A', 'B'])
        validate_model_data(spec, frame, ['condition'])
        spec['response_mapping']['left'] = 'upper'
        with self.assertRaisesRegex(ValueError, 'once each to lower and upper'):
            validate_model_data(spec, frame)

    def test_run_blocks_invalid_draft_before_creating_results(self):
        """The main Run action must reject drafts before creating a result window."""
        from PsySummary import PsyData
        window = PsyData()
        window.data = pd.DataFrame({
            'rt': [.4, .5, .6], 'choice': ['left', 'right', 'timeout'],
            'correct': [1, 1, 0],
        })
        spec = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right', 'timeout'],
            accuracy_variable='correct')
        window.data_list.addItem(f'rt@{RATCLIFF_MODEL}')
        window.data_list.item(0).setData(MODEL_SPEC_ROLE, spec)
        with patch('PsySummary.MessageBox.warning') as warning, \
                patch('PsySummary.PivotedDataWidget') as results:
            window.runSummary()
            results.assert_not_called()
            self.assertEqual(warning.call_args.args[1], 'Invalid Model Settings')
            self.assertIn('contains 3 values', warning.call_args.args[2])
        window.close()

    def test_cognitive_run_records_cdf_pooling_omegas_once(self):
        """Model setup and preflight filtering must not duplicate omega script lines."""
        from PsySummary import PsyData
        from app.psyDataFunc import PsyDataFunc
        from app.tool import StatisticTool

        window = PsyData()
        window.data = pd.DataFrame({
            'rt': [.35, .45, .55, .65],
            'choice': [6.0, 5.0, 6.0, 5.0],
            'correct': [1, 1, 0, 0],
        })
        specification = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', [6.0, 5.0],
            accuracy_variable='correct')
        window.data_list.addItem(f'rt@{RATCLIFF_MODEL}')
        window.data_list.item(0).setData(MODEL_SPEC_ROLE, specification)
        window.filter_list.addItem('choice: = 6.0 = 5.0')

        def build_result(dataframe, row_vars, col_vars, _targets, rules):
            StatisticTool.filterData(
                row_vars, col_vars, dataframe, rules, record_script=True)
            result_widget = QWidget()
            result_widget.analysisFinished = MagicMock()
            result_widget.analysisFailed = MagicMock()
            return result_widget

        with patch.object(PsyDataFunc, 'genScript') as generate_script, \
                patch('PsySummary.PivotedDataWidget', side_effect=build_result):
            window.runSummary()

        omega_lines = [
            call.args[0] for call in generate_script.call_args_list
            if call.args and isinstance(call.args[0], str)
            and call.args[0].startswith('cdfPoolingOmegas = ')
        ]
        self.assertEqual(omega_lines, ['cdfPoolingOmegas = [-1]'])
        window.model_fit_running = False
        window.close()

    def test_pivoted_cognitive_fit_runs_asynchronously_and_reports_progress(self):
        """The Results widget returns before fitting and runs its optimizer off the GUI thread."""
        from app.cognitiveModels import fit_cognitive_model
        from app.lib.pivotedDataWidget import PivotedDataWidget
        from app.psyDataFunc import PsyDataFunc

        dataframe = pd.DataFrame({
            'rt': [.35, .40, .45, .50, .55, .60, .65, .70],
            'choice': [6.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 5.0],
            'correct': [1, 1, 0, 0, 1, 1, 0, 0],
        })
        specification = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', [6.0, 5.0],
            accuracy_variable='correct')
        for parameter in specification['parameters']:
            parameter['mode'] = 'fixed'
        target = {'model_specification': specification}
        main_thread_id = int(QThread.currentThreadId())
        fitting_thread_ids = []

        def record_thread(*args, **kwargs):
            fitting_thread_ids.append(int(QThread.currentThreadId()))
            return fit_cognitive_model(*args, **kwargs)

        with patch.object(PsyDataFunc, 'genScript'), \
                patch.object(PsyDataFunc, 'printOut') as print_out, \
                patch(
                    'app.lib.fitCognitiveModelThread.fit_cognitive_model',
                    side_effect=record_thread):
            result_widget = PivotedDataWidget(
                dataframe, [], [], [target], [])
            finished = QSignalSpy(result_widget.analysisFinished)
            failed = QSignalSpy(result_widget.analysisFailed)
            self.assertIsNone(result_widget.table)
            self.assertIsNone(result_widget.fit_dist_thread)
            self.assertTrue(finished.wait(5000))

        self.assertEqual(len(failed), 0)
        self.assertIsNotNone(result_widget.table)
        self.assertTrue(fitting_thread_ids)
        self.assertTrue(all(thread_id != main_thread_id for thread_id in fitting_thread_ids))
        messages = [call.args[0] for call in print_out.call_args_list]
        self.assertTrue(any('Model fitting started' in message for message in messages))
        self.assertTrue(any('Fitting 1/1' in message for message in messages))
        self.assertTrue(any('Finished 1/1' in message for message in messages))
        self.assertTrue(any('Model fitting finished' in message for message in messages))
        lifecycle_calls = [
            call for call in print_out.call_args_list
            if call.args and any(marker in call.args[0] for marker in (
                'Model fitting started', 'Fitting 1/1', 'Finished 1/1',
                'Model fitting finished', 'Model fitting failed'))
        ]
        self.assertTrue(lifecycle_calls)
        self.assertTrue(all(call.args[2] is False for call in lifecycle_calls))
        result_widget.close()

    def test_pivoted_cognitive_fit_can_be_cancelled_cooperatively(self):
        """Cancelling a live worker stops its optimizer path without reporting failure."""
        from app.lib.pivotedDataWidget import PivotedDataWidget
        from app.psyDataFunc import PsyDataFunc

        dataframe = pd.DataFrame({
            'rt': [.35, .40, .45, .50],
            'choice': [6.0, 5.0, 6.0, 5.0],
            'correct': [1, 1, 0, 0],
        })
        specification = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', [6.0, 5.0],
            accuracy_variable='correct')
        target = {'model_specification': specification}
        worker_entered = threading.Event()

        def wait_for_cancel(*_args, cancel_check=None, **_kwargs):
            worker_entered.set()
            while True:
                raise_if_fit_cancelled(cancel_check)
                time.sleep(.001)

        with patch.object(PsyDataFunc, 'genScript'), \
                patch.object(PsyDataFunc, 'printOut'), \
                patch(
                    'app.lib.fitCognitiveModelThread.fit_cognitive_model',
                    side_effect=wait_for_cancel):
            result_widget = PivotedDataWidget(
                dataframe, [], [], [target], [])
            cancelled = QSignalSpy(result_widget.analysisCancelled)
            failed = QSignalSpy(result_widget.analysisFailed)
            deadline = time.monotonic() + 2.0
            while not worker_entered.is_set() and time.monotonic() < deadline:
                QTest.qWait(10)
            self.assertTrue(worker_entered.is_set())
            result_widget.cancelAnalysis()
            self.assertTrue(len(cancelled) or cancelled.wait(2000))

        self.assertEqual(len(failed), 0)
        self.assertIsNone(result_widget.fit_dist_thread)
        result_widget.close()

    def test_pivoted_rt_distribution_fit_runs_off_the_gui_thread(self):
        """RT distribution fitting uses the same non-blocking background queue."""
        from app.lib.pivotedDataWidget import PivotedDataWidget
        from app.psyDataFunc import PsyDataFunc
        from app.rtDist import fit_rt_distribution

        dataframe = pd.DataFrame({
            'rt': [.31, .35, .39, .44, .50, .57, .65, .74, .85, .98],
        })
        main_thread_id = int(QThread.currentThreadId())
        fitting_thread_ids = []

        def record_thread(*args, **kwargs):
            fitting_thread_ids.append(int(QThread.currentThreadId()))
            return fit_rt_distribution(*args, **kwargs)

        with patch.object(PsyDataFunc, 'genScript'), \
                patch.object(PsyDataFunc, 'printOut'), \
                patch(
                    'app.lib.fitRTsDistThread.fit_rt_distribution',
                    side_effect=record_thread):
            result_widget = PivotedDataWidget(
                dataframe, [], [], ['rt@Gamma (k, θ)'], [])
            finished = QSignalSpy(result_widget.analysisFinished)
            failed = QSignalSpy(result_widget.analysisFailed)
            self.assertIsNone(result_widget.table)
            self.assertTrue(finished.wait(5000))

        self.assertEqual(len(failed), 0)
        self.assertIsNotNone(result_widget.table)
        self.assertTrue(fitting_thread_ids)
        self.assertTrue(all(thread_id != main_thread_id for thread_id in fitting_thread_ids))
        result_widget.close()

    def test_run_is_rejected_while_model_fit_is_active(self):
        """A second Run click explains the active background fit instead of re-entering it."""
        from PsySummary import PsyData

        window = PsyData()
        window.model_fit_running = True
        with patch('PsySummary.MessageBox.exec_') as execute:
            window.runSummary()

        execute.assert_called_once()
        dialog = window._model_fit_message_box
        self.assertEqual(len(dialog.text().splitlines()), 2)
        self.assertIn('background', dialog.text())
        self.assertGreaterEqual(dialog.minimumWidth(), 540)
        self.assertTrue(window.model_fit_running)
        window.model_fit_running = False
        window.close()

    def test_close_can_keep_an_active_fit_running(self):
        """Rejecting the close confirmation leaves the fit and window active."""
        from PsySummary import PsyData

        window = PsyData()
        window.model_fit_running = True
        with patch.object(window, '_confirmStopModelFit', return_value=False):
            window.close()

        self.assertTrue(window.model_fit_running)
        self.assertFalse(window._closing_after_model_cancel)
        window.model_fit_running = False
        window.close()

    def test_confirmed_close_cancels_pending_fit_then_closes(self):
        """Confirming close requests cancellation and discards the pending result."""
        from PsySummary import PsyData

        class PendingResult(QWidget):
            analysisCancelled = pyqtSignal()

            def __init__(self):
                super().__init__()
                self.cancel_requested = False

            def cancelAnalysis(self):
                self.cancel_requested = True
                self.analysisCancelled.emit()

        window = PsyData()
        pending_result = PendingResult()
        pending_result.analysisCancelled.connect(
            lambda: window._modelFitCancelled(pending_result))
        window._pending_result_widget = pending_result
        window.model_fit_running = True

        with patch.object(window, '_confirmStopModelFit', return_value=True):
            window.close()
        self.app.processEvents()

        self.assertTrue(pending_result.cancel_requested)
        self.assertFalse(window.model_fit_running)
        self.assertIsNone(window._pending_result_widget)
        self.assertTrue(window.model_fit_overlay.isHidden())
        self.assertFalse(window.model_fit_overlay.animation_timer.isActive())
        self.assertFalse(window._closing_after_model_cancel)

    def test_model_results_are_revealed_only_after_background_fitting(self):
        """A first model run keeps Results hidden until the pending widget succeeds."""
        from PsySummary import PsyData

        class PendingResult(QWidget):
            analysisFinished = pyqtSignal()
            analysisFailed = pyqtSignal(str)
            analysisProgress = pyqtSignal(int, int, str)

        window = PsyData()
        window.data = pd.DataFrame({'rt': [.3, .4, .5, .6]})
        window.data_list.addItem('rt@Gamma (k, θ)')
        pending_result = PendingResult()

        with patch(
                'PsySummary.PivotedDataWidget',
                return_value=pending_result):
            window.runSummary()

        self.assertTrue(window.model_fit_running)
        self.assertIs(window._pending_result_widget, pending_result)
        self.assertIsNone(window.pivotTableWindow)
        self.assertTrue(window.results_dock.isHidden())
        self.assertTrue(window.results_toggle_button.isHidden())
        self.assertFalse(window.model_fit_overlay.isHidden())
        self.assertTrue(window.model_fit_overlay.animation_timer.isActive())
        self.assertEqual(window.model_fit_overlay.animation_timer.interval(), FRAME_INTERVAL_MS)
        self.assertFalse(window.menuBar().isEnabled())
        self.assertNotIn('Fitting model', window.output.text_edit.toPlainText())
        self.assertEqual(
            window.model_fit_overlay._message, 'Preparing model fitting…')

        pending_result.analysisProgress.emit(
            2, 3, 'target.rt @ Ratcliff Diffusion Model')
        self.app.processEvents()
        self.assertEqual(
            window.model_fit_overlay._message, 'Fitting model 2 of 3…')
        self.assertEqual(
            window.model_fit_overlay._detail,
            'target.rt @ Ratcliff Diffusion Model')

        pending_result.analysisFinished.emit()
        self.app.processEvents()
        self.assertFalse(window.model_fit_running)
        self.assertTrue(window.model_fit_overlay.isHidden())
        self.assertFalse(window.model_fit_overlay.animation_timer.isActive())
        self.assertTrue(window.menuBar().isEnabled())
        self.assertIsNone(window._pending_result_widget)
        self.assertIs(window.results_dock.widget(), pending_result)
        self.assertFalse(window.results_dock.isHidden())
        window.close()

    def test_model_fit_overlay_covers_and_blocks_the_main_content(self):
        """The spinner overlay stays above the full client area until explicitly stopped."""
        from PsySummary import PsyData

        window = PsyData()
        window.show()
        self.app.processEvents()
        window._startModelFitOverlay()
        self.app.processEvents()

        button_center = window.run_button.mapTo(
            window, window.run_button.rect().center())
        self.assertEqual(window.model_fit_overlay.geometry(), window.rect())
        self.assertIs(window.childAt(button_center), window.model_fit_overlay)
        self.assertFalse(window.menuBar().isEnabled())
        self.assertEqual(
            window.model_fit_overlay.OVERLAY_COLOR, (245, 245, 245, 64))
        self.assertEqual(
            window.model_fit_overlay.SPINNER_COLOR, (70, 70, 70))
        self.assertEqual(
            window.model_fit_overlay.MESSAGE_COLOR, (25, 25, 25))
        self.assertEqual(
            window.model_fit_overlay.DETAIL_COLOR, (47, 111, 176))

        window._stopModelFitOverlay()
        self.app.processEvents()
        self.assertTrue(window.model_fit_overlay.isHidden())
        self.assertTrue(window.menuBar().isEnabled())
        window.close()

    def test_existing_results_remain_visible_until_refit_succeeds(self):
        """A refit replaces an existing result atomically instead of showing an empty dock."""
        from PsySummary import PsyData

        class PendingResult(QWidget):
            analysisFinished = pyqtSignal()
            analysisFailed = pyqtSignal(str)

        window = PsyData()
        old_result = QWidget()
        window._showAggregationResults(old_result)
        window.data = pd.DataFrame({'rt': [.3, .4, .5, .6]})
        window.data_list.addItem('rt@Gamma (k, θ)')
        pending_result = PendingResult()

        with patch(
                'PsySummary.PivotedDataWidget',
                return_value=pending_result):
            window.runSummary()

        self.assertIs(window.results_dock.widget(), old_result)
        pending_result.analysisFinished.emit()
        self.app.processEvents()
        self.assertIs(window.results_dock.widget(), pending_result)
        self.assertIsNone(old_result.parent())
        window.close()

    def test_failed_refit_preserves_existing_results(self):
        """A failed pending model must not replace the last successful result."""
        from PsySummary import PsyData

        class PendingResult(QWidget):
            analysisFinished = pyqtSignal()
            analysisFailed = pyqtSignal(str)

        window = PsyData()
        old_result = QWidget()
        window._showAggregationResults(old_result)
        window.data = pd.DataFrame({'rt': [.3, .4, .5, .6]})
        window.data_list.addItem('rt@Gamma (k, θ)')
        pending_result = PendingResult()

        with patch(
                'PsySummary.PivotedDataWidget',
                return_value=pending_result):
            window.runSummary()

        with patch('PsySummary.MessageBox.warning') as warning:
            pending_result.analysisFailed.emit('fit failed')
            self.app.processEvents()

        self.assertFalse(window.model_fit_running)
        self.assertTrue(window.model_fit_overlay.isHidden())
        self.assertFalse(window.model_fit_overlay.animation_timer.isActive())
        self.assertTrue(window.menuBar().isEnabled())
        self.assertIsNone(window._pending_result_widget)
        self.assertIs(window.results_dock.widget(), old_result)
        warning.assert_called_once_with(window, 'Model Fitting Error', 'fit failed')
        window.close()

    def test_aggregation_results_use_fixed_right_side_dock(self):
        """Aggregation results stay in a closable, non-floating dock owned by Data Summary."""
        from PsySummary import PsyData

        window = PsyData()
        window.show()
        self.app.processEvents()
        collapsed_width = window.width()
        collapsed_central_width = window.central_widget.width()
        self.assertTrue(window.results_dock.isHidden())
        self.assertFalse(window.results_action.isEnabled())
        self.assertEqual(window.results_toggle_button.text(), '》')
        self.assertFalse(window.results_toggle_button.isEnabled())
        self.assertTrue(window.results_toggle_button.isHidden())
        self.assertEqual(
            window.results_dock.allowedAreas(), Qt.RightDockWidgetArea)
        self.assertEqual(
            window.results_dock.features(), QDockWidget.DockWidgetClosable)

        first_result = QWidget()
        window._showAggregationResults(first_result)
        self.assertIs(window.results_dock.widget(), first_result)
        self.assertIs(window.pivotTableWindow, first_result)
        self.assertFalse(window.results_dock.isHidden())
        self.assertTrue(window.results_action.isEnabled())
        self.assertFalse(window.results_dock.isFloating())
        self.assertEqual(window.results_toggle_button.text(), '《')
        self.assertFalse(window.results_toggle_button.isHidden())
        self.assertIs(
            window.results_toggle_button.parentWidget(), window.variables_header)
        self.assertEqual(window.results_toggle_button.height(), 24)
        toggle_position = window.results_toggle_button.pos()

        window.results_toggle_button.click()
        self.app.processEvents()
        self.assertTrue(window.results_dock.isHidden())
        self.assertEqual(window.results_toggle_button.text(), '》')
        self.assertIs(
            window.results_toggle_button.parentWidget(), window.variables_header)
        self.assertEqual(window.results_toggle_button.y(), toggle_position.y())
        self.assertEqual(window.width(), collapsed_width)
        self.assertEqual(window.central_widget.width(), collapsed_central_width)

        window.results_toggle_button.click()
        self.app.processEvents()
        self.assertFalse(window.results_dock.isHidden())
        self.assertEqual(window.results_toggle_button.text(), '《')
        self.assertIs(
            window.results_toggle_button.parentWidget(), window.variables_header)
        self.assertEqual(window.results_toggle_button.y(), toggle_position.y())

        window.results_dock.close()
        QTest.qWait(1)
        self.assertTrue(window.results_dock.isHidden())
        self.assertEqual(window.width(), collapsed_width)
        self.assertFalse(window.results_action.isChecked())

        window.results_action.trigger()
        self.app.processEvents()
        self.assertFalse(window.results_dock.isHidden())
        self.assertEqual(window.results_toggle_button.text(), '《')

        second_result = QWidget()
        window._showAggregationResults(second_result)
        self.assertIs(window.results_dock.widget(), second_result)
        self.assertIsNone(first_result.parent())
        window.close()

    def test_filtered_race_retains_parameters_for_surviving_responses(self):
        """Accumulator parameters follow their observed response when filters remove a level."""
        frame = pd.DataFrame({'rt': [.4, .5, .6], 'choice': ['a', 'b', 'c']})
        spec = make_model_specification(LBA_MODEL, 'rt', 'choice', ['a', 'b', 'c'])
        next(row for row in spec['parameters'] if row['name'] == 'mean_v[3]')['value'] = 4.2
        dialog = CognitiveModelDialog(frame.iloc[1:], LBA_MODEL, 'rt', spec)
        saved = dialog.specification()
        self.assertEqual(saved['response_mapping'], {'b': 1, 'c': 2})
        self.assertEqual(next(row for row in saved['parameters'] if row['name'] == 'mean_v[2]')['value'], 4.2)
        with self.assertRaisesRegex(ValueError, 'must be Fixed'):
            validate_model_data(saved, frame.iloc[1:])
        next(row for row in saved['parameters'] if row['name'] == 'sd_v[1]')['mode'] = 'fixed'
        validate_model_data(saved, frame.iloc[1:])
        dialog.close()

    def test_dialog_collects_structured_variable_and_parameter_settings(self):
        dataframe = pd.DataFrame({
            'rt': [300, 420, 510, 610],
            'choice': ['left', 'right', 'left', 'right'],
            'correct': [1, 1, 0, 1],
        })
        dialog = CognitiveModelDialog(dataframe, RATCLIFF_MODEL, 'rt')
        self.assertEqual(dialog.boundary_coding_combo.currentData(), ACCURACY_CODING)
        self.assertEqual(dialog.response_combo.currentText(), dialog.NONE_LABEL)
        self.assertEqual(dialog.parameter_table.rowCount(), 0)
        self.assertEqual(dialog.seed_spin.value(), 0)
        self.assertEqual(dialog.seed_spin.text(), 'Time-based')
        dialog.response_combo.setCurrentText('choice')
        dialog.accuracy_combo.setCurrentText('correct')
        specification = dialog.specification()
        self.assertEqual(dialog.mapping_group.title(), 'Boundary Mapping')
        self.assertFalse(dialog.accuracy_mapping_table.isHidden())
        self.assertEqual(dialog.accuracy_mapping_table.item(0, 0).text(), '0 (Error)')
        self.assertEqual(dialog.accuracy_mapping_table.item(0, 1).text(), 'lower')
        self.assertEqual(dialog.accuracy_mapping_table.item(1, 0).text(), '1 (Correct)')
        self.assertEqual(dialog.accuracy_mapping_table.item(1, 1).text(), 'upper')
        self.assertEqual(dialog.mapping_table.horizontalHeaderItem(0).text(), 'Observed Response')
        self.assertEqual(dialog.mapping_table.horizontalHeaderItem(1).text(), 'Physical Boundary')
        mapping_row_height = dialog.mapping_table.verticalHeader().defaultSectionSize()
        parameter_row_height = dialog.parameter_table.verticalHeader().defaultSectionSize()
        self.assertLess(dialog.mapping_table.height(), mapping_row_height * 3.5)
        self.assertGreaterEqual(dialog.parameter_table.height(), parameter_row_height * 9)
        dialog.close()
        self.assertEqual(specification['rt_unit'], 'milliseconds')
        self.assertEqual(specification['response_variable'], 'choice')
        self.assertEqual(specification['accuracy_variable'], 'correct')
        self.assertEqual(specification['response_mapping'], {'left': 'lower', 'right': 'upper'})
        self.assertIsNone(specification['optimizer']['seed'])
        self.assertTrue(all({'name', 'mode', 'value', 'lower', 'upper'} <= set(parameter)
                            for parameter in specification['parameters']))
        for row in range(dialog.parameter_table.rowCount()):
            self.assertTrue(dialog.parameter_table.item(row, 0).toolTip())
            for column in range(1, 5):
                self.assertTrue(dialog.parameter_table.cellWidget(row, column).toolTip())

    def test_response_coding_hides_fixed_accuracy_boundary_mapping(self):
        """Response Coding presents physical responses as the fitted model boundaries."""
        dataframe = pd.DataFrame({
            'rt': [.35, .45, .55, .65],
            'choice': ['left', 'right', 'left', 'right'],
            'correct': [1, 1, 0, 0],
        })
        dialog = CognitiveModelDialog(dataframe, RATCLIFF_MODEL, 'rt')
        dialog.response_combo.setCurrentText('choice')
        dialog.accuracy_combo.setCurrentText('correct')
        dialog.boundary_coding_combo.setCurrentIndex(
            dialog.boundary_coding_combo.findData(RESPONSE_CODING))

        self.assertEqual(dialog.mapping_group.title(), 'Boundary Mapping')
        self.assertTrue(dialog.accuracy_mapping_table.isHidden())
        self.assertTrue(dialog.physical_mapping_label.isHidden())
        self.assertEqual(dialog.mapping_table.horizontalHeaderItem(0).text(), 'Observed Response')
        self.assertEqual(dialog.mapping_table.horizontalHeaderItem(1).text(), 'Model Boundary')
        self.assertEqual(
            dialog.specification()['response_mapping'],
            {'left': 'lower', 'right': 'upper'},
        )
        dialog.close()

    def test_data_item_model_settings_round_trip_independently_of_display_text(self):
        specification = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'], minimum_rt=0.3)
        widget = DraggableListWidget(DraggableListWidget.DataType)
        display_text = f'rt@{RATCLIFF_MODEL}'
        widget.addItem(display_text)
        widget.contentList.append(display_text)
        widget.item(0).setData(MODEL_SPEC_ROLE, specification)
        saved = widget.modelSpecifications()

        widget.item(0).setData(MODEL_SPEC_ROLE, None)
        widget.restoreModelSpecifications(saved)
        self.assertEqual(widget.item(0).text(), display_text)
        self.assertEqual(widget.item(0).data(MODEL_SPEC_ROLE), specification)
        widget.close()

    def test_legacy_ratcliff_specification_defaults_to_response_coding(self):
        """Old saved dictionaries without the new property retain response-coded behavior."""
        dataframe = pd.DataFrame({
            'rt': [.35, .45, .55, .65],
            'choice': ['left', 'right', 'left', 'right'],
            'correct': [1, 1, 0, 0],
        })
        specification = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'],
            accuracy_variable='correct')
        specification.pop('boundary_coding')
        dialog = CognitiveModelDialog(dataframe, RATCLIFF_MODEL, 'rt', specification)
        self.assertEqual(dialog.boundary_coding_combo.currentData(), RESPONSE_CODING)
        self.assertEqual(dialog.specification()['boundary_coding'], RESPONSE_CODING)
        dialog.close()

    def test_data_item_context_menu_keeps_model_settings_visible_but_conditional(self):
        widget = DraggableListWidget(DraggableListWidget.DataType)
        widget.addItem('rt@Mean')
        plain_menu = widget._create_data_context_menu(widget.item(0))
        plain_actions = {action.text(): action for action in plain_menu.actions()}
        self.assertTrue(plain_actions['Change Analysis...'].isEnabled())
        self.assertFalse(plain_actions['Model Settings...'].isEnabled())

        widget.item(0).setText(f'rt@{LBA_MODEL}')
        model_menu = widget._create_data_context_menu(widget.item(0))
        model_actions = {action.text(): action for action in model_menu.actions()}
        self.assertTrue(model_actions['Change Analysis...'].isEnabled())
        self.assertTrue(model_actions['Model Settings...'].isEnabled())
        widget.close()

    def test_plain_data_item_single_click_opens_selector_without_delay(self):
        widget = DraggableListWidget(DraggableListWidget.DataType)
        widget.addItem('rt@Mean')
        widget._last_mouse_button = Qt.LeftButton
        with patch.object(widget, '_show_operation_selector') as show_selector:
            widget.itemSingleClick(widget.item(0))

        show_selector.assert_called_once_with(widget.item(0))
        self.assertFalse(widget._single_click_timer.isActive())
        widget.close()

    def test_cognitive_data_item_single_click_waits_for_double_click(self):
        widget = DraggableListWidget(DraggableListWidget.DataType)
        widget.addItem(f'rt@{RATCLIFF_MODEL}')
        widget._last_mouse_button = Qt.LeftButton
        with patch.object(widget, '_show_operation_selector') as show_selector:
            widget.itemSingleClick(widget.item(0))

        show_selector.assert_not_called()
        self.assertTrue(widget._single_click_timer.isActive())
        self.assertIs(widget._pending_single_click_item, widget.item(0))
        widget.close()

    def test_plain_data_item_double_click_opens_operation_selector(self):
        widget = DraggableListWidget(DraggableListWidget.DataType)
        widget.addItem('rt@Mean')
        with patch.object(widget, '_show_operation_selector') as show_selector, \
                patch.object(widget, '_configure_model_item') as configure_model:
            widget.itemDoubleClick(widget.item(0))

        show_selector.assert_called_once_with(widget.item(0))
        configure_model.assert_not_called()
        widget.close()

    def test_operation_selector_can_switch_items_without_double_deleting_combo(self):
        """Replacing an item widget must not delete its combo box twice."""
        widget = DraggableListWidget(DraggableListWidget.DataType)
        widget.addItems(['rt@Mean', 'rt@Median'])

        widget._show_operation_selector(widget.item(0))
        first_combo = widget.combo_box
        widget._show_operation_selector(widget.item(1))
        self.app.processEvents()

        self.assertIsNot(widget.combo_box, first_combo)
        self.assertIs(widget.itemLabel, widget.item(1))
        self.assertIsNotNone(widget.combo_box)
        self.assertFalse(sip.isdeleted(widget.combo_box))

        stale_combo = widget.combo_box
        stale_combo.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.assertTrue(sip.isdeleted(stale_combo))
        widget._show_operation_selector(widget.item(1))
        self.assertIsNot(widget.combo_box, stale_combo)
        self.assertFalse(sip.isdeleted(widget.combo_box))
        widget.close()

    def test_operation_selector_cleanup_survives_clear_and_active_item_removal(self):
        """Clear and Delete release selector references before Qt destroys child widgets."""
        widget = DraggableListWidget(DraggableListWidget.DataType)
        widget.contentList = ['rt@Mean']
        widget.addItem('rt@Mean')
        widget._show_operation_selector(widget.item(0))
        widget.clear()
        self.app.processEvents()
        self.assertIsNone(widget.itemLabel)
        self.assertIsNone(widget.combo_box)

        widget.contentList = ['rt@Median']
        widget.addItem('rt@Median')
        active_item = widget.item(0)
        widget._show_operation_selector(active_item)
        widget.removeItem(active_item)
        self.app.processEvents()
        self.assertIsNone(widget.itemLabel)
        self.assertIsNone(widget.combo_box)
        widget.close()

    def test_cognitive_data_item_double_click_opens_model_settings(self):
        widget = DraggableListWidget(DraggableListWidget.DataType)
        widget.addItem(f'rt@{LBA_MODEL}')
        with patch.object(widget, '_show_operation_selector') as show_selector, \
                patch.object(widget, '_configure_model_item') as configure_model:
            widget.itemDoubleClick(widget.item(0))

        show_selector.assert_not_called()
        configure_model.assert_called_once_with(widget.item(0), LBA_MODEL)
        widget.close()

    def test_model_dialog_receives_current_filtered_rows(self):
        dataframe = pd.DataFrame({
            'rt': [0.3, 0.4, 0.5],
            'choice': ['left', 'right', 'unused'],
        })
        filtered = dataframe.iloc[:2].copy()
        specification = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'], minimum_rt=0.3)
        widget = DraggableListWidget(DraggableListWidget.DataType)
        display_text = f'rt@{RATCLIFF_MODEL}'
        widget.addItem(display_text)
        widget.setModelContext(dataframe, lambda: filtered)
        received = {}

        def fake_editor(model_dataframe, model, rt_variable, current_specification, parent):
            received['dataframe'] = model_dataframe
            received['model'] = model
            received['rt_variable'] = rt_variable
            return specification

        with patch(
                'app.lib.draggablelistwidget.edit_cognitive_model',
                side_effect=fake_editor):
            self.assertTrue(widget._configure_model_item(widget.item(0), RATCLIFF_MODEL))

        pd.testing.assert_frame_equal(received['dataframe'], filtered)
        self.assertEqual(received['model'], RATCLIFF_MODEL)
        self.assertEqual(received['rt_variable'], 'rt')
        widget.close()

    def test_switching_models_does_not_reuse_previous_model_parameters(self):
        dataframe = pd.DataFrame({
            'rt': [0.3, 0.4, 0.5, 0.6],
            'choice': ['left', 'right', 'left', 'right'],
        })
        previous_specification = make_model_specification(
            RATCLIFF_MODEL, 'rt', 'choice', ['left', 'right'], minimum_rt=0.3)
        widget = DraggableListWidget(DraggableListWidget.DataType)
        widget.addItem(f'rt@{RATCLIFF_MODEL}')
        widget.item(0).setData(MODEL_SPEC_ROLE, previous_specification)
        widget.setModelContext(dataframe)
        received = {}

        def fake_editor(model_dataframe, model, rt_variable, specification, parent):
            received['model'] = model
            received['specification'] = specification
            return make_model_specification(
                LBA_MODEL, rt_variable, 'choice', ['left', 'right'], minimum_rt=0.3)

        with patch(
                'app.lib.draggablelistwidget.edit_cognitive_model',
                side_effect=fake_editor):
            self.assertTrue(widget._configure_model_item(widget.item(0), LBA_MODEL))

        self.assertEqual(received['model'], LBA_MODEL)
        self.assertIsNone(received['specification'])
        widget.close()

        dialog = CognitiveModelDialog(
            dataframe, LBA_MODEL, 'rt', previous_specification)
        self.assertEqual(dialog.response_combo.currentText(), dialog.NONE_LABEL)
        self.assertEqual(dialog.parameter_table.rowCount(), 0)
        dialog.response_combo.setCurrentText('choice')
        parameter_names = [
            dialog.parameter_table.item(row, 0).text()
            for row in range(dialog.parameter_table.rowCount())
        ]
        self.assertEqual(parameter_names[:4], ['A', 'b', 't0', 'st0'])
        self.assertIn('mean_v[1]', parameter_names)
        self.assertNotIn('z', parameter_names)
        dialog.close()

    def test_reopened_dialog_does_not_overlay_stale_table_editors(self):
        dataframe = pd.DataFrame({
            'rt': [0.3, 0.4, 0.5, 0.6],
            'choice': [6.0, 5.0, 6.0, 5.0],
            'correct': [1, 1, 0, 1],
        })
        first_dialog = CognitiveModelDialog(dataframe, RATCLIFF_MODEL, 'rt')
        first_dialog.response_combo.setCurrentText('choice')
        first_dialog.accuracy_combo.setCurrentText('correct')
        specification = first_dialog.specification()
        first_dialog.close()

        reopened = CognitiveModelDialog(
            dataframe, RATCLIFF_MODEL, 'rt', specification)
        reopened.show()
        self.app.processEvents()
        visible_mapping_combos = [
            widget for widget in reopened.mapping_table.findChildren(QComboBox)
            if widget.isVisibleTo(reopened.mapping_table)
        ]
        visible_parameter_combos = [
            widget for widget in reopened.parameter_table.findChildren(QComboBox)
            if widget.isVisibleTo(reopened.parameter_table)
        ]
        visible_parameter_edits = [
            widget for widget in reopened.parameter_table.findChildren(QLineEdit)
            if widget.isVisibleTo(reopened.parameter_table)
        ]
        self.assertEqual(reopened.mapping_table.item(0, 0).text(), '6.0')
        self.assertEqual(reopened.parameter_table.item(0, 0).text(), 'a')
        self.assertEqual(len(visible_mapping_combos), 2)
        self.assertEqual(len(visible_parameter_combos), 9)
        self.assertEqual(len(visible_parameter_edits), 27)
        reopened.close()

    def test_reopened_dialog_removes_response_levels_excluded_by_filters(self):
        complete_data = pd.DataFrame({
            'rt': [0.3, 0.4, 0.5],
            'choice': ['left', 'right', 'unused'],
        })
        specification = make_model_specification(
            LBA_MODEL, 'rt', 'choice', ['left', 'right', 'unused'], minimum_rt=0.3)
        filtered_data = complete_data.iloc[:2].copy()
        dialog = CognitiveModelDialog(
            filtered_data, LBA_MODEL, 'rt', specification)
        self.assertEqual(
            [dialog.mapping_table.item(row, 0).text() for row in range(dialog.mapping_table.rowCount())],
            ['left', 'right'],
        )
        self.assertEqual(dialog.parameter_table.rowCount(), 8)
        dialog.close()


if __name__ == '__main__':
    unittest.main()
