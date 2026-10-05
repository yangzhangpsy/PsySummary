"""Qt-free numerical preparation for PsySummary diagnostic plots."""

import html

import numpy as np
from scipy.integrate import cumulative_trapezoid

from app.rtDist import rt_distribution_cdf, rt_distribution_pdf
from app.cognitiveModelSpec import COGNITIVE_MODEL_NAMES, RATCLIFF_MODEL, RESPONSE_CODING
from app.cognitiveModels import fitted_accuracy_pdf, fitted_response_pdf


_CURVE_LABEL = '\ufff0curve\ufff1'
_RECORD_LABEL = '\ufff0record\ufff1'


class _NumericAxes:
    """Collect numeric plotting commands without importing or calling Matplotlib objects."""

    def __init__(self, commands, axis, cancel_check):
        self.commands = commands
        self.axis = axis
        self.cancel_check = cancel_check

    def _append(self, method, args, options):
        if self.cancel_check():
            raise InterruptedError('Curve preparation cancelled.')
        self.commands.append((self.axis, method, args, options))

    def plot(self, *args, **options):
        self._append('plot', args, options)

    def step(self, *args, **options):
        self._append('step', args, options)

    def hist(self, values, bins, density, histtype, **options):
        # Match Matplotlib's stepfilled histogram, but calculate bins off the GUI thread.
        counts, edges = np.histogram(values, bins=bins, density=density)
        self._append('stairs', (counts, edges), dict(options, fill=True))


class _CurveBuilder:
    """Prepare color/selection-independent curves using the existing diagnostic formulas."""

    def __init__(self, accuracy_mode):
        self.accuracy_mode = accuracy_mode

    def prepare(self, record, cancel_check):
        self.cancel_check = cancel_check
        self.failed = False
        self._check_cancelled()
        commands = []
        density_axis = _NumericAxes(commands, 0, cancel_check)
        cdf_axis = _NumericAxes(commands, 1, cancel_check)
        if record.get('model') in COGNITIVE_MODEL_NAMES:
            count = 2 if self.accuracy_mode else max(1, len(record['specification']['response_values']))
            status = self._draw_cognitive_record(
                record, density_axis, cdf_axis, list(range(count)), _CURVE_LABEL, _RECORD_LABEL)
            return {'commands': commands, 'status': status, 'failed': self.failed}
        data = np.asarray(record['data'], dtype=float)
        parameters = np.asarray(record['parameters'], dtype=float)
        convergence = 'Converged' if record['converged'] else 'Not converged'
        status = (
            f'<b>{_RECORD_LABEL}</b>: {convergence}; '
            f"N={record['n_valid']}; boundary={record['parameter_boundary']} "
            f"({html.escape(record['boundary_details'])}); shift warning={record['shift_warning']} "
            f"({html.escape(record['shift_warning_details'])}); LL={self._number(record['log_likelihood'])}; "
            f"AIC={self._number(record['aic'])}; BIC={self._number(record['bic'])}.")
        if data.size:
            bins = min(50, max(8, int(np.sqrt(data.size))))
            density_axis.hist(data, bins=bins, density=True, histtype='stepfilled',
                              alpha=0.12, color=0, edgecolor=0, label=f'{_CURVE_LABEL} — observed')
            sorted_data = np.sort(data)
            empirical = np.arange(1, data.size + 1) / data.size
            cdf_axis.step(sorted_data, empirical, where='post', color=0, linestyle='--',
                          linewidth=1.8, label=f'{_CURVE_LABEL} — empirical CDF')
            if np.all(np.isfinite(parameters)):
                spread = max(float(np.ptp(data)), abs(float(np.mean(data))) * 0.05, 1e-6)
                x_values = np.linspace(float(np.min(data)) - 0.05 * spread,
                                       float(np.max(data)) + 0.10 * spread, 500)
                try:
                    pdf = rt_distribution_pdf(record['distribution'], x_values, parameters)
                    if cancel_check():
                        raise InterruptedError('Curve preparation cancelled.')
                    cdf = rt_distribution_cdf(record['distribution'], x_values, parameters)
                    valid_pdf = np.isfinite(pdf) & (pdf >= 0)
                    valid_cdf = np.isfinite(cdf)
                    density_axis.plot(x_values[valid_pdf], pdf[valid_pdf], color=0,
                                      linewidth=2, label=f'{_CURVE_LABEL} — Fitted model')
                    cdf_axis.plot(x_values[valid_cdf], cdf[valid_cdf], color=0,
                                  linewidth=2, label=f'{_CURVE_LABEL} — Fitted model')
                except InterruptedError:
                    raise
                except Exception as error:
                    self.failed = True
                    status += f' Curve unavailable: {html.escape(str(error))}.'
        return {'commands': commands, 'status': status, 'failed': self.failed}

    def _check_cancelled(self):
        """Stop numerical preparation at a bounded chunk boundary."""
        if self.cancel_check():
            raise InterruptedError('Curve preparation cancelled.')

    def _density(self, function, record, values, category):
        """Preserve every grid point while allowing cancellation between PDF chunks."""
        chunks = []
        for start in range(0, len(values), 32):
            self._check_cancelled()
            chunks.append(function(record, values[start:start + 32], category))
        self._check_cancelled()
        return np.concatenate(chunks)

    def _draw_cognitive_record(self, record, density_axis, cdf_axis, curve_colors,
                               curve_label, label):
        """Draw response-conditioned diagnostics for one cognitive RT model fit."""
        data = np.asarray(record['rt'], dtype=float)
        responses = np.asarray(record['response'], dtype=int)
        status = 'Converged' if record['converged'] else 'Not converged'
        status_line = (
            f'<b>{html.escape(label)}</b>: {status}; N={record["n_valid"]}; '
            f'coding={html.escape(record.get("boundary_coding", RESPONSE_CODING))}; '
            f'responses={html.escape(str(record["response_counts"]))}; '
            f'LL={self._number(record["log_likelihood"])}; '
            f'AIC={self._number(record["aic"])}; BIC={self._number(record["bic"])}.'
        )
        if record.get('accuracy_rate') is not None:
            status_line += f' Accuracy={self._number(record["accuracy_rate"])}.'
        if data.size == 0:
            return status_line
        if self.accuracy_mode:
            return self._draw_cognitive_accuracy_record(
                record, density_axis, cdf_axis, curve_colors, curve_label, status_line)
        specification = record['specification']
        response_values = specification['response_values']
        response_mapping = specification['response_mapping']
        mapped_labels = []
        for response_index in range(1, len(response_values) + 1):
            target = ('lower' if response_index == 1 else 'upper') \
                if record['model'] == RATCLIFF_MODEL else response_index
            mapped_labels.append(next(
                (value for value in response_values if response_mapping.get(str(value)) == target), target))
        spread = max(float(np.ptp(data)), abs(float(np.mean(data))) * 0.05, 1e-6)
        x_values = np.linspace(0.0, float(np.max(data)) + 0.10 * spread, 400)
        for response_index, observed_value in enumerate(mapped_labels, start=1):
            mask = responses == response_index
            response_data = np.sort(data[mask])
            if response_data.size == 0:
                continue
            color = curve_colors[response_index - 1]
            response_label = f'{curve_label} / {observed_value}'
            edges = np.histogram_bin_edges(response_data, bins=min(40, max(6, int(np.sqrt(response_data.size)))))
            counts, edges = np.histogram(response_data, bins=edges)
            widths = np.diff(edges)
            centers = edges[:-1] + widths / 2.0
            density_axis.step(
                centers, counts / (data.size * widths), where='mid', color=color,
                linestyle='--', linewidth=1.4, label=f'{response_label} — observed')
            empirical = np.arange(1, response_data.size + 1) / data.size
            cdf_axis.step(
                response_data, empirical, where='post', color=color, linestyle='--',
                linewidth=1.4, label=f'{response_label} — empirical')
            try:
                fitted_density = self._density(fitted_response_pdf, record, x_values, response_index)
                density_axis.plot(
                    x_values, fitted_density, color=color, linewidth=2,
                    label=f'{response_label} — Fitted model')
                fitted_cdf = cumulative_trapezoid(fitted_density, x_values, initial=0.0)
                cdf_axis.plot(
                    x_values, fitted_cdf, color=color, linewidth=2,
                    label=f'{response_label} — Fitted model')
            except InterruptedError:
                raise
            except Exception as error:
                self.failed = True
                status_line += f' Curve unavailable: {html.escape(str(error))}.'
        return status_line


    def _draw_cognitive_accuracy_record(self, record, density_axis, cdf_axis, curve_colors,
                                        curve_label, status_line):
        """Draw observed and model-implied correct/error RT distributions."""
        accuracy = record.get('accuracy')
        if accuracy is None:
            return status_line + ' Correct/Error diagnostics require an Accuracy Variable.'
        data = np.asarray(record['rt'], dtype=float)
        responses = np.asarray(record['response'], dtype=int)
        accuracy = np.asarray(accuracy, dtype=float)
        valid = np.isfinite(accuracy) & np.isin(accuracy, (0.0, 1.0))
        if not np.any(valid):
            return status_line + ' No valid Accuracy/Correct values are available for diagnostics.'
        valid_count = int(np.sum(valid))
        category_colors = curve_colors[:2]
        for category, category_label, color in zip((1.0, 0.0), ('Correct', 'Error'), category_colors):
            category_data = np.sort(data[valid & (accuracy == category)])
            if category_data.size == 0:
                continue
            response_label = f'{curve_label} / {category_label}'
            edges = np.histogram_bin_edges(
                category_data, bins=min(40, max(6, int(np.sqrt(category_data.size)))))
            counts, edges = np.histogram(category_data, bins=edges)
            widths = np.diff(edges)
            centers = edges[:-1] + widths / 2.0
            density_axis.step(
                centers, counts / (valid_count * widths), where='mid', color=color,
                linestyle='--', linewidth=1.4, label=f'{response_label} — observed')
            empirical = np.arange(1, category_data.size + 1) / valid_count
            cdf_axis.step(
                category_data, empirical, where='post', color=color, linestyle='--',
                linewidth=1.4, label=f'{response_label} — empirical')

        response_count = len(record['specification']['response_values'])
        correct_weights = record.get('correct_response_weights')
        warning = ''
        if correct_weights is None:
            correct_weights, warning = self._correct_response_weights(
                responses[valid], accuracy[valid], response_count)
        else:
            correct_weights = np.asarray(correct_weights, dtype=float)
        if correct_weights is None:
            return status_line + f' {warning}'
        observed_accuracy = float(np.mean(accuracy[valid]))
        weight_labels = (
            ['lower', 'upper'] if record['model'] == RATCLIFF_MODEL
            else [f'Accumulator {index}' for index in range(1, response_count + 1)])
        weights_text = ' / '.join(
            f'{name}: {weight:.2%}' for name, weight in zip(weight_labels, correct_weights))
        status_line += (
            f'<br>Observed correct/error: {observed_accuracy:.2%} / {1.0 - observed_accuracy:.2%}. '
            f'Correct-response weights ({weights_text}) reflect which response should be correct. '
            'Both displays use the same shared parameter set; switching display does not refit the model.')
        spread = max(float(np.ptp(data[valid])), abs(float(np.mean(data[valid]))) * 0.05, 1e-6)
        x_values = np.linspace(0.0, float(np.max(data[valid])) + 0.10 * spread, 400)
        try:
            correct_density = self._density(fitted_accuracy_pdf, record, x_values, True)
            error_density = self._density(fitted_accuracy_pdf, record, x_values, False)
            for density, category_label, color in zip(
                    (correct_density, error_density), ('Correct', 'Error'), category_colors):
                response_label = f'{curve_label} / {category_label}'
                density_axis.plot(
                    x_values, density, color=color, linewidth=2,
                    label=f'{response_label} — Fitted model')
                fitted_cdf = cumulative_trapezoid(density, x_values, initial=0.0)
                cdf_axis.plot(
                    x_values, fitted_cdf, color=color, linewidth=2,
                    label=f'{response_label} — Fitted model')
        except InterruptedError:
            raise
        except Exception as error:
            self.failed = True
            status_line += f' Curve unavailable: {html.escape(str(error))}.'
        return status_line


    @staticmethod
    def _correct_response_weights(responses, accuracy, response_count):
        """Infer design weights for each correct response from response and accuracy rows."""
        if response_count == 2:
            correct_responses = np.where(accuracy == 1.0, responses, 3 - responses)
        else:
            correct_responses = np.unique(responses[accuracy == 1.0])
            if correct_responses.size != 1:
                return None, (
                    'Fitted Correct/Error curves are unavailable because a multi-response model '
                    'requires one stable correct response per fitted group.')
            correct_responses = np.full(responses.shape, correct_responses[0], dtype=int)
        weights = np.bincount(correct_responses, minlength=response_count + 1)[1:].astype(float)
        weights /= np.sum(weights)
        return weights, ''


    @staticmethod
    def _number(value):
        """Format a diagnostic number for the status label.

        :param value: Numeric value to format.
        :return: Compact display string.
        """
        return f'{value:.4f}' if np.isfinite(value) else 'NA'
