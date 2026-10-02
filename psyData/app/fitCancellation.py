"""Shared cooperative-cancellation helpers for PsySummary model fitting."""


class FitCancelled(Exception):
    """Signal that a background model fit was cancelled by the user."""


def raise_if_fit_cancelled(cancel_check):
    """Raise ``FitCancelled`` when the supplied cancellation callback is active."""
    if cancel_check is not None and cancel_check():
        raise FitCancelled('Model fitting was cancelled by the user.')
