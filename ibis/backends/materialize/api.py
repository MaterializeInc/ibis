"""Materialize backend API functions."""

from __future__ import annotations

import ibis.expr.types as ir
from ibis.backends.materialize import operations as mz_ops


def mz_now() -> ir.TimestampScalar:
    """Return the logical timestamp in Materialize.

    This returns Materialize's `mz_now()` function, which provides the logical
    time at which the query was executed. This is different from `ibis.now()`
    (PostgreSQL's `now()`) which returns the system clock time.

    Key differences from `now()`:
    - Returns logical timestamp (for streaming/incremental computation)
    - Can be used in temporal filters in materialized views
    - Value represents query execution time in Materialize's consistency model

    Returns
    -------
    TimestampScalar
        An expression representing Materialize's logical timestamp

    Examples
    --------
    >>> import ibis
    >>> from ibis.backends.materialize.api import mz_now
    >>> # Get the current logical timestamp
    >>> mz_now()

    Use in temporal filters (e.g., last 30 seconds of data):

    >>> events = con.table("events")
    >>> # Best practice: Isolate mz_now() on one side of comparison
    >>> recent = events.filter(mz_now() > events.event_ts + ibis.interval(seconds=30))

    Compare with regular now():

    >>> # System clock time (wall clock)
    >>> ibis.now()
    >>> # Logical timestamp (streaming time)
    >>> mz_now()

    See Also
    --------
    ibis.now : PostgreSQL's now() function (system clock time)

    Notes
    -----
    mz_now() is fundamental to Materialize's streaming SQL model and is used
    for temporal filters in materialized views to enable incremental computation.

    **Best Practice**: When using mz_now() in temporal filters, isolate it on one
    side of the comparison for optimal incremental computation:

    - ✅ Good: `mz_now() > created_at + INTERVAL '1 day'`
    - ❌ Bad: `mz_now() - created_at > INTERVAL '1 day'`

    This pattern enables Materialize to efficiently compute incremental updates
    without reprocessing the entire dataset.

    References
    ----------
    - Function documentation: https://materialize.com/docs/sql/functions/now_and_mz_now/
    - Idiomatic patterns: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#temporal-filters
    """
    return mz_ops.MzNow().to_expr()
