"""Materialize backend API functions."""

from __future__ import annotations

import ibis
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


def mz_top_k(
    table: ir.Table,
    k: int,
    by: list[str] | str,
    order_by: list[str] | str | list[tuple[str, bool]],
    desc: bool = True,
    group_size: int | None = None,
) -> ir.Table:
    """Get top-k rows per group using idiomatic Materialize SQL.

    Parameters
    ----------
    table : Table
        The input table
    k : int
        Number of rows to keep per group
    by : str or list of str
        Column(s) to group by (partition keys)
    order_by : str or list of str or list of (str, bool)
        Column(s) to order by within each group.
        If tuple, second element is True for DESC, False for ASC.
    desc : bool, default True
        Default sort direction when order_by is just column names
    group_size : int, optional
        Materialize-specific query hint to control memory usage.
        For k=1: Sets DISTINCT ON INPUT GROUP SIZE
        For k>1: Sets LIMIT INPUT GROUP SIZE
        Ignored for non-Materialize backends.

    Returns
    -------
    Table
        Top k rows per group

    Examples
    --------
    >>> import ibis
    >>> from ibis.backends.materialize.api import mz_top_k
    >>> con = ibis.materialize.connect(...)
    >>> orders = con.table("orders")
    >>>
    >>> # Top 3 items per order by subtotal
    >>> mz_top_k(orders, k=3, by="order_id", order_by="subtotal", desc=True)
    >>>
    >>> # Top seller per region (k=1 uses DISTINCT ON)
    >>> sales = con.table("sales")
    >>> mz_top_k(sales, k=1, by="region", order_by="total_sales")
    >>>
    >>> # Multiple order-by columns with explicit direction
    >>> events = con.table("events")
    >>> mz_top_k(
    ...     events,
    ...     k=10,
    ...     by="user_id",
    ...     order_by=[
    ...         ("priority", True),   # DESC (high priority first)
    ...         ("timestamp", False)   # ASC (oldest first)
    ...     ]
    ... )
    >>>
    >>> # Use group_size hint to optimize memory usage
    >>> mz_top_k(
    ...     orders,
    ...     k=5,
    ...     by="customer_id",
    ...     order_by="order_date",
    ...     group_size=1000  # Hint: expect ~1000 orders per customer
    ... )

    Notes
    -----
    The `group_size` parameter helps Materialize optimize memory usage by
    providing an estimate of the expected number of rows per group. This is
    particularly useful for large datasets.

    References
    ----------
    https://materialize.com/docs/transform-data/idiomatic-materialize-sql/top-k/
    https://materialize.com/docs/transform-data/optimization/#query-hints
    """
    from ibis.backends.materialize import Backend as MaterializeBackend

    # Normalize inputs
    if isinstance(by, str):
        by = [by]

    # Normalize order_by to list of (column, desc) tuples
    if isinstance(order_by, str):
        order_by = [(order_by, desc)]
    elif isinstance(order_by, list):
        if order_by and not isinstance(order_by[0], tuple):
            order_by = [(col, desc) for col in order_by]

    backend = table._find_backend()

    if isinstance(backend, MaterializeBackend):
        if k == 1:
            return _top_k_distinct_on(table, by, order_by, group_size)
        else:
            return _top_k_lateral(table, k, by, order_by, group_size)
    else:
        return _top_k_generic(table, k, by, order_by)


def _top_k_distinct_on(table, by, order_by, group_size):
    """Use DISTINCT ON for k=1 in Materialize."""
    backend = table._find_backend()
    table_name = table.get_name()

    # Build column lists
    by_cols = ", ".join(by)
    order_exprs = ", ".join(
        [f"{col} {'DESC' if desc else 'ASC'}" for col, desc in order_by]
    )

    # Add group size hint if provided
    options_clause = ""
    if group_size is not None:
        options_clause = f"\n    OPTIONS (DISTINCT ON INPUT GROUP SIZE = {group_size})"

    sql = f"""
    SELECT DISTINCT ON({by_cols}) *
    FROM {table_name}{options_clause}
    ORDER BY {by_cols}, {order_exprs}
    """

    return backend.sql(sql)


def _top_k_lateral(table, k, by, order_by, group_size):
    """Use LATERAL join pattern for k>1 in Materialize."""
    backend = table._find_backend()
    table_name = table.get_name()

    # Build column lists
    by_cols = ", ".join(by)

    # Get all columns except group by columns for the lateral select
    all_cols = list(table.columns)
    lateral_cols = [col for col in all_cols if col not in by]
    lateral_select = ", ".join(lateral_cols)

    # Build WHERE clause for lateral join
    where_clause = " AND ".join([f"{col} = grp.{col}" for col in by])

    # Build ORDER BY for lateral subquery
    lateral_order = ", ".join(
        [f"{col} {'DESC' if desc else 'ASC'}" for col, desc in order_by]
    )

    # Build final ORDER BY (group keys + order keys)
    final_order_cols = ", ".join(
        [f"{col} {'DESC' if desc else 'ASC'}" for col, desc in order_by]
    )

    # Add group size hint if provided
    options_clause = ""
    if group_size is not None:
        options_clause = f"\n                OPTIONS (LIMIT INPUT GROUP SIZE = {group_size})"

    sql = f"""
    SELECT grp.{by_cols}, lateral_data.*
    FROM (SELECT DISTINCT {by_cols} FROM {table_name}) grp,
         LATERAL (
             SELECT {lateral_select}
             FROM {table_name}
             WHERE {where_clause}{options_clause}
             ORDER BY {lateral_order}
             LIMIT {k}
         ) lateral_data
    ORDER BY {by_cols}, {final_order_cols}
    """

    return backend.sql(sql)


def _top_k_generic(table, k, by, order_by):
    """Generic ROW_NUMBER() implementation for non-Materialize backends."""
    # Build window function
    order_keys = [ibis.desc(col) if desc else ibis.asc(col) for col, desc in order_by]

    return (
        table.mutate(_rn=ibis.row_number().over(group_by=by, order_by=order_keys))
        .filter(ibis._["_rn"] <= k)
        .drop("_rn")
    )
