"""Offline feature store using DuckDB with Arrow integration.

Exports:
    OfflineFeatureStore(path) - DuckDB-based feature store
        .save(features, name, ...) - Save feature DataFrame
        .load(name, columns=None, ...) -> DataFrame - Load features
        .list_features() -> list[str] - List stored features
        .delete(name) - Delete stored features
        .point_in_time_join(features, labels, ...) -> DataFrame

    FeatureStoreError - Exception for store operations

This module provides a DuckDB-based offline feature store that enables:
- Zero-copy integration with Polars via Arrow
- Efficient storage and retrieval of computed features
- Point-in-time correct feature joins
- Partitioned storage for large datasets

Design Philosophy:
1. Zero-Copy: Arrow integration eliminates data copying
2. Performance: DuckDB provides fast analytics on cached features
3. Simplicity: Simple API for save/load operations
4. Correctness: Point-in-time joins prevent data leakage
"""

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    duckdb = None  # type: ignore[assignment]

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import duckdb
    import polars as pl


class FeatureStoreError(Exception):
    """Raised when feature store operations fail."""


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ArtifactKind = Literal["features", "labels", "targets", "artifact"]


def _quote_identifier(identifier: str) -> str:
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier must be a non-empty string")
    return '"' + identifier.replace('"', '""') + '"'


def _validate_table_name(table_name: str) -> None:
    if not table_name or not isinstance(table_name, str):
        raise ValueError("table_name must be a non-empty string")
    if not _IDENTIFIER_RE.match(table_name):
        raise ValueError(
            "table_name must be a valid SQL identifier containing only letters, "
            "numbers, and underscores, and may not start with a number"
        )


def _validate_columns(columns: list[str] | None) -> None:
    if columns is None:
        return
    if not isinstance(columns, list):
        raise TypeError("columns must be a list of strings")
    if len(columns) == 0:
        raise ValueError("columns list cannot be empty")
    if not all(isinstance(col, str) and col for col in columns):
        raise TypeError("all columns must be non-empty strings")


def _select_clause(columns: list[str] | None) -> str:
    return "*" if columns is None else ", ".join(_quote_identifier(col) for col in columns)


def _as_polars_frame(value: "pl.DataFrame | str", *, name: str) -> "pl.DataFrame | str":
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{name} table name must be a non-empty string")
        return value
    if not isinstance(value, pl.DataFrame):
        raise TypeError(f"{name} must be a Polars DataFrame or table name")
    if value.is_empty():
        raise ValueError(f"{name} DataFrame cannot be empty")
    return value


class OfflineFeatureStore:
    """DuckDB-based offline feature store with Arrow integration.

    Provides efficient storage and retrieval of computed features using
    DuckDB's columnar storage and Arrow zero-copy integration.

    Features:
        - Zero-copy Polars ↔ DuckDB via Arrow
        - Point-in-time correct feature retrieval
        - Partitioned storage for large datasets
        - SQL query support for filtering

    Example:
        >>> store = OfflineFeatureStore("features.duckdb")
        >>> store.save_features(features_df, "rsi_macd_features")
        >>> loaded = store.load_features("rsi_macd_features")
        >>> store.close()

        >>> # Or use context manager
        >>> with OfflineFeatureStore("features.duckdb") as store:
        ...     store.save_features(df, "my_features")
    """

    def __init__(
        self,
        path: str | Path | None = None,
        read_only: bool = False,
    ):
        """Initialize offline feature store.

        Args:
            path: Path to DuckDB database file. If None, creates in-memory DB.
            read_only: Whether to open database in read-only mode

        Raises:
            FeatureStoreError: If DuckDB not installed or connection fails

        Example:
            >>> # Persistent storage
            >>> store = OfflineFeatureStore("features.duckdb")

            >>> # In-memory (for testing)
            >>> store = OfflineFeatureStore()

            >>> # Read-only mode
            >>> store = OfflineFeatureStore("features.duckdb", read_only=True)
        """
        if not HAS_DUCKDB:
            raise FeatureStoreError("DuckDB not installed. Install with: pip install duckdb")

        if not HAS_POLARS:
            raise FeatureStoreError("Polars not installed. Install with: pip install polars")

        self.path = Path(path) if path else None
        self.read_only = read_only
        self._connection: duckdb.DuckDBPyConnection | None = None

        # Initialize connection
        self._connect()

    def _connect(self) -> None:
        """Establish DuckDB connection with Arrow support.

        Raises:
            FeatureStoreError: If connection or Arrow extension fails
        """
        try:
            # Connect to database (or in-memory)
            if self.path:
                # Ensure parent directory exists
                self.path.parent.mkdir(parents=True, exist_ok=True)

                # Connect with appropriate mode
                if self.read_only:
                    self._connection = duckdb.connect(str(self.path), read_only=True)
                else:
                    self._connection = duckdb.connect(str(self.path))
            else:
                # In-memory database
                self._connection = duckdb.connect(":memory:")

            # Note: Arrow integration is built into DuckDB 1.0+ core
            # No need to load any extension - Polars ↔ DuckDB works out of the box
            # Zero-copy operations are automatically enabled when available

        except Exception as e:
            raise FeatureStoreError(f"Failed to connect to DuckDB: {e}") from e

    @property
    def connection(self) -> "duckdb.DuckDBPyConnection":
        """Get active DuckDB connection.

        Returns:
            DuckDB connection object

        Raises:
            FeatureStoreError: If connection is closed
        """
        if self._connection is None:
            raise FeatureStoreError("Connection is closed. Call connect() first.")

        return self._connection

    def close(self) -> None:
        """Close DuckDB connection and release resources.

        Safe to call multiple times. After closing, the store cannot be used
        unless reconnected.

        Example:
            >>> store = OfflineFeatureStore("features.duckdb")
            >>> # ... use store ...
            >>> store.close()
        """
        if hasattr(self, "_connection") and self._connection is not None:
            try:
                self._connection.close()
            except Exception as e:
                warnings.warn(f"Error closing connection: {e}", UserWarning, stacklevel=2)
            finally:
                self._connection = None

    def __enter__(self) -> "OfflineFeatureStore":
        """Context manager entry.

        Returns:
            Self for context manager use
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit - ensures connection is closed."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensures connection is closed."""
        self.close()

    def is_connected(self) -> bool:
        """Check if connection is active.

        Returns:
            True if connected, False otherwise

        Example:
            >>> store = OfflineFeatureStore()
            >>> store.is_connected()
            True
            >>> store.close()
            >>> store.is_connected()
            False
        """
        return self._connection is not None

    def list_tables(self) -> list[str]:
        """List all tables in the feature store.

        Returns:
            List of table names

        Example:
            >>> store = OfflineFeatureStore("features.duckdb")
            >>> store.save_features(df, "my_features")
            >>> store.list_tables()
            ['my_features']
        """
        result = self.connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()

        return [row[0] for row in result]

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the store.

        Args:
            table_name: Name of table to check

        Returns:
            True if table exists, False otherwise

        Example:
            >>> store = OfflineFeatureStore()
            >>> store.table_exists("my_features")
            False
            >>> store.save_features(df, "my_features")
            >>> store.table_exists("my_features")
            True
        """
        _validate_table_name(table_name)
        return table_name in self.list_tables()

    def execute(self, query: str) -> "duckdb.DuckDBPyRelation":
        """Execute raw SQL query on the store.

        Args:
            query: SQL query to execute

        Returns:
            DuckDB relation with query results

        Example:
            >>> result = store.execute("SELECT COUNT(*) FROM my_features")
            >>> count = result.fetchone()[0]
        """
        return self.connection.sql(query)

    def save_features(
        self,
        df: "pl.DataFrame",
        table_name: str,
        mode: str = "replace",
    ) -> None:
        """Save features to DuckDB with zero-copy Arrow integration.

        Args:
            df: Polars DataFrame with features to save
            table_name: Name of table to create/update
            mode: Write mode - "replace" (default), "append", or "fail"
                - replace: Drop and recreate table
                - append: Add rows to existing table
                - fail: Raise error if table exists

        Raises:
            FeatureStoreError: If mode is invalid or table exists with mode="fail"
            ValueError: If df is empty or table_name is invalid

        Example:
            >>> store = OfflineFeatureStore("features.duckdb")
            >>> store.save_features(df, "rsi_features")
            >>> store.save_features(df2, "rsi_features", mode="append")
        """
        self.save_artifact(df, table_name, mode=mode, kind="features")

    def save_labels(
        self,
        df: "pl.DataFrame",
        table_name: str,
        mode: str = "replace",
    ) -> None:
        """Save supervised labels or targets to DuckDB.

        This mirrors :meth:`save_features` but keeps the caller's intent clear
        when the table stores labels rather than model inputs.
        """
        self.save_artifact(df, table_name, mode=mode, kind="labels")

    def save_artifact(
        self,
        df: "pl.DataFrame",
        table_name: str,
        mode: str = "replace",
        *,
        kind: ArtifactKind = "artifact",
    ) -> None:
        """Save a feature, label, target, or generic artifact table to DuckDB."""
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"df must be a Polars DataFrame, got {type(df).__name__}")

        if df.is_empty():
            raise ValueError("Cannot save empty DataFrame")

        _validate_table_name(table_name)

        if mode not in ("replace", "append", "fail"):
            raise ValueError(f"mode must be 'replace', 'append', or 'fail', got '{mode}'")
        if kind not in ("features", "labels", "targets", "artifact"):
            raise ValueError("kind must be one of: 'features', 'labels', 'targets', or 'artifact'")

        exists = self.table_exists(table_name)
        quoted_table = _quote_identifier(table_name)

        if mode == "fail" and exists:
            raise FeatureStoreError(f"Table '{table_name}' already exists and mode='fail'")

        if mode == "replace" and exists:
            self.connection.execute(f"DROP TABLE {quoted_table}")

        try:
            if mode == "append" and exists:
                self.connection.execute(f"INSERT INTO {quoted_table} SELECT * FROM df")
            else:
                self.connection.execute(f"CREATE TABLE {quoted_table} AS SELECT * FROM df")
        except Exception as e:
            raise FeatureStoreError(f"Failed to save {kind} to '{table_name}': {e}") from e

    def load_features(
        self,
        table_name: str,
        columns: list[str] | None = None,
        filter_expr: str | None = None,
        limit: int | None = None,
    ) -> "pl.DataFrame":
        """Load features from DuckDB with zero-copy Arrow integration.

        Args:
            table_name: Name of table to load
            columns: Optional list of columns to load (loads all if None)
            filter_expr: Optional SQL WHERE clause (without "WHERE" keyword)
                Example: "timestamp >= '2024-01-01'"
            limit: Optional row limit for result set

        Returns:
            Polars DataFrame with requested features

        Raises:
            FeatureStoreError: If table doesn't exist or query fails
            ValueError: If columns list is empty

        Example:
            >>> # Load all features
            >>> df = store.load_features("rsi_features")

            >>> # Load specific columns
            >>> df = store.load_features("rsi_features", columns=["timestamp", "rsi_14"])

            >>> # Load with filter
            >>> df = store.load_features("rsi_features", filter_expr="rsi_14 > 70")

            >>> # Load recent data with limit
            >>> df = store.load_features("rsi_features", limit=1000)
        """
        _validate_table_name(table_name)

        if not self.table_exists(table_name):
            raise FeatureStoreError(
                f"Table '{table_name}' does not exist. Available tables: {self.list_tables()}"
            )

        _validate_columns(columns)

        query = f"SELECT {_select_clause(columns)} FROM {_quote_identifier(table_name)}"

        # Add WHERE clause if provided
        if filter_expr:
            if not isinstance(filter_expr, str):
                raise TypeError("filter_expr must be a string")
            query += f" WHERE {filter_expr}"

        # Add LIMIT clause if provided
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            query += f" LIMIT {limit}"

        # Execute query and convert to Polars via Arrow (zero-copy)
        try:
            result = self.connection.execute(query)
            # Use .pl() method for zero-copy Arrow → Polars conversion
            return result.pl()
        except Exception as e:
            raise FeatureStoreError(f"Failed to load features from '{table_name}': {e}") from e

    def exact_join(
        self,
        left: "pl.DataFrame | str",
        right_table: str,
        on: list[str],
        *,
        columns: list[str] | None = None,
        filter_expr: str | None = None,
        how: str = "inner",
    ) -> "pl.DataFrame":
        """Join a DataFrame or stored table to another stored table by exact keys.

        This is the right tool for dense panel artifacts where features and
        labels share the same asset/time grain, for example ``symbol`` and
        ``timestamp``.
        """
        left = _as_polars_frame(left, name="left")
        if not isinstance(on, list) or not on:
            raise ValueError("on must be a non-empty list of column names")
        if not all(isinstance(col, str) and col for col in on):
            raise TypeError("all join keys must be non-empty strings")
        if how not in ("inner", "left", "outer", "semi", "anti"):
            raise ValueError("how must be one of: 'inner', 'left', 'outer', 'semi', 'anti'")
        _validate_columns(columns)
        right_load_columns = None if columns is None else list(dict.fromkeys([*on, *columns]))

        if isinstance(left, pl.DataFrame):
            missing = [key for key in on if key not in left.columns]
            if missing:
                raise ValueError(f"join keys {missing} not found in left DataFrame")
            right = self.load_features(
                right_table,
                columns=right_load_columns,
                filter_expr=filter_expr,
            )
            missing = [key for key in on if key not in right.columns]
            if missing:
                raise ValueError(f"join keys {missing} not found in right table '{right_table}'")
            return left.join(right, on=on, how=how)

        self._validate_table_exists(left, label="left")
        self._validate_table_exists(right_table, label="right")
        right_cols = self._table_columns(right_table)
        right_select_cols = right_cols if columns is None else columns
        missing = [key for key in on if key not in self._table_columns(left)]
        if missing:
            raise ValueError(f"join keys {missing} not found in left table '{left}'")
        missing = [key for key in on if key not in right_cols]
        if missing:
            raise ValueError(f"join keys {missing} not found in right table '{right_table}'")

        left_alias = "l"
        right_alias = "r"
        select_cols = [f"{left_alias}.*"]
        select_cols.extend(
            f"{right_alias}.{_quote_identifier(col)}" for col in right_select_cols if col not in on
        )
        join_predicate = " AND ".join(
            f"{left_alias}.{_quote_identifier(key)} = {right_alias}.{_quote_identifier(key)}"
            for key in on
        )
        join_type = "FULL OUTER" if how == "outer" else how.upper()
        filter_sql = f" AND ({filter_expr})" if filter_expr else ""
        query = (
            f"SELECT {', '.join(select_cols)} "
            f"FROM {_quote_identifier(left)} AS {left_alias} "
            f"{join_type} JOIN {_quote_identifier(right_table)} AS {right_alias} "
            f"ON {join_predicate}{filter_sql}"
        )
        try:
            return self.connection.execute(query).pl()
        except Exception as e:
            raise FeatureStoreError(
                f"Failed to exact join '{left}' with '{right_table}': {e}"
            ) from e

    def point_in_time_join(
        self,
        labels: "pl.DataFrame",
        features_table: str,
        timestamp_col: str = "timestamp",
        join_keys: list[str] | None = None,
        tolerance: str | None = None,
        feature_timestamp_col: str | None = None,
        columns: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> "pl.DataFrame":
        """Perform point-in-time correct join to prevent data leakage.

        Joins labels with features, ensuring each label only uses features
        that were available at or before the label's timestamp. This prevents
        look-ahead bias in ML models.

        Args:
            labels: DataFrame with labels and timestamps
            features_table: Name of features table to join
            timestamp_col: Name of timestamp column (default: "timestamp")
            join_keys: Optional list of additional join keys (e.g., ["symbol"])
                If None, joins only on time
            tolerance: Optional time tolerance (e.g., "1h", "1d")
                Maximum time difference allowed for a match
            feature_timestamp_col: Optional timestamp column in the feature table.
                Defaults to ``timestamp_col``. Use this for feature artifacts keyed
                by availability time such as ``available_at``.
            columns: Optional columns to load from the feature table. Join keys and
                timestamp columns are added automatically when omitted.
            filter_expr: Optional SQL WHERE clause pushed down before loading features.

        Returns:
            Polars DataFrame with labels and point-in-time correct features

        Raises:
            FeatureStoreError: If features table doesn't exist or query fails
            ValueError: If labels DataFrame is invalid or missing timestamp column

        Example:
            >>> # Simple time-based join
            >>> result = store.point_in_time_join(
            ...     labels=labels_df,
            ...     features_table="rsi_features"
            ... )

            >>> # Join with additional keys (e.g., per-symbol)
            >>> result = store.point_in_time_join(
            ...     labels=labels_df,
            ...     features_table="rsi_features",
            ...     join_keys=["symbol"]
            ... )

            >>> # Join with time tolerance (use features within 1 hour)
            >>> result = store.point_in_time_join(
            ...     labels=labels_df,
            ...     features_table="rsi_features",
            ...     tolerance="1h"
            ... )

        Notes:
            For each label row, this joins the most recent feature row where:
            - feature.timestamp <= label.timestamp (no look-ahead)
            - Additional join keys match (if specified)
            - Within tolerance window (if specified)

            This is critical for backtesting to avoid data leakage.
        """
        if not isinstance(labels, pl.DataFrame):
            raise TypeError(f"labels must be a Polars DataFrame, got {type(labels).__name__}")

        if labels.is_empty():
            raise ValueError("labels DataFrame cannot be empty")

        if timestamp_col not in labels.columns:
            raise ValueError(
                f"timestamp column '{timestamp_col}' not found in labels. "
                f"Available columns: {labels.columns}"
            )

        self._validate_table_exists(features_table, label="features")
        feature_timestamp_col = feature_timestamp_col or timestamp_col
        _validate_columns(columns)

        if join_keys:
            if not isinstance(join_keys, list):
                raise TypeError("join_keys must be a list of strings")
            missing_keys = [key for key in join_keys if key not in labels.columns]
            if missing_keys:
                raise ValueError(
                    f"join_keys {missing_keys} not found in labels. "
                    f"Available columns: {labels.columns}"
                )
        else:
            join_keys = []

        feature_columns = self._table_columns(features_table)
        required_feature_cols = [*join_keys, feature_timestamp_col]
        missing_feature_cols = [col for col in required_feature_cols if col not in feature_columns]
        if missing_feature_cols:
            raise ValueError(
                f"feature columns {missing_feature_cols} not found in '{features_table}'. "
                f"Available columns: {feature_columns}"
            )

        load_columns = columns
        if load_columns is not None:
            load_columns = list(dict.fromkeys([*required_feature_cols, *load_columns]))

        try:
            features_df = self.load_features(
                features_table,
                columns=load_columns,
                filter_expr=filter_expr,
            )

            if feature_timestamp_col != timestamp_col:
                features_df = features_df.rename({feature_timestamp_col: timestamp_col})

            sort_cols = join_keys + [timestamp_col] if join_keys else [timestamp_col]
            labels_sorted = labels.sort(sort_cols)
            features_sorted = features_df.sort(sort_cols)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Sortedness of columns cannot be checked when 'by' groups provided",
                    category=UserWarning,
                )
                result = labels_sorted.join_asof(
                    features_sorted,
                    on=timestamp_col,
                    by=join_keys or None,
                    strategy="backward",
                    tolerance=tolerance,
                )

            return result

        except Exception as e:
            raise FeatureStoreError(
                f"Failed to perform point-in-time join with '{features_table}': {e}"
            ) from e

    def _validate_table_exists(self, table_name: str, *, label: str) -> None:
        _validate_table_name(table_name)
        if not self.table_exists(table_name):
            raise FeatureStoreError(
                f"{label.capitalize()} table '{table_name}' does not exist. "
                f"Available tables: {self.list_tables()}"
            )

    def _table_columns(self, table_name: str) -> list[str]:
        self._validate_table_exists(table_name, label="table")
        result = self.connection.execute(f"DESCRIBE {_quote_identifier(table_name)}").fetchall()
        return [row[0] for row in result]

    def __repr__(self) -> str:
        """String representation of feature store."""
        location = f"path={self.path}" if self.path else "in-memory"

        status = "connected" if self.is_connected() else "closed"
        mode = "read-only" if self.read_only else "read-write"

        return f"OfflineFeatureStore({location}, {mode}, {status})"
