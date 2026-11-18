"""AIRR rearrangement tables (``tables`` reads them): the repertoire, QC and validation report (``report``,
``alignair analyze``) and the tool-to-tool comparison (``compare``, ``alignair compare``)."""
from .report import analyze_file, analyze_rows, format_text, qc, repertoire
from .validate import validate_airr

__all__ = ["analyze_file", "analyze_rows", "format_text", "qc", "repertoire", "validate_airr"]
