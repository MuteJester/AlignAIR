import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const errors: DocPage = {
  slug: "errors",
  title: "Errors and diagnostics API",
  section: "Reference",
  lead: "Exception types, diagnostic fields, operation context, and error reporting in alignair.errors.",
  body: () => (
    <>
      <H2 id="handling">Handle an AlignAIR failure</H2>
      <p>
        Catch a domain exception when recovery depends on the failure type, or <code>AlignAIRError</code> to handle
        classified AlignAIR failures together. Native filesystem exceptions and unexpected Python exceptions can
        still propagate from library calls.
      </p>
      <CodeBlock lang="python" code={`from alignair import Aligner
from alignair.errors import AlignAIRError, RegistryError, report_error

try:
    aligner = Aligner.from_pretrained("alignair-igh-human")
except RegistryError as error:
    report_error(error, json_output=True)
except AlignAIRError as error:
    report_error(error)`} />
      <p>
        <code>report_error</code> writes to stderr and returns an exit code; it does not terminate the process.
        Use exception types or <code>error.code</code> for branching. Message text is intended for people.
        The original cause, when supplied with <code>raise ... from ...</code>, is available as <code>error.__cause__</code>.
      </p>

      <H2 id="exceptions">Exception reference</H2>
      <p>
        Import these classes from <code>alignair.errors</code>. Each inherits from <code>AlignAIRError</code>.
        The base is also exported as <code>alignair.AlignAIRError</code>.
      </p>
      <DocTable
        head={["Exception", "Use"]}
        rows={[
          [<code>InputError</code>, "Invalid records, sequence values, or input-file contents. Also a ValueError."],
          [<code>ConfigurationError</code>, "Invalid options or incompatible configuration. Also a ValueError."],
          [<code>DataContractError</code>, "A record, batch, or component violates its declared contract. Also a RuntimeError."],
          [<code>ModelError</code>, "Missing, incompatible, or unusable model artifacts. Also a ValueError."],
          [<code>ModelFormatError</code>, "A ModelError for malformed, truncated, or unsupported model containers."],
          [<code>ModelSectionError</code>, "A ModelError for a missing model section. Also a KeyError."],
          [<code>ArtifactIntegrityError</code>, "Artifact bytes or bound identities fail verification. Also a RuntimeError."],
          [<code>ModelIntegrityError</code>, "Both a ModelError and an ArtifactIntegrityError."],
          [<code>RegistryError</code>, "Registry lookup or model transfer failed. Also a RuntimeError and ValueError."],
          [<code>TrainingError</code>, "Training failed. Also a RuntimeError."],
          [<code>TrainingDataError</code>, "Both a TrainingError and a DataContractError for invalid training records."],
          [<code>InferenceError</code>, "Prediction or result assembly failed. Also a RuntimeError."],
          [<code>NumericalError</code>, "Non-finite outputs, losses, or gradients. Also a RuntimeError and ValueError."],
          [<code>MissingDependencyError</code>, "A requested feature needs an optional dependency. Also an ImportError."],
          [<code>AccessPolicyError</code>, "Protected dataset or artifact access is refused. Also a RuntimeError."],
          [<code>OperationIOError</code>, "An operating-system failure translated at the CLI boundary. Also a RuntimeError."],
          [<code>InternalError</code>, "An unexpected defect classified at the CLI boundary. Also a RuntimeError."],
          [<code>OperationCancelledError</code>, "An operation interrupted by the user at the CLI boundary."],
        ]}
      />
      <p>
        Existing domain exceptions keep their import paths and built-in compatibility. For example,
        <code> alignair.train.guards.TrainingConfigError</code> remains a <code>ValueError</code>, and
        <code> alignair.train.guards.NonFiniteLossError</code> remains a <code>RuntimeError</code>.
        Both also inherit from <code>AlignAIRError</code>.
      </p>

      <H2 id="diagnostics">Constructor and diagnostic fields</H2>
      <CodeBlock lang="python" code={`AlignAIRError(
    message: str = "",
    *,
    context: Mapping | None = None,
    hint: str | None = None,
    retryable: bool = False,
)`} />
      <p>
        Domain classes accept the same arguments unless their own constructor specifies otherwise.
        <code> context</code> accepts operation metadata, <code>hint</code> describes a possible recovery action,
        and <code>retryable</code> records whether the caller may consider a retry. No retry happens automatically.
      </p>
      <DocTable
        head={["Attribute", "Meaning"]}
        rows={[
          [<code>code: str</code>, "Stable failure identifier, such as ALIGNAIR_INPUT_INVALID."],
          [<code>category: str</code>, "Domain category, such as input, model, registry, or training."],
          [<code>exit_code: ExitCode</code>, "Suggested application exit status."],
          [<code>message: str</code>, "Human-readable explanation; also returned by str(error)."],
          [<code>hint: str | None</code>, "Optional recovery guidance."],
          [<code>context: dict</code>, "An independent copy of the metadata captured when the exception was created."],
          [<code>retryable: bool</code>, "Retry metadata; defaults to False."],
        ]}
      />
      <p>
        <code>to_dict() → dict</code> returns the versioned diagnostic envelope.
        <code> to_json() → str</code> serializes it as strict JSON. Both include
        <code> schema_version: 1</code> and <code>type</code> (the exception class name), followed by the fields above.
        The serialized <code>exit_code</code> is an integer. Raw causes and tracebacks are excluded.
      </p>
      <CodeBlock lang="python" code={`from alignair.errors import InputError

error = InputError(
    "Required column 'sequence' is missing.",
    context={"path": "reads.tsv", "stage": "csv_header"},
    hint="Add the required column to the input table.",
)
diagnostic = error.to_dict()`} />
      <CodeBlock lang="json" code={`{
  "schema_version": 1,
  "type": "InputError",
  "code": "ALIGNAIR_INPUT_INVALID",
  "category": "input",
  "exit_code": 1,
  "message": "Required column 'sequence' is missing.",
  "hint": "Add the required column to the input table.",
  "context": {"path": "reads.tsv", "stage": "csv_header"},
  "retryable": false
}`} />

      <H2 id="context">Operation context and cause chains</H2>
      <p>
        <code>error_context(**context)</code> attaches metadata to exceptions created inside its scope.
        Scopes nest; explicit exception context takes precedence. <code>current_error_context() → dict</code>
        returns a copy of the active scope.
      </p>
      <CodeBlock lang="python" code={`from alignair.errors import TrainingDataError, error_context

def germline_start(record, gene):
    field = f"{gene.lower()}_germline_start"
    try:
        return record[field]
    except KeyError as cause:
        raise TrainingDataError(
            "Required germline coordinate is missing.",
            context={"gene": gene, "field": field},
            hint="Check the training record annotations.",
        ) from cause

with error_context(run_id="run-123", step=42, record_id="read-7"):
    start = germline_start({"sequence_id": "read-7"}, "D")`} />
      <p>
        The exception captures the run, step, record, gene, and field. Context is restored when the scope exits,
        including when an exception escapes. Asyncio tasks have isolated contexts; threads and worker processes
        require explicit context propagation. Entering a scope does not modify existing exception objects.
      </p>
      <p>
        Diagnostic context bounds nesting, collection sizes, and string values. It redacts credential and sequence
        fields by key, removes URL credentials and fragments, and replaces URL query strings. Non-finite numbers
        become strings; cycles and unsupported objects receive descriptive placeholders. Messages and debug
        tracebacks are not redacted, so keep credentials and sequence payloads out of message text.
      </p>
      <p>
        Exceptions support pickling for worker communication. Diagnostic metadata and custom attributes are retained;
        custom attributes must themselves be pickle-compatible. Tracebacks and Python cause chains remain local
        to the process that raised the exception.
      </p>

      <H2 id="reporting">Reporting and logging</H2>
      <CodeBlock lang="python" code={`format_error(error, *, json_output=False) -> str
report_error(error, *, stream=None, debug=None, json_output=None) -> int
log_error(logger, error, *, level=logging.ERROR, debug=False) -> None
error_reporting(*, debug=False, json_output=False)`} />
      <DocTable
        head={["Function", "Behavior"]}
        rows={[
          [<code>format_error</code>, "Returns text or JSON without writing output or changing the traceback."],
          [<code>report_error</code>, "Writes one diagnostic to stream (stderr by default) and returns the error's exit code."],
          [<code>log_error</code>, "Writes through the supplied logger and attaches the diagnostic as LogRecord.alignair_error."],
          [<code>error_reporting</code>, "Context manager providing task-local defaults for report_error. Defaults are restored on exit."],
        ]}
      />
      <p>
        In <code>report_error</code>, a <code>None</code> option inherits the active reporting default; an explicit
        boolean overrides it. <code>debug=True</code> includes the chained traceback. When JSON and debug are both
        enabled, the traceback is a field in the same JSON object. <code>log_error</code> also includes the traceback
        only when <code>debug=True</code>. These helpers do not configure logging or install global exception hooks.
        Report a failure at the application boundary to avoid duplicate log entries.
      </p>

      <H2 id="cli">Command-line reporting</H2>
      <CodeBlock code={`alignair predict --model model.alignair --input reads.fasta --out output.tsv --error-format json
alignair --debug predict --model model.alignair --input reads.fasta --out output.tsv`} />
      <p>
        Reporting flags work before or after a subcommand, including nested model commands. Runtime diagnostics go
        to stderr. Argparse help and syntax errors retain their usual output format. A failed operation may leave
        documented partial results; its exit status does not imply that output files were rolled back.
      </p>
      <DocTable
        head={["ExitCode member", "Value", "Meaning"]}
        rows={[
          [<code>SUCCESS</code>, "0", "Command completed successfully."],
          [<code>ERROR</code>, "1", "Classified operational failure."],
          [<code>USAGE</code>, "2", "Command-line syntax or usage error."],
          [<code>INTERNAL</code>, "70", "Unexpected exception caught at the CLI boundary."],
          [<code>INTERRUPTED</code>, "130", "User interruption."],
        ]}
      />

      <H2 id="extensions">Define an application-specific exception</H2>
      <p>
        Subclass the closest domain type and assign a unique code when callers need to distinguish a new failure.
        Preserve cause chains when translating lower-level exceptions.
      </p>
      <CodeBlock lang="python" code={`from alignair.errors import DataContractError

class SampleMetadataError(DataContractError):
    code = "PIPELINE_SAMPLE_METADATA_INVALID"

raise SampleMetadataError(
    "Sample metadata contains duplicate identifiers.",
    context={"path": "samples.tsv", "field": "sample_id"},
)`} />
      <p>
        For input formats and output validation, see <DocLink to="python-api">Python API</DocLink> and
        <DocLink to="airr-fields"> AIRR output fields</DocLink>.
      </p>
    </>
  ),
};

export const errorPages: DocPage[] = [errors];
