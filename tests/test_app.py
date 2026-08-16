import builtins
import importlib
import io
import json
import os
import runpy
import sys
import warnings
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from fastmcp import Client
from fastmcp.exceptions import ToolError

import mcp_pyphotomol
from mcp_pyphotomol.paths import (
    RESULTS_DIR_ENV_VAR,
    USER_DATA_DIR_NAME,
    get_example_data_root,
    get_user_data_root,
)
import mcp_pyphotomol.server as photomol_server
import mcp_pyphotomol.tools._photomol as photomol_tools
from mcp_pyphotomol.main import run_app

EXAMPLE_DATA_DIR = get_example_data_root()
MASS_EXAMPLE_FILES = {
    "masses_monomer_1nM.csv",
    "masses_monomer_2nM.csv",
    "masses_monomer_4nM.csv",
    "masses_monomer_8nM.csv",
    "masses_monomer_16nM.csv",
    "masses_monomer_32nM.csv",
    "masses_monomer_64nM.csv",
}
CONTRAST_EXAMPLE_FILE = "contrasts.csv"
NOTEBOOK_DEMO_FILE = "demo.h5"
EXPECTED_EXAMPLE_MODEL_NAMES = [
    "1.0 nM",
    "2.0 nM",
    "4.0 nM",
    "8.0 nM",
    "16.0 nM",
    "32.0 nM",
    "64.0 nM",
]


class AsyncLineStream:
    """Async iterator that feeds predetermined lines into stdio transport tests."""

    def __init__(self, lines):
        self._lines = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration from None


class AsyncTextSink:
    """Async stdout sink used so stdio transport tests do not touch pytest capture."""

    async def write(self, text):
        return None

    async def flush(self):
        return None


def test_package_has_version():
    """Verify the package exposes distribution metadata through ``__version__``."""
    assert mcp_pyphotomol.__version__ is not None


def test_user_data_root_defaults_to_home(monkeypatch):
    """Verify default user data is written outside the installed package tree."""
    monkeypatch.delenv(RESULTS_DIR_ENV_VAR, raising=False)
    assert get_user_data_root() == Path.home() / USER_DATA_DIR_NAME


def test_user_data_root_uses_configured_results_dir(monkeypatch, tmp_path):
    """Verify users can choose the folder for MCP output files."""
    results_dir = tmp_path / "results"
    monkeypatch.setenv(RESULTS_DIR_ENV_VAR, str(results_dir))
    assert get_user_data_root() == results_dir


def test_server_instructions_show_current_data_folder():
    """Verify MCP clients can see where results are saved at initialization."""
    assert photomol_server.DATA_DIR in photomol_server.SERVER_INSTRUCTIONS
    assert "Plots and log files for this session are saved in:" in photomol_server.SERVER_INSTRUCTIONS


def test_example_data_files_are_present():
    """Verify the bundled CSV and notebook fixtures needed by integration tests exist."""
    files = {path.name for path in EXAMPLE_DATA_DIR.glob("*.csv")}
    assert MASS_EXAMPLE_FILES | {CONTRAST_EXAMPLE_FILE} <= files
    assert (EXAMPLE_DATA_DIR / NOTEBOOK_DEMO_FILE).is_file()


@pytest.mark.asyncio
async def test_stdio_transport_ignores_blank_lines():
    """
    Verify blank stdio lines are filtered before JSON-RPC parsing.

    The transport should deliver the first real JSON-RPC message instead of
    turning an empty line into a stream exception.
    """
    from mcp_pyphotomol.stdio import stdio_server_ignoring_empty_lines

    ping = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
    stdin = AsyncLineStream(["\n", "   \n", ping])

    async with stdio_server_ignoring_empty_lines(
        stdin=stdin,
        stdout=AsyncTextSink(),
    ) as (read_stream, write_stream):
        received = await read_stream.receive()
        await write_stream.aclose()
        await read_stream.aclose()

    assert not isinstance(received, Exception)
    # Support both MCP 1.x (.root.method) and MCP 2.x (.method) APIs
    message = received.message.root if hasattr(received.message, 'root') else received.message
    assert message.method == "ping"


@pytest.mark.asyncio
async def test_stdio_transport_defaults_to_process_stdin():
    """
    Verify the hardened stdio transport wraps process stdin by default.

    The fake stdio server inspects the generated stdin iterator so this test can
    check the filtering behavior without using the real process streams.
    """
    import mcp_pyphotomol.stdio as stdio_module

    calls = {}
    observed_lines = []

    @asynccontextmanager
    async def fake_stdio_server(stdin=None, stdout=None):
        calls["stdin"] = stdin
        calls["stdout"] = stdout
        async for line in stdin:
            observed_lines.append(line)
            break
        yield "read-stream", "write-stream"

    with (
        patch.object(stdio_module.sys, "stdin", SimpleNamespace(buffer="raw-stdin")),
        patch.object(stdio_module, "TextIOWrapper", lambda buffer, encoding: f"{encoding}:{buffer}"),
        patch.object(
            stdio_module.anyio,
            "wrap_file",
            lambda stream: AsyncLineStream(["\n", '{"jsonrpc":"2.0","id":1,"method":"ping"}\n']),
        ),
        patch.object(stdio_module, "stdio_server", fake_stdio_server),
    ):
        async with stdio_module.stdio_server_ignoring_empty_lines():
            pass

    assert calls["stdout"] is None
    assert hasattr(calls["stdin"], "__aiter__")
    assert observed_lines[0].strip().startswith("{")


@pytest.mark.asyncio
async def test_run_stdio_ignoring_empty_lines_uses_hardened_transport():
    """
    Verify the hardened stdio runner wires its streams into FastMCP.

    The fake low-level server records lifecycle calls only; it does not emulate
    MCP behavior beyond confirming the runner passes initialized streams and
    options to ``_mcp_server.run``.
    """
    import fastmcp.server.context as fastmcp_context
    import fastmcp.utilities.cli as fastmcp_cli
    import fastmcp.utilities.logging as fastmcp_logging
    import mcp_pyphotomol.stdio as stdio_module

    calls = []

    @asynccontextmanager
    async def fake_lifespan_manager():
        calls.append("lifespan-entered")
        yield

    @asynccontextmanager
    async def fake_stdio_server():
        calls.append("stdio-entered")
        yield "read-stream", "write-stream"

    @contextmanager
    def fake_temporary_log_level(log_level):
        calls.append(("log-level", log_level))
        yield

    class FakeLowLevelServer:
        """Minimal FastMCP low-level server stand-in for runner wiring checks."""

        def create_initialization_options(self, **kwargs):
            notification_options = kwargs["notification_options"]
            calls.append(("init-options", notification_options.tools_changed))
            return "init-options"

        async def run(self, read_stream, write_stream, init_options, stateless=False):
            assert read_stream
            assert write_stream
            assert init_options
            calls.append(("run", stateless))

    fake_server = SimpleNamespace(
        name="fake-server",
        _lifespan_manager=fake_lifespan_manager,
        _mcp_server=FakeLowLevelServer(),
    )

    with (
        patch.object(fastmcp_cli, "log_server_banner", lambda server: calls.append(("banner", server.name))),
        patch.object(fastmcp_context, "set_transport", lambda transport: calls.append(("set", transport)) or "token"),
        patch.object(fastmcp_context, "reset_transport", lambda token: calls.append(("reset", token))),
        patch.object(fastmcp_logging, "temporary_log_level", fake_temporary_log_level),
        patch.object(stdio_module, "stdio_server_ignoring_empty_lines", fake_stdio_server),
    ):
        await stdio_module.run_stdio_ignoring_empty_lines(
            fake_server,
            show_banner=True,
            log_level="DEBUG",
            stateless=True,
        )

    assert any(call[0] == "banner" for call in calls)
    assert ("set", "stdio") in calls
    assert ("log-level", "DEBUG") in calls
    assert ("init-options", True) in calls
    assert ("run", True) in calls
    assert any(call[0] == "reset" for call in calls)


def test_run_stdio_sync_wrapper_dispatches_anyio_run():
    """Verify the synchronous stdio wrapper delegates execution to ``anyio.run``."""
    import mcp_pyphotomol.stdio as stdio_module

    calls = []

    with patch.object(stdio_module.anyio, "run", lambda runner: calls.append(runner)):
        stdio_module.run_stdio("server", stateless=True)

    assert calls


@pytest.mark.asyncio
async def test_mcp_server_tools_with_example_data(isolated_tool_log_dir):
    """
    Exercise the public MCP tools against bundled example data.

    This is a broad integration smoke test: it checks tool registration, import
    workflows, histogram/fitting calls, plotting branches, calibration setup,
    and that expected files are written to the isolated log directory.
    """
    log_dir = isolated_tool_log_dir

    async with Client(mcp_pyphotomol.mcp) as client:
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools}

        assert {
            "list_MP_files_in_folder",
            "reset_analyzer",
            "reset_calibrator",
            "get_model_names",
            "import_folder",
            "load_example_data",
            "create_histogram_automatic",
            "fit_multi_gaussian",
            "show_fitted_parameters",
            "load_example_data_for_calibration",
        } <= tool_names

        result = await client.call_tool("get_user_name", {})
        assert isinstance(result.data, str)
        assert result.data

        result = await client.call_tool(
            "list_MP_files_in_folder",
            {"folder_path": str(EXAMPLE_DATA_DIR)},
        )
        listed_files = {file_name.strip() for file_name in result.data.split(",")}
        assert MASS_EXAMPLE_FILES | {CONTRAST_EXAMPLE_FILE, NOTEBOOK_DEMO_FILE} <= listed_files

        result = await client.call_tool(
            "import_folder",
            {"folder_path": str(EXAMPLE_DATA_DIR), "pattern": "does-not-exist"},
        )
        assert result.data == f"No files found in {EXAMPLE_DATA_DIR}."

        await client.call_tool("reset_analyzer", {})
        result = await client.call_tool(
            "import_single_file",
            {"file_path": str(EXAMPLE_DATA_DIR / "masses_monomer_1nM.csv")},
        )
        assert result.data == f"Data imported successfully from {EXAMPLE_DATA_DIR / 'masses_monomer_1nM.csv'}."

        result = await client.call_tool("get_model_names", {})
        assert result.data == ["masses_monomer_1nM"]

        with pytest.raises(ToolError, match="Histogram for model masses_monomer_1nM has not been created"):
            await client.call_tool("fit_multi_gaussian", {})

        result = await client.call_tool(
            "create_histogram_manual",
            {"min_value": 0, "max_value": 300, "bin_width": 8},
        )
        assert "Histograms were created successfully" in result.data
        assert "Bin width: 8" in result.data

        result = await client.call_tool(
            "fit_multi_gaussian",
            {"peaks_guess": [80, 160], "mean_tolerance": 80, "std_tolerance": 80},
        )
        assert result.data == "Multi-gaussian fitting completed successfully."

        await client.call_tool("reset_analyzer", {})
        result = await client.call_tool(
            "import_folder",
            {"folder_path": str(EXAMPLE_DATA_DIR), "pattern": "masses_monomer"},
        )
        assert result.data == f"{len(MASS_EXAMPLE_FILES)} files were imported successfully from {EXAMPLE_DATA_DIR}."

        result = await client.call_tool("get_model_names", {})
        assert {f.removesuffix(".csv") for f in MASS_EXAMPLE_FILES} == set(result.data)

        await client.call_tool("reset_analyzer", {})
        result = await client.call_tool("load_example_data", {})
        assert result.data == "Example data loaded successfully."

        result = await client.call_tool("get_model_names", {})
        assert result.data == EXPECTED_EXAMPLE_MODEL_NAMES

        result = await client.call_tool("create_histogram_automatic", {})
        assert "Histograms were created successfully" in result.data
        assert "Using masses: True" in result.data

        result = await client.call_tool("fit_multi_gaussian", {"experiment": "not-present"})
        assert result.data == "Multi-gaussian fitting completed successfully."

        result = await client.call_tool("fit_multi_gaussian", {})
        assert result.data == "Multi-gaussian fitting completed successfully."

        result = await client.call_tool("show_fitted_parameters", {})
        fitted_parameters = json.loads(result.data)
        assert {row["name"] for row in fitted_parameters} == set(EXPECTED_EXAMPLE_MODEL_NAMES)
        assert all("Position / kDa" in row for row in fitted_parameters)

        result = await client.call_tool(
            "update_plot_config",
            {
                "plot_width": 640,
                "plot_height": 480,
                "plot_type": "browser",
                "x_range": [0, 300],
            },
        )
        assert result.data == "Plot configuration updated successfully."

        result = await client.call_tool(
            "update_legend_config",
            {"add_percentage_to_legend": True, "line_width": 2},
        )
        assert result.data == "Legend configuration updated successfully."

        result = await client.call_tool(
            "update_layout_config",
            {"stacked": False, "show_subplot_titles": True},
        )
        assert result.data == "Layout configuration updated successfully."

        result = await client.call_tool("update_axis_config", {"n_y_axis_ticks": 5})
        assert result.data == "Axis configuration updated successfully."

        result = await client.call_tool(
            "plot_histograms",
            {"colors_hist": "red", "save_as_html": True},
        )
        assert result.data == "Histogram plot created successfully, but no valid image format was specified for saving."

        result = await client.call_tool("get_legends_dataframe", {"repeat_colors": False})
        legends = json.loads(result.data)
        assert {column for row in legends for column in row} >= {"legends", "color", "select", "show_legend"}

        result = await client.call_tool(
            "plot_histograms_and_fits",
            {"legends_df": json.dumps(legends), "colors_hist": "blue", "save_as_html": True},
        )
        assert result.data == "Histograms and fits plot created successfully, but no valid image format was specified for saving."
        assert any(path.name.startswith("histogram_") for path in log_dir.glob("*.html"))
        assert any(path.name.startswith("histograms_and_fits_") for path in log_dir.glob("*.html"))

        await client.call_tool("reset_calibrator", {})
        result = await client.call_tool("load_example_data_for_calibration", {})
        assert result.data == "Example calibration data loaded successfully."

        result = await client.call_tool("get_model_names", {"calibrator": True})
        assert result.data == ["file1", "file2"]

        result = await client.call_tool(
            "create_histogram_automatic",
            {"use_masses": False, "calibrator": True},
        )
        assert "Using contrasts: True" in result.data

        result = await client.call_tool("fit_multi_gaussian", {"calibrator": True})
        assert result.data == "Multi-gaussian fitting completed successfully."

        result = await client.call_tool("show_fitted_parameters", {"calibrator": True})
        calibration_parameters = json.loads(result.data)
        assert {row["name"] for row in calibration_parameters} == {"file1", "file2"}
        assert all("Position / contrasts" in row for row in calibration_parameters)

        with pytest.raises(ToolError, match="Length of known_standards must match number of models"):
            await client.call_tool("calibrate", {"known_standards": [480]})

        photomol_tools.MP_CALIBRATOR.known_standards = [66, 148, 480]
        photomol_tools.MP_CALIBRATOR.calibration_dic = {
            "exp_points": [-0.0035, -0.0086, -0.0288],
            "fit_params": [-0.00005, 0.0],
            "fit_r2": 0.99,
        }
        result = await client.call_tool("plot_calibration", {"save_as_html": True})
        assert result.data == "Calibration plot created successfully, but no valid image format was specified for saving."
        assert any(path.name.startswith("calibration_") for path in log_dir.glob("*.html"))


@pytest.mark.asyncio
async def test_mcp_results_match_simple_example_notebook(isolated_tool_log_dir):
    """
    Compare fitted mass-photometry values against the simple example notebook.

    This test uses the real ``demo.h5`` fixture and real fitting code, then
    checks fitted positions, widths, counts, percentages, and amplitudes against
    known expected values.
    """
    async with Client(mcp_pyphotomol.mcp) as client:
        await client.call_tool("reset_analyzer", {})
        await client.call_tool(
            "import_single_file",
            {"file_path": str(EXAMPLE_DATA_DIR / NOTEBOOK_DEMO_FILE), "name": "demo1"},
        )
        await client.call_tool(
            "create_histogram_manual",
            {"min_value": 0, "max_value": 800, "bin_width": 10},
        )
        await client.call_tool(
            "fit_multi_gaussian",
            {"peaks_guess": [65, 145, 465], "threshold": 40, "fit_baseline": False},
        )

        result = await client.call_tool("show_fitted_parameters", {})
        fit_table = json.loads(result.data)

    expected_rows = [
        {
            "Position / kDa": 65.272253,
            "Sigma / kDa": 15.861073,
            "Counts": 870.825870,
            "Counts / %": 61.0,
            "Amplitudes": 232.080891,
        },
        {
            "Position / kDa": 145.751043,
            "Sigma / kDa": 20.221630,
            "Counts": 293.735298,
            "Counts / %": 21.0,
            "Amplitudes": 57.949551,
        },
        {
            "Position / kDa": 480.554337,
            "Sigma / kDa": 29.687422,
            "Counts": 171.772627,
            "Counts / %": 12.0,
            "Amplitudes": 23.082962,
        },
    ]

    assert len(fit_table) == len(expected_rows)
    for actual, expected in zip(fit_table, expected_rows, strict=True):
        assert actual["name"] == "demo1"
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, rel=1e-6)


@pytest.mark.asyncio
async def test_mcp_results_match_simple_calibration_notebook(isolated_tool_log_dir):
    """
    Compare calibration fitting output against the simple calibration notebook.

    This test uses the real contrast CSV fixture, fits the calibration peaks,
    runs ``calibrate``, and checks the resulting calibration parameters and R²
    against known expected values.
    """
    async with Client(mcp_pyphotomol.mcp) as client:
        await client.call_tool("reset_calibrator", {})
        await client.call_tool(
            "import_single_file",
            {
                "file_path": str(EXAMPLE_DATA_DIR / CONTRAST_EXAMPLE_FILE),
                "name": "notebook_contrasts",
                "calibrator": True,
            },
        )
        await client.call_tool(
            "create_histogram_manual",
            {
                "min_value": -0.04,
                "max_value": 0,
                "bin_width": 0.0004,
                "use_masses": False,
                "calibrator": True,
            },
        )
        await client.call_tool(
            "fit_multi_gaussian",
            {
                "calibrator": True,
                "peaks_guess": [-0.03, -0.01, -0.005],
                "mean_tolerance": 0.1,
                "std_tolerance": 0.1,
                "threshold": -0.0022,
                "baseline": 0,
            },
        )
        result = await client.call_tool("calibrate", {"known_standards": [480, 146, 66]})

    assert "Calibration results" in result.data
    assert photomol_tools.MP_CALIBRATOR.calibration_dic["fit_params"][0] == pytest.approx(
        -6.115911272669366e-05,
        rel=1e-8,
    )
    assert photomol_tools.MP_CALIBRATOR.calibration_dic["fit_params"][1] == pytest.approx(
        0.0004374498828378568,
        rel=1e-8,
    )
    assert photomol_tools.MP_CALIBRATOR.calibration_dic["fit_r2"] == pytest.approx(
        0.9999993694482743,
        rel=1e-8,
    )


@pytest.mark.asyncio
async def test_mcp_logbook_resource(resource_data_root):
    """
    Verify the logbook MCP resource resolves valid, empty, and malformed dates.

    The fixture redirects the resource data root so the test controls the
    available logbook files without reading user data from the real workspace.
    """
    date_dir = resource_data_root / "2026-01-02"
    date_dir.mkdir()
    (date_dir / "mcp_logbook.json").write_text(json.dumps({"calls": [{"tool": "load_example_data"}]}))
    empty_date_dir = resource_data_root / "2026-01-03"
    empty_date_dir.mkdir()

    async with Client(mcp_pyphotomol.mcp) as client:
        result = await client.read_resource("data://02-01-2026/logbook")
        assert json.loads(result[0].text) == {"calls": [{"tool": "load_example_data"}]}

        result = await client.read_resource("data://2026-01-03/logbook")
        assert json.loads(result[0].text) == {"error": "No logbook found for the specified date."}

        result = await client.read_resource("data://not-a-date/logbook")
        assert json.loads(result[0].text) == {"error": "No logbook found for the specified date."}


def test_cli_version_and_transports():
    """
    Verify CLI version output and transport dispatch.

    The MCP server runners are patched so the test can assert which transport
    branch is selected without starting long-lived stdio or HTTP servers.
    """
    runner = CliRunner()
    result = runner.invoke(run_app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == mcp_pyphotomol.__version__

    calls = []
    mcp_module = importlib.import_module("mcp_pyphotomol.mcp")
    stdio_module = importlib.import_module("mcp_pyphotomol.stdio")

    with (
        patch.object(mcp_module.mcp, "run", lambda **kwargs: calls.append(kwargs)),
        patch.object(
            stdio_module,
            "run_stdio",
            lambda server: calls.append({"transport": "stdio"}),
        ),
    ):
        result = runner.invoke(run_app, [])
        assert result.exit_code == 0
        assert calls[-1] == {"transport": "stdio"}
        assert f"mcp_pyphotomol results folder: {photomol_server.DATA_DIR}" in result.stderr

        result = runner.invoke(run_app, ["--transport", "http", "--port", "9999", "--host", "127.0.0.1"])
        assert result.exit_code == 0
        assert calls[-1] == {"transport": "http", "port": 9999, "host": "127.0.0.1"}
        assert f"mcp_pyphotomol results folder: {photomol_server.DATA_DIR}" in result.stderr

        result = runner.invoke(run_app, ["--transport", "sse"])
        assert result.exit_code == 0
        assert calls[-1] == {"transport": "sse"}
        assert f"mcp_pyphotomol results folder: {photomol_server.DATA_DIR}" in result.stderr

        result = runner.invoke(run_app, ["--env", "production"])
        assert result.exit_code == 1
        assert isinstance(result.exception, NotImplementedError)


def test_module_entrypoints_print_version(capsys):
    """Verify package and module ``__main__`` entrypoints print the version."""
    with patch.object(sys, "argv", ["mcp_pyphotomol", "--version"]):
        with pytest.raises(SystemExit) as package_exit:
            runpy.run_path(Path(mcp_pyphotomol.__file__), run_name="__main__")
    assert package_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mcp_pyphotomol.__version__

    with patch.object(sys, "argv", ["mcp_pyphotomol.main", "--version"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(SystemExit) as main_exit:
                runpy.run_module("mcp_pyphotomol.main", run_name="__main__")
    assert main_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mcp_pyphotomol.__version__


def test_server_initializes_user_data_when_not_skipped():
    """
    Verify server import initializes user-data folders and logbook contents.

    Filesystem calls are patched so the initialization branches run without
    creating directories or writing files outside the in-memory test buffer.
    """
    created_dirs = []

    class NoCloseStringIO(io.StringIO):
        def close(self):
            pass

    written = NoCloseStringIO()
    original_exists = os.path.exists
    original_open = open

    def fake_exists(path):
        if "user_data" in str(path):
            return False
        return original_exists(path)

    def fake_makedirs(path):
        created_dirs.append(str(path))

    def fake_open(path, mode="r", *args, **kwargs):
        if str(path).endswith("mcp_logbook.txt") and "w" in mode:
            return written
        return original_open(path, mode, *args, **kwargs)

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MCP_PYPHOTOMOL_SKIP_USER_DATA_INIT", None)
        with (
            patch.object(os.path, "exists", fake_exists),
            patch.object(os, "makedirs", fake_makedirs),
            patch.object(builtins, "open", fake_open),
        ):
            namespace = runpy.run_path(Path(photomol_server.__file__))

    assert any(path.endswith(USER_DATA_DIR_NAME) for path in created_dirs)
    assert any(path.endswith(namespace["today"]) for path in created_dirs)
    assert "MCP Logbook" in written.getvalue()
    assert f"Date: {namespace['today']}" in written.getvalue()


def test_tool_failure_and_error_branches(isolated_tool_log_dir, tmp_path):
    """
    Verify selected tool failure paths and validation errors.

    The analyzer import methods and model dictionaries are patched to trigger
    failure states directly, avoiding unnecessary fixture data processing while
    still exercising the real tool functions.
    """
    with (
        patch.object(photomol_tools.MP_ANALYZER, "models", {}),
        patch.object(photomol_tools.MP_ANALYZER, "import_files", lambda *args, **kwargs: None),
    ):
        assert (
            photomol_tools.import_single_file(EXAMPLE_DATA_DIR / "masses_monomer_1nM.csv")
            == f"Data import failed for {EXAMPLE_DATA_DIR / 'masses_monomer_1nM.csv'}."
        )

        import_dir = tmp_path / "import"
        import_dir.mkdir()
        (import_dir / "measurement.csv").write_text("masses_kDa\n80\n160\n")
        assert photomol_tools.import_folder(import_dir) == f"Data import failed for {import_dir}."

    with patch.object(
        photomol_tools.MP_ANALYZER,
        "models",
        {"missing_masses": SimpleNamespace(masses=None, contrasts=None)},
    ):
        with pytest.raises(ValueError, match="Mass data is missing for model missing_masses"):
            photomol_tools.create_histogram_automatic()

    with patch.object(
        photomol_tools.MP_ANALYZER,
        "models",
        {"missing_contrasts": SimpleNamespace(masses=None, contrasts=None)},
    ):
        with pytest.raises(ValueError, match="Contrast data is missing for model missing_contrasts"):
            photomol_tools.create_histogram_automatic(use_masses=False)


def test_guess_peaks_branches(isolated_tool_log_dir):
    """
    Verify peak guessing success, experiment filtering, and missing histogram errors.

    This uses real bundled mass CSV fixtures and the real pyphotomol analyzer so
    the peak values come from actual histogram data. The assertions focus on
    experiment filtering and the missing-histogram guard.
    """
    photomol_tools.reset_analyzer()
    photomol_tools.import_single_file(EXAMPLE_DATA_DIR / "masses_monomer_1nM.csv", name="skipped")
    photomol_tools.import_single_file(EXAMPLE_DATA_DIR / "masses_monomer_2nM.csv", name="selected")
    photomol_tools.create_histogram_manual(min_value=0, max_value=300, bin_width=8)

    result = json.loads(
        photomol_tools.guess_peaks(
            min_height=2,
            min_distance=3,
            prominence=4,
            experiment="selected",
        )
    )

    assert set(result) == {"selected"}
    assert all(isinstance(value, float) for value in result["selected"])
    assert photomol_tools.MP_ANALYZER.models["skipped"].peaks_guess is None
    assert photomol_tools.MP_ANALYZER.models["selected"].peaks_guess is not None

    photomol_tools.reset_analyzer()
    photomol_tools.import_single_file(EXAMPLE_DATA_DIR / "masses_monomer_1nM.csv", name="missing")
    with pytest.raises(ValueError, match="Histogram for model missing has not been created"):
        photomol_tools.guess_peaks()


def test_fit_multi_gaussian_error_and_dict_branches(isolated_tool_log_dir):
    """
    Verify fit error handling and dictionary-based peak guesses.

    This uses the real ``demo.h5`` fixture for dictionary peak guesses. A single
    real model method is monkeypatched to simulate the rare case where automatic
    peak guessing leaves no usable peaks; numerical fit correctness is covered
    by the notebook comparison tests.
    """
    photomol_tools.reset_analyzer()
    photomol_tools.import_single_file(EXAMPLE_DATA_DIR / NOTEBOOK_DEMO_FILE, name="sample")
    photomol_tools.create_histogram_manual(min_value=0, max_value=800, bin_width=10)
    model = photomol_tools.MP_ANALYZER.models["sample"]
    model.peaks_guess = None
    with patch.object(model, "guess_peaks", lambda **kwargs: None):
        with pytest.raises(ValueError, match="No peaks available for model sample"):
            photomol_tools.fit_multi_gaussian()

    with pytest.raises(ValueError, match="No peaks provided for experiment 'sample'"):
        photomol_tools.fit_multi_gaussian(peaks_guess={"other": [65, 145, 465]})

    result = photomol_tools.fit_multi_gaussian(
        peaks_guess={"sample": [65, 145, 465]},
        threshold=40,
        fit_baseline=False,
    )

    assert result == "Multi-gaussian fitting completed successfully."
    assert len(photomol_tools.MP_ANALYZER.models["sample"].fit_table) == 3


def test_auto_histogram_large_mass_bins(isolated_tool_log_dir, tmp_path):
    """
    Verify automatic histogram bin-width choices for larger mass ranges.

    Temporary CSV files provide controlled mass ranges while still going through
    real import and histogram creation code.
    """
    medium = tmp_path / "medium.csv"
    medium.write_text("masses_kDa\n0\n600\n")
    photomol_tools.reset_analyzer()
    photomol_tools.import_single_file(medium, name="medium")
    result = photomol_tools.create_histogram_automatic()
    assert "Bin width: 10" in result

    large = tmp_path / "large.csv"
    large.write_text("masses_kDa\n0\n1500\n")
    photomol_tools.reset_analyzer()
    photomol_tools.import_single_file(large, name="large")
    result = photomol_tools.create_histogram_automatic()
    assert "Bin width: 12" in result


def test_plot_image_export_branches(isolated_tool_log_dir):
    """
    Verify plot export branches write HTML and image outputs.

    ``FakeFigure`` replaces Plotly figures so the test can exercise the
    repository's save-path logic without invoking Kaleido or browser rendering.
    """
    class FakeFigure:
        """Minimal figure stand-in that records HTML and image export paths."""

        def __init__(self):
            self.html_paths = []
            self.image_paths = []

        def write_html(self, path):
            self.html_paths.append(path)
            Path(path).write_text("<html></html>")

        def write_image(self, path, **kwargs):
            self.image_paths.append((path, kwargs))
            Path(path).write_text("<svg></svg>")

    histogram_fig = FakeFigure()
    fits_fig = FakeFigure()
    calibration_fig = FakeFigure()

    with (
        patch.multiple(photomol_tools.PLOT_CONFIG, plot_type="svg", plot_width=640, plot_height=480),
        patch.object(photomol_tools.MP_ANALYZER, "models", {"sample": object()}),
        patch.multiple(
            photomol_tools,
            pm_plot_histogram=lambda *args, **kwargs: histogram_fig,
            pm_plot_histograms_and_fits=lambda *args, **kwargs: fits_fig,
            pm_plot_calibration=lambda *args, **kwargs: calibration_fig,
        ),
    ):
        result = photomol_tools.plot_histograms(colors_hist="red", save_as_html=True)
        assert result.startswith("Histogram plot saved as svg at ")
        assert histogram_fig.html_paths
        assert histogram_fig.image_paths

        result = photomol_tools.plot_histograms_and_fits(colors_hist="blue", save_as_html=True)
        assert result.startswith("Histograms and fits plot saved as svg at ")
        assert fits_fig.html_paths
        assert fits_fig.image_paths

        photomol_tools.MP_CALIBRATOR.known_standards = [66, 148, 480]
        photomol_tools.MP_CALIBRATOR.calibration_dic = {
            "exp_points": [-0.0035, -0.0086, -0.0288],
            "fit_params": [-0.00005, 0.0],
            "fit_r2": 0.99,
        }
        result = photomol_tools.plot_calibration(save_as_html=True)
        assert result.startswith("Calibration plot saved as svg at ")
        assert calibration_fig.html_paths
        assert calibration_fig.image_paths
