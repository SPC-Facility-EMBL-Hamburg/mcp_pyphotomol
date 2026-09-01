import importlib
import json
import os
import runpy
import sys
import warnings
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

import mcp_pyphotomol
from mcp_pyphotomol.paths import (
    RESULTS_DIR_ENV_VAR,
    USER_DATA_DIR_NAME,
    get_example_data_root,
    get_user_data_root,
)
import mcp_pyphotomol.config as photomol_config
import mcp_pyphotomol.server as photomol_server
import mcp_pyphotomol.tools._photomol as photomol_tools
from mcp_pyphotomol.main import run_app
from mcp_pyphotomol.mcp import mcp

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
    instructions = photomol_config.build_server_instructions(photomol_server.DATA_DIR)

    assert photomol_server.DATA_DIR in instructions
    assert "Plots and log files for this session are saved in:" in instructions


def test_example_data_files_are_present():
    """Verify the bundled CSV and notebook fixtures needed by integration tests exist."""
    files = {path.name for path in EXAMPLE_DATA_DIR.glob("*.csv")}
    assert MASS_EXAMPLE_FILES | {CONTRAST_EXAMPLE_FILE} <= files
    assert (EXAMPLE_DATA_DIR / NOTEBOOK_DEMO_FILE).is_file()



def _result_data(result):
    """Extract Python values from an MCP v2 CallToolResult."""
    structured = getattr(result, "structured_content", None)

    if structured is not None:
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]

        return structured

    content = getattr(result, "content", None) or []

    values = []

    for item in content:
        if not hasattr(item, "text"):
            values.append(item)
            continue

        value = item.text

        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass

        values.append(value)

    if len(values) == 1:
        return values[0]

    return values


def _result_error_text(result) -> str:
    """Return the textual error payload from an MCP v2 tool result."""
    content = getattr(result, "content", None) or []
    return "\n".join(
        item.text for item in content if hasattr(item, "text")
    )


@pytest.mark.asyncio
async def test_mcp_server_tools_with_example_data(isolated_tool_log_dir):
    """
    Exercise the public MCP tools against bundled example data.

    This is a broad integration smoke test: it checks tool registration, import
    workflows, histogram/fitting calls, plotting branches, calibration setup,
    and that expected files are written to the isolated log directory.
    """
    log_dir = isolated_tool_log_dir

    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools.tools}

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
        assert isinstance(_result_data(result), str)
        assert _result_data(result)

        result = await client.call_tool(
            "list_MP_files_in_folder",
            {"folder_path": str(EXAMPLE_DATA_DIR)},
        )
        listed_files = {file_name.strip() for file_name in _result_data(result).split(",")}
        assert MASS_EXAMPLE_FILES | {CONTRAST_EXAMPLE_FILE, NOTEBOOK_DEMO_FILE} <= listed_files

        result = await client.call_tool(
            "import_folder",
            {"folder_path": str(EXAMPLE_DATA_DIR), "pattern": "does-not-exist"},
        )
        assert _result_data(result) == f"No files found in {EXAMPLE_DATA_DIR}."

        await client.call_tool("reset_analyzer", {})
        result = await client.call_tool(
            "import_single_file",
            {"file_path": str(EXAMPLE_DATA_DIR / "masses_monomer_1nM.csv")},
        )
        assert _result_data(result) == f"Data imported successfully from {EXAMPLE_DATA_DIR / 'masses_monomer_1nM.csv'}."

        result = await client.call_tool("get_model_names", {})
        print("RESULT:", repr(result))
        print("STRUCTURED:", repr(result.structured_content))
        print("CONTENT:", repr(result.content))
        print("EXTRACTED:", repr(_result_data(result)))

        assert _result_data(result) == ["masses_monomer_1nM"]

        result = await client.call_tool("fit_multi_gaussian", {})
        assert result.is_error
        # now package mcp returns 'Error executing tool...'
        # assert "Histogram for model masses_monomer_1nM has not been created" in _result_error_text(result)

        result = await client.call_tool(
            "create_histogram_manual",
            {"min_value": 0, "max_value": 300, "bin_width": 8},
        )
        assert "Histograms were created successfully" in _result_data(result)
        assert "Bin width: 8" in _result_data(result)

        result = await client.call_tool(
            "fit_multi_gaussian",
            {"peaks_guess": [80, 160], "mean_tolerance": 80, "std_tolerance": 80},
        )
        assert _result_data(result) == "Multi-gaussian fitting completed successfully."

        await client.call_tool("reset_analyzer", {})
        result = await client.call_tool(
            "import_folder",
            {"folder_path": str(EXAMPLE_DATA_DIR), "pattern": "masses_monomer"},
        )
        assert _result_data(result) == f"{len(MASS_EXAMPLE_FILES)} files were imported successfully from {EXAMPLE_DATA_DIR}."

        result = await client.call_tool("get_model_names", {})
        assert {f.removesuffix(".csv") for f in MASS_EXAMPLE_FILES} == set(_result_data(result))

        await client.call_tool("reset_analyzer", {})
        result = await client.call_tool("load_example_data", {})
        assert _result_data(result) == "Example data loaded successfully."

        result = await client.call_tool("get_model_names", {})
        assert _result_data(result) == EXPECTED_EXAMPLE_MODEL_NAMES

        result = await client.call_tool("create_histogram_automatic", {})
        assert "Histograms were created successfully" in _result_data(result)
        assert "Using masses: True" in _result_data(result)

        result = await client.call_tool("fit_multi_gaussian", {"experiment": "not-present"})
        assert _result_data(result) == "Multi-gaussian fitting completed successfully."

        result = await client.call_tool("fit_multi_gaussian", {})
        assert _result_data(result) == "Multi-gaussian fitting completed successfully."

        result = await client.call_tool("show_fitted_parameters", {})
        fitted_parameters = json.loads(_result_data(result))
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
        assert _result_data(result) == "Plot configuration updated successfully."

        result = await client.call_tool(
            "update_legend_config",
            {"add_percentage_to_legend": True, "line_width": 2},
        )
        assert _result_data(result) == "Legend configuration updated successfully."

        result = await client.call_tool(
            "update_layout_config",
            {"stacked": False, "show_subplot_titles": True},
        )
        assert _result_data(result) == "Layout configuration updated successfully."

        result = await client.call_tool("update_axis_config", {"n_y_axis_ticks": 5})
        assert _result_data(result) == "Axis configuration updated successfully."

        result = await client.call_tool(
            "plot_histograms",
            {"colors_hist": "red", "save_as_html": True},
        )
        assert _result_data(result) == "Histogram plot created successfully, but no valid image format was specified for saving."

        result = await client.call_tool(
            "get_legends_dataframe",
            {"repeat_colors": False},
        )

        legends = _result_data(result)

        if isinstance(legends, str):
            legends = json.loads(legends)

        assert {
            column
            for row in legends
            for column in row
        } >= {
            "legends",
            "color",
            "select",
            "show_legend",
        }

        result = await client.call_tool(
            "plot_histograms_and_fits",
            {"legends_df": json.dumps(legends), "colors_hist": "blue", "save_as_html": True},
        )

        assert not result.is_error, _result_error_text(result)


        assert _result_data(result) == "Histograms and fits plot created successfully, but no valid image format was specified for saving."
        assert any(path.name.startswith("histogram_") for path in log_dir.glob("*.html"))
        assert any(path.name.startswith("histograms_and_fits_") for path in log_dir.glob("*.html"))

        await client.call_tool("reset_calibrator", {})
        result = await client.call_tool("load_example_data_for_calibration", {})
        assert _result_data(result) == "Example calibration data loaded successfully."

        result = await client.call_tool("get_model_names", {"calibrator": True})
        assert _result_data(result) == ["file1", "file2"]

        result = await client.call_tool(
            "create_histogram_automatic",
            {"use_masses": False, "calibrator": True},
        )
        assert "Using contrasts: True" in _result_data(result)

        result = await client.call_tool("fit_multi_gaussian", {"calibrator": True})
        assert _result_data(result) == "Multi-gaussian fitting completed successfully."

        result = await client.call_tool("show_fitted_parameters", {"calibrator": True})
        calibration_parameters = json.loads(_result_data(result))
        assert {row["name"] for row in calibration_parameters} == {"file1", "file2"}
        assert all("Position / contrasts" in row for row in calibration_parameters)

        result = await client.call_tool("calibrate", {"known_standards": [480]})
        assert result.is_error
        assert "Length of known_standards must match number of models" in _result_error_text(result)

        photomol_tools.MP_CALIBRATOR.known_standards = [66, 148, 480]
        photomol_tools.MP_CALIBRATOR.calibration_dic = {
            "exp_points": [-0.0035, -0.0086, -0.0288],
            "fit_params": [-0.00005, 0.0],
            "fit_r2": 0.99,
        }
        result = await client.call_tool("plot_calibration", {"save_as_html": True})
        assert _result_data(result) == "Calibration plot created successfully, but no valid image format was specified for saving."
        assert any(path.name.startswith("calibration_") for path in log_dir.glob("*.html"))


@pytest.mark.asyncio
async def test_mcp_results_match_simple_example_notebook(isolated_tool_log_dir):
    """
    Compare fitted mass-photometry values against the simple example notebook.

    This test uses the real ``demo.h5`` fixture and real fitting code, then
    checks fitted positions, widths, counts, percentages, and amplitudes against
    known expected values.
    """
    async with Client(mcp) as client:
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
        fit_table = json.loads(_result_data(result))

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
    async with Client(mcp) as client:
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

    assert "Calibration results" in _result_data(result)
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
    """
    date_dir = resource_data_root / "2026-01-02"
    date_dir.mkdir()
    (date_dir / "mcp_logbook.json").write_text(
        json.dumps(
            {
                "calls": [
                    {"tool": "load_example_data"},
                ]
            }
        )
    )

    empty_date_dir = resource_data_root / "2026-01-03"
    empty_date_dir.mkdir()

    async with Client(mcp) as client:
        result = await client.read_resource(
            "data://02-01-2026/logbook"
        )
        assert json.loads(result.contents[0].text) == {
            "calls": [
                {"tool": "load_example_data"},
            ]
        }

        result = await client.read_resource(
            "data://2026-01-03/logbook"
        )
        assert json.loads(result.contents[0].text) == {
            "error": "No logbook found for the specified date."
        }

        result = await client.read_resource(
            "data://not-a-date/logbook"
        )
        assert json.loads(result.contents[0].text) == {
            "error": "No logbook found for the specified date."
        }


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

    with patch.object(mcp_module.mcp, "run", lambda **kwargs: calls.append(kwargs)):
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


def test_config_initializes_user_data_when_not_skipped(monkeypatch, tmp_path):
    """Verify config initialization creates the user-data folders and logbook."""
    results_dir = tmp_path / "user_data"
    monkeypatch.setenv(RESULTS_DIR_ENV_VAR, str(results_dir))
    monkeypatch.delenv("MCP_PYPHOTOMOL_SKIP_USER_DATA_INIT", raising=False)

    reloaded_config = importlib.reload(photomol_config)

    assert results_dir.is_dir()
    assert Path(reloaded_config.DATA_DIR).is_dir()
    assert Path(reloaded_config.DATA_DIR).parent == results_dir
    assert Path(reloaded_config.LOGBOOK_FILE).is_file()

    logbook = Path(reloaded_config.LOGBOOK_FILE).read_text()
    assert "MCP Logbook" in logbook
    assert f"Date: {reloaded_config.today}" in logbook

    # Restore the module without initializing the normal user-data directory.
    monkeypatch.setenv("MCP_PYPHOTOMOL_SKIP_USER_DATA_INIT", "1")
    monkeypatch.delenv(RESULTS_DIR_ENV_VAR, raising=False)
    importlib.reload(photomol_config)


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
        with pytest.raises(ToolError, match="Mass data is missing for model missing_masses"):
            photomol_tools.create_histogram_automatic()

    with patch.object(
        photomol_tools.MP_ANALYZER,
        "models",
        {"missing_contrasts": SimpleNamespace(masses=None, contrasts=None)},
    ):
        with pytest.raises(ToolError, match="Contrast data is missing for model missing_contrasts"):
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
    with pytest.raises(ToolError, match="Histogram for model missing has not been created"):
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
        with pytest.raises(ToolError, match="No peaks available for model sample"):
            photomol_tools.fit_multi_gaussian()

    with pytest.raises(ToolError, match="No peaks provided for experiment 'sample'"):
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
