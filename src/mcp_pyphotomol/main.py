import enum
import logging
import sys

import click

from .resources import *  # noqa: F403 import all resources to register them
from .tools import *  # noqa: F403 import all tools to register them


class EnvironmentType(enum.Enum):
    """Enum to define environment type."""

    PRODUCTION = enum.auto()
    DEVELOPMENT = enum.auto()


def run_development_transport(
    mcp,
    transport: str,
    port: int,
    hostname: str,
) -> None:
    """Run the configured development transport."""
    if transport == "http":
        mcp.run(
            transport="http",
            port=port,
            host=hostname,
        )
    else:
        mcp.run(transport=transport)


@click.command(name="run")
@click.option(
    "-t",
    "--transport",
    "transport",
    type=str,
    help="MCP transport option. Defaults to 'stdio'.",
    default="stdio",
    envvar="MCP_TRANSPORT",
)
@click.option(
    "-p",
    "--port",
    "port",
    type=int,
    help="Port of MCP server. Defaults to '8000'.",
    default=8000,
    envvar="MCP_PORT",
    required=False,
)
@click.option(
    "-h",
    "--host",
    "hostname",
    type=str,
    help="Hostname of MCP server. Defaults to '0.0.0.0'.",
    default="0.0.0.0",
    envvar="MCP_HOSTNAME",
    required=False,
)
@click.option(
    "-v",
    "--version",
    "version",
    is_flag=True,
    help="Get version of package.",
)
@click.option(
    "-e",
    "--env",
    "environment",
    type=click.Choice(EnvironmentType, case_sensitive=False),
    default=EnvironmentType.DEVELOPMENT,
    envvar="MCP_ENVIRONMENT",
    help="MCP server environment. Defaults to 'development'.",
)
def run_app(
    transport: str = "stdio",
    port: int = 8000,
    hostname: str = "0.0.0.0",
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT,
    version: bool = False,
):
    """
    Run the MCP server "mcp_pyphotomol".

    Analysis of mass photometry data.

    The transport defaults to stdio. HTTP transports use the configured
    hostname and port.
    """
    if version:
        from mcp_pyphotomol import __version__

        click.echo(__version__)
        sys.exit(0)

    logger = logging.getLogger(__name__)

    from mcp_pyphotomol.mcp import mcp
    from mcp_pyphotomol.server import DATA_DIR

    if environment == EnvironmentType.DEVELOPMENT:
        logger.info("Starting MCP server (DEVELOPMENT mode)")

        click.echo(
            f"mcp_pyphotomol results folder: {DATA_DIR}",
            err=True,
        )

        run_development_transport(
            mcp,
            transport,
            port,
            hostname,
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    run_app()