"""Neo4j client utilities (graph grounding)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from .config import Settings, load_settings
from .logging import get_logger


LOGGER = get_logger(__name__)


@dataclass
class MyNeo4j:
    """Thin wrapper around a Neo4j driver with a default database."""

    driver: Any
    settings: Settings
    database: str

    def close(self) -> None:
        try:
            self.driver.close()
        except Exception:
            return

    def run(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        readonly: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run a Cypher query and return records as plain dicts."""

        params = params or {}
        try:
            with self.driver.session(database=self.database) as session:
                res = session.run(cypher, params)
                return [r.data() for r in res]
        except Neo4jError as exc:
            LOGGER.warning(
                "Neo4j query failed (db=%s): %s",
                self.database,
                getattr(exc, "message", str(exc)),
            )
            raise


def _auth_tuple(user: str, password: str) -> Optional[Tuple[str, str]]:
    user = (user or "").strip()
    if not user:
        return None
    return (user, password or "")


def create_graph_client(settings: Optional[Settings] = None) -> MyNeo4j:
    """Create a Neo4j driver for the configured graph database."""

    if settings is None:
        settings = load_settings()

    LOGGER.info(
        "Connecting to Neo4j at %s (db=%s)",
        settings.neo4j_uri,
        settings.neo4j_database,
    )
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=_auth_tuple(settings.neo4j_user, settings.neo4j_password),
    )
    return MyNeo4j(driver=driver, settings=settings, database=settings.neo4j_database)


__all__ = [
    "MyNeo4j",
    "create_graph_client",
]
