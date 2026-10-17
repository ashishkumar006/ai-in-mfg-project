"""SQLite-backed repository for the Sandvik-style tooling catalog.

The schema mirrors ISO 13399-style attributes so the same code can
later load the real catalog by parsing the vendor export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, declarative_base, mapped_column, Mapped
from sqlalchemy.types import Float, Integer, String

from hybrid_ai.core.schemas import ISOMaterialGroup, Operation, ToolCandidate

Base = declarative_base()


class ToolRow(Base):
    __tablename__ = "tools"

    insert_code: Mapped[str] = mapped_column(String(32), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, default=0, index=True)
    iso_group: Mapped[str] = mapped_column(String(2), index=True)
    operation: Mapped[str] = mapped_column(String(16), index=True)
    nose_radius_mm: Mapped[float] = mapped_column(Float)
    rake_angle_deg: Mapped[float] = mapped_column(Float)
    v_c_min_mpm: Mapped[float] = mapped_column(Float)
    v_c_max_mpm: Mapped[float] = mapped_column(Float)
    f_min_mmpr: Mapped[float] = mapped_column(Float)
    f_max_mmpr: Mapped[float] = mapped_column(Float)
    ap_max_mm: Mapped[float] = mapped_column(Float)
    description: Mapped[str] = mapped_column(String(255), default="")


class ToolCatalogRepository:
    """Thin wrapper around an SQLite-backed tool catalog."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        Base.metadata.create_all(self.engine)

    def _write(
        self,
        tools: Iterable[ToolCandidate],
        *,
        merge: bool,
    ) -> None:
        with Session(self.engine) as session:
            if not merge:
                session.query(ToolRow).delete(synchronize_session=False)
            for i, t in enumerate(tools):
                row = ToolRow(
                    insert_code=t.insert_code,
                    position=i,
                    iso_group=t.iso_group,
                    operation=t.operation,
                    nose_radius_mm=t.nose_radius_mm,
                    rake_angle_deg=t.rake_angle_deg,
                    v_c_min_mpm=t.v_c_min_mpm,
                    v_c_max_mpm=t.v_c_max_mpm,
                    f_min_mmpr=t.f_min_mmpr,
                    f_max_mmpr=t.f_max_mmpr,
                    ap_max_mm=t.ap_max_mm,
                    description=t.description,
                )
                if merge:
                    session.merge(row)
                else:
                    session.add(row)
            session.commit()

    def upsert(self, tools: Iterable[ToolCandidate]) -> None:
        self._write(tools, merge=True)

    def replace(self, tools: Iterable[ToolCandidate]) -> None:
        """Rewrite the whole table, preserving the given row order.

        Selection behaves slightly differently depending on candidate
        order (CLIPS fires in catalog order), so a refresh from the
        catalog file must keep positions aligned with the CSV.
        """
        self._write(tools, merge=False)

    def all(self) -> list[ToolCandidate]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(ToolRow).order_by(ToolRow.position)).scalars().all()
            return [_row_to_candidate(r) for r in rows]

    def filter(
        self,
        *,
        iso_group: ISOMaterialGroup | None = None,
        operation: Operation | None = None,
    ) -> list[ToolCandidate]:
        with Session(self.engine) as session:
            stmt = select(ToolRow)
            if iso_group is not None:
                stmt = stmt.where(ToolRow.iso_group == iso_group)
            if operation is not None:
                stmt = stmt.where(ToolRow.operation == operation)
            rows = session.execute(stmt).scalars().all()
            return [_row_to_candidate(r) for r in rows]


def _row_to_candidate(row: ToolRow) -> ToolCandidate:
    return ToolCandidate(
        insert_code=row.insert_code,
        iso_group=row.iso_group,
        operation=row.operation,
        nose_radius_mm=row.nose_radius_mm,
        rake_angle_deg=row.rake_angle_deg,
        v_c_min_mpm=row.v_c_min_mpm,
        v_c_max_mpm=row.v_c_max_mpm,
        f_min_mmpr=row.f_min_mmpr,
        f_max_mmpr=row.f_max_mmpr,
        ap_max_mm=row.ap_max_mm,
        description=row.description,
    )
