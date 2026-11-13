"""Persist project types for NB conversion.

Revision ID: e2d834c6af91
Revises: c4c1b852b4d0
"""

import sqlalchemy as sa

from alembic import op
from app.config import settings

revision = "e2d834c6af91"
down_revision = "c4c1b852b4d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "project_type",
            sa.Enum("GENERAL", "VIBRATION_FEEDER_POINT", name="projecttype", native_enum=False),
            nullable=False,
            server_default="GENERAL",
        ),
    )
    # Preserve the conversion access already granted by the deployment.
    legacy_projects = sorted(
        {item.strip() for item in settings.nb_conversion_projects.split(",")} - {""}
    )
    if legacy_projects:
        projects = sa.table(
            "projects", sa.column("id"), sa.column("name"), sa.column("project_type")
        )
        op.execute(
            projects.update()
            .where(sa.or_(projects.c.id.in_(legacy_projects), projects.c.name.in_(legacy_projects)))
            .values(project_type="VIBRATION_FEEDER_POINT")
        )


def downgrade() -> None:
    op.drop_column("projects", "project_type")
