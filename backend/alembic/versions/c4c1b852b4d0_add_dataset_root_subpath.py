"""add dataset root subpath

Revision ID: c4c1b852b4d0
Revises: 7b2e25d3f2bb
Create Date: 2026-09-22 15:30:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c4c1b852b4d0"
down_revision: Union[str, None] = "7b2e25d3f2bb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "dataset_versions",
        sa.Column("root_subpath", sa.String(length=1000), nullable=False, server_default="."),
    )


def downgrade() -> None:
    op.drop_column("dataset_versions", "root_subpath")
