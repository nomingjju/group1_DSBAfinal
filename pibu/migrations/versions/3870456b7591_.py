"""empty message

Revision ID: 3870456b7591
Revises: 8c0e92410a8e
Create Date: 2023-11-16 10:08:41.076702

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '3870456b7591'
down_revision = '8c0e92410a8e'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('chart',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('copy__member__profile',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('member__profile',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('pw', sa.Text(), nullable=False),
    sa.Column('gender', sa.Integer(), nullable=False),
    sa.Column('age', sa.Integer(), nullable=False),
    sa.Column('agree', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('skin__cancer',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('social__login',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('uv__page',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('uv__page')
    op.drop_table('social__login')
    op.drop_table('skin__cancer')
    op.drop_table('member__profile')
    op.drop_table('copy__member__profile')
    op.drop_table('chart')
    # ### end Alembic commands ###
