from typing import Optional

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.deps import CurrentUser, DbSession, get_project_role
from app.models import AuditLog, Project, ProjectMember, ProjectRole, SystemRole, User
from app.schemas import ProjectCreate, ProjectMemberAdd, ProjectMemberView, ProjectView

router = APIRouter(prefix="/projects", tags=["项目与成员"])


def to_project_view(project: Project, current_user: User) -> ProjectView:
    current_membership = next((m for m in project.members if m.user_id == current_user.id), None)
    return ProjectView(
        id=project.id,
        name=project.name,
        description=project.description,
        created_by=project.created_by,
        created_at=project.created_at,
        updated_at=project.updated_at,
        current_user_role=(current_membership.role if current_membership else None),
        members=[
            ProjectMemberView(
                id=member.id,
                user_id=member.user_id,
                email=member.user.email,
                name=member.user.name,
                role=member.role,
            )
            for member in project.members
        ],
    )


def load_project(db: DbSession, project_id: str) -> Project:
    project = db.scalar(
        select(Project)
        .where(Project.id == project_id)
        .options(selectinload(Project.members).selectinload(ProjectMember.user))
    )
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


def ensure_project_access(db: DbSession, project: Project, user: User) -> Optional[ProjectRole]:
    role = get_project_role(db, project.id, user)
    if role is None and user.system_role != SystemRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return role


@router.get(
    "",
    response_model=list[ProjectView],
    summary="查询项目列表",
    description="管理员可查看全部项目，普通用户只能查看自己参与的项目。",
)
def list_projects(db: DbSession, user: CurrentUser) -> list[ProjectView]:
    statement = select(Project).options(
        selectinload(Project.members).selectinload(ProjectMember.user)
    )
    if user.system_role != SystemRole.ADMIN:
        statement = statement.join(ProjectMember).where(ProjectMember.user_id == user.id)
    projects = db.scalars(statement.order_by(Project.updated_at.desc())).unique().all()
    return [to_project_view(project, user) for project in projects]


@router.post(
    "",
    response_model=ProjectView,
    status_code=status.HTTP_201_CREATED,
    summary="创建项目",
    description="创建项目，并自动将当前用户加入为项目所有者。",
)
def create_project(payload: ProjectCreate, db: DbSession, user: CurrentUser) -> ProjectView:
    project = Project(
        name=payload.name.strip(), description=payload.description.strip(), created_by=user.id
    )
    db.add(project)
    db.flush()
    db.add(ProjectMember(project_id=project.id, user_id=user.id, role=ProjectRole.OWNER))
    db.add(
        AuditLog(
            actor_id=user.id,
            action="project.create",
            resource_type="project",
            resource_id=project.id,
            detail=project.name,
        )
    )
    db.commit()
    return to_project_view(load_project(db, project.id), user)


@router.get(
    "/{project_id}",
    response_model=ProjectView,
    summary="获取项目详情",
    description="获取项目基本信息、当前用户角色和项目成员列表。",
)
def get_project(project_id: str, db: DbSession, user: CurrentUser) -> ProjectView:
    project = load_project(db, project_id)
    ensure_project_access(db, project, user)
    return to_project_view(project, user)


@router.post(
    "/{project_id}/members",
    response_model=ProjectMemberView,
    summary="添加或更新项目成员",
    description="按邮箱将用户加入项目，或更新该用户在项目中的角色。项目所有者或管理员可操作。",
)
def add_project_member(
    project_id: str, payload: ProjectMemberAdd, db: DbSession, user: CurrentUser
) -> ProjectMemberView:
    project = load_project(db, project_id)
    role = ensure_project_access(db, project, user)
    if user.system_role != SystemRole.ADMIN and role != ProjectRole.OWNER:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Owner access required")

    member_user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if member_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    existing = db.scalar(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id, ProjectMember.user_id == member_user.id
        )
    )
    if existing:
        existing.role = payload.role
        membership = existing
    else:
        membership = ProjectMember(project_id=project_id, user_id=member_user.id, role=payload.role)
        db.add(membership)
    db.add(
        AuditLog(
            actor_id=user.id,
            action="project.member.upsert",
            resource_type="project",
            resource_id=project_id,
            detail=f"{member_user.email}:{payload.role.value}",
        )
    )
    db.commit()
    db.refresh(membership)
    return ProjectMemberView(
        id=membership.id,
        user_id=member_user.id,
        email=member_user.email,
        name=member_user.name,
        role=membership.role,
    )
