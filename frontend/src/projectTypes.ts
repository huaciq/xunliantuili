import type { ProjectType } from './types'

export const projectTypeLabels: Record<ProjectType, string> = {
  general: '通用训练',
  vibration_feeder_point: '振动盘点位定位',
}

export const projectTypeOptions = Object.entries(projectTypeLabels).map(([value, label]) => ({
  value, label,
}))
