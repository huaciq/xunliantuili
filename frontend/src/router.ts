import { createRouter, createWebHistory } from 'vue-router'
import AppLayout from './components/AppLayout.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', component: () => import('./views/LoginView.vue') },
    {
      path: '/', component: AppLayout, meta: { requiresAuth: true },
      children: [
        { path: '', redirect: '/projects' },
        { path: 'projects', component: () => import('./views/ProjectsView.vue') },
        { path: 'projects/:id', component: () => import('./views/ProjectDetailView.vue') },
        {
          path: 'projects/:projectId/datasets',
          component: () => import('./views/ProjectResourcesView.vue'),
          props: { kind: 'datasets' },
        },
        {
          path: 'projects/:projectId/code',
          component: () => import('./views/ProjectResourcesView.vue'),
          props: { kind: 'code' },
        },
        {
          path: 'projects/:projectId/runs',
          component: () => import('./views/TrainingRunsView.vue'),
        },
        {
          path: 'projects/:projectId/models',
          component: () => import('./views/ModelsView.vue'),
        },
        { path: 'runs/:runId', component: () => import('./views/TrainingRunDetailView.vue') },
        { path: 'users', component: () => import('./views/UsersView.vue') },
      ],
    },
  ],
})

router.beforeEach((to) => {
  const authenticated = Boolean(localStorage.getItem('access_token'))
  if (to.meta.requiresAuth && !authenticated) return '/login'
  if (to.path === '/login' && authenticated) return '/projects'
})

export default router
