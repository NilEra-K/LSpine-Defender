import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import AboutView from '@/views/AboutView.vue'
import FeatureOneView from '@/views/FeatureOneView.vue'
import StageOneView from '@/views/StageOneView.vue'
import SeverityPredictView from '@/views/SeverityPredictView.vue'

const routes = [
  {
    path: '/',
    name: 'HomePage',
    component: HomeView
  },
  {
    path: '/home',
    name: 'Home',
    component: HomeView
  },
  {
    path: '/about',
    name: 'About',
    component: AboutView
  },
  {
    path: '/feature-one',
    name: 'FeatureOne',
    component: FeatureOneView
  },
  {
    path: '/stage-one',
    name: 'StageOne',
    component: StageOneView
  },
  {
    path: '/severity-predict',
    name: 'SeverityPredict',
    component: SeverityPredictView
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})


export default router