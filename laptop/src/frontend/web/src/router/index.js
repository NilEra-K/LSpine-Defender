import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import AboutView from '@/views/AboutView.vue'
import FeatureOneView from '@/views/FeatureOneView.vue'
import ImagingPredictView from '@/views/ImagingPredictView.vue'
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
    path: '/imaging-predict',
    name: 'ImagingPredict',
    component: ImagingPredictView
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