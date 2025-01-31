import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import AboutView from '@/views/AboutView.vue'
import ImagingPredictView from '@/views/ImagingPredictView.vue'
import SeverityPredictView from '@/views/SeverityPredictView.vue'
import ApiDocView from '@/views/ApiDocView.vue'
import DataEdaView from '@/views/DataEdaView.vue'

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
    path: '/data-eda',
    name: 'DataEda',
    component: DataEdaView
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
  },
  {
    path: '/api-doc',
    name: 'ApiDoc',
    component: ApiDocView
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})


export default router