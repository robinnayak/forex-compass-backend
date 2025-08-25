from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome_screen, name='welcome'),
    path('analyze/', views.AnalyzeMarketData.as_view(), name='analyze'),
]
