from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome_screen, name='welcome'),
    path('analyze/<str:symbol>/', views.SentimentAnalysisView.as_view(), name='analyze'),
    path('health/', views.HealthCheckView.as_view(), name='health'),
    path('analyze/reddit/', views.RedditAnalysisView.as_view(), name='analyze_reddit'),
    path('analyze/news/', views.NewsAnalysisView.as_view(), name='analyze_news'),
    path('analyze/ai/', views.AIAnalysisView.as_view(), name='analyze_ai'),
]