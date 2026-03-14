from django.urls import path

from .views import (
	health,
	history,
	predict_brain,
	predict_health,
	predict_kidney,
	predict_liver,
	stats,
)

urlpatterns = [
	path('health/', health, name='health'),
	path('predict/brain/', predict_brain, name='predict-brain'),
	path('predict/kidney/', predict_kidney, name='predict-kidney'),
	path('predict/liver/', predict_liver, name='predict-liver'),
	path('predict/health/', predict_health, name='predict-health'),
	path('history/', history, name='history'),
	path('stats/', stats, name='stats'),
]
