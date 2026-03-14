from django.conf import settings
from django.db import models


class PredictionRecord(models.Model):
	user = models.ForeignKey(
		settings.AUTH_USER_MODEL,
		on_delete=models.SET_NULL,
		null=True,
		blank=True,
		related_name="prediction_records",
	)
	organ = models.CharField(max_length=64)
	model_used = models.CharField(max_length=255)
	prediction = models.CharField(max_length=255)
	confidence = models.FloatField()
	risk_level = models.CharField(max_length=32)
	raw_response = models.JSONField(default=dict)
	input_data = models.JSONField(default=dict)
	image = models.ImageField(upload_to="predictions/", null=True, blank=True)
	created_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ["-created_at"]

	def __str__(self) -> str:
		return f"{self.organ} | {self.prediction} | {self.confidence:.2f}%"
