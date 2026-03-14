from rest_framework import serializers

from .models import PredictionRecord


class PredictionRecordSerializer(serializers.ModelSerializer):
	class Meta:
		model = PredictionRecord
		fields = [
			"id",
			"user",
			"organ",
			"model_used",
			"prediction",
			"confidence",
			"risk_level",
			"raw_response",
			"input_data",
			"image",
			"created_at",
		]
		read_only_fields = ["id", "created_at", "user"]


class StatsSerializer(serializers.Serializer):
	total = serializers.IntegerField()
	by_organ = serializers.DictField(child=serializers.IntegerField())
	avg_confidence = serializers.FloatField()
	high_risk = serializers.IntegerField()
