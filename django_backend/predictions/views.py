from __future__ import annotations

from django.conf import settings
from django.db.models import Avg, Count
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

import requests

from .models import PredictionRecord
from .serializers import PredictionRecordSerializer


def _safe_json(resp: requests.Response) -> dict:
	try:
		return resp.json()
	except Exception:
		return {"detail": resp.text[:500] or "Invalid upstream response"}


def _build_record(request, payload: dict, input_data: dict, image_file=None) -> PredictionRecord:
	record = PredictionRecord.objects.create(
		user=request.user if getattr(request.user, "is_authenticated", False) else None,
		organ=payload.get("organ", "Unknown"),
		model_used=payload.get("model", "Unknown"),
		prediction=payload.get("prediction", "Unknown"),
		confidence=float(payload.get("confidence", 0.0) or 0.0),
		risk_level=payload.get("risk_level", "Low"),
		raw_response=payload,
		input_data=input_data,
		image=image_file,
	)
	return record


def _forward_json(path: str, json_body: dict) -> tuple[int, dict]:
	url = f"{settings.FASTAPI_BASE_URL}{path}"
	try:
		resp = requests.post(url, json=json_body, timeout=60)
	except requests.RequestException as exc:
		return status.HTTP_503_SERVICE_UNAVAILABLE, {"detail": f"FastAPI service unreachable: {exc}"}
	return resp.status_code, _safe_json(resp)


def _forward_file(path: str, file_obj) -> tuple[int, dict]:
	url = f"{settings.FASTAPI_BASE_URL}{path}"
	file_tuple = (file_obj.name, file_obj.read(), file_obj.content_type or "application/octet-stream")
	try:
		resp = requests.post(url, files={"file": file_tuple}, timeout=60)
	except requests.RequestException as exc:
		return status.HTTP_503_SERVICE_UNAVAILABLE, {"detail": f"FastAPI service unreachable: {exc}"}
	return resp.status_code, _safe_json(resp)


@api_view(["GET"])
@permission_classes([AllowAny])
def health(request):
	return Response({"status": "ok", "service": "django-gateway"})


@api_view(["POST"])
@permission_classes([AllowAny])
@parser_classes([MultiPartParser, FormParser])
def predict_brain(request):
	image = request.FILES.get("file")
	if not image:
		return Response({"detail": "Missing file in form-data with key 'file'"}, status=400)

	status_code, payload = _forward_file("/predict/brain", image)
	if status_code >= 400:
		return Response(payload, status=status_code)

	record = _build_record(request, payload, {"filename": image.name}, image_file=image)
	payload["record_id"] = record.id
	payload["saved_at"] = record.created_at.isoformat()
	return Response(payload)


@api_view(["POST"])
@permission_classes([AllowAny])
@parser_classes([JSONParser])
def predict_kidney(request):
	body = request.data if isinstance(request.data, dict) else {}
	status_code, payload = _forward_json("/predict/kidney", body)
	if status_code >= 400:
		return Response(payload, status=status_code)

	record = _build_record(request, payload, body)
	payload["record_id"] = record.id
	payload["saved_at"] = record.created_at.isoformat()
	return Response(payload)


@api_view(["POST"])
@permission_classes([AllowAny])
@parser_classes([JSONParser])
def predict_liver(request):
	body = request.data if isinstance(request.data, dict) else {}
	status_code, payload = _forward_json("/predict/liver", body)
	if status_code >= 400:
		return Response(payload, status=status_code)

	record = _build_record(request, payload, body)
	payload["record_id"] = record.id
	payload["saved_at"] = record.created_at.isoformat()
	return Response(payload)


@api_view(["POST"])
@permission_classes([AllowAny])
@parser_classes([JSONParser])
def predict_health(request):
	body = request.data if isinstance(request.data, dict) else {}
	status_code, payload = _forward_json("/predict/health", body)
	if status_code >= 400:
		return Response(payload, status=status_code)

	record = _build_record(request, payload, body)
	payload["record_id"] = record.id
	payload["saved_at"] = record.created_at.isoformat()
	return Response(payload)


@api_view(["GET"])
@permission_classes([AllowAny])
def history(request):
	records = PredictionRecord.objects.all()[:200]
	data = PredictionRecordSerializer(records, many=True).data
	return Response(data)


@api_view(["GET"])
@permission_classes([AllowAny])
def stats(request):
	qs = PredictionRecord.objects.all()
	total = qs.count()
	by_organ = {row["organ"]: row["count"] for row in qs.values("organ").annotate(count=Count("id"))}
	avg_confidence = float((qs.aggregate(v=Avg("confidence")).get("v") or 0.0))
	high_risk = qs.filter(risk_level="High").count()

	return Response(
		{
			"total": total,
			"by_organ": by_organ,
			"avg_confidence": round(avg_confidence, 2),
			"high_risk": high_risk,
		}
	)
