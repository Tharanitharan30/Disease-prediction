from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="PredictionRecord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("organ", models.CharField(max_length=64)),
                ("model_used", models.CharField(max_length=255)),
                ("prediction", models.CharField(max_length=255)),
                ("confidence", models.FloatField()),
                ("risk_level", models.CharField(max_length=32)),
                ("raw_response", models.JSONField(default=dict)),
                ("input_data", models.JSONField(default=dict)),
                ("image", models.ImageField(blank=True, null=True, upload_to="predictions/")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="prediction_records",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={"ordering": ["-created_at"]},
        )
    ]