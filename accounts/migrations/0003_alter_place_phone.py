# Generated by Django 3.2.15 on 2022-10-09 06:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0002_auto_20221005_0436'),
    ]

    operations = [
        migrations.AlterField(
            model_name='place',
            name='phone',
            field=models.CharField(blank=True, max_length=18),
        ),
    ]
