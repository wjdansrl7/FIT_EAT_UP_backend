# Generated by Django 3.2.15 on 2022-09-06 07:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Place',
            fields=[
                ('address_name', models.CharField(max_length=50)),
                ('category_group_name', models.CharField(max_length=50)),
                ('category_name', models.CharField(max_length=50)),
                ('id', models.CharField(max_length=50, primary_key=True, serialize=False)),
                ('phone', models.CharField(blank=True, max_length=13)),
                ('place_name', models.CharField(max_length=50)),
                ('place_url', models.URLField(blank=True)),
                ('road_address_name', models.CharField(max_length=50)),
                ('x', models.DecimalField(decimal_places=10, max_digits=13)),
                ('y', models.DecimalField(decimal_places=10, max_digits=13)),
                ('image', models.ImageField(blank=True, upload_to='accounts/image/%Y/%m/%d')),
            ],
        ),
        migrations.AddField(
            model_name='user',
            name='like_places',
            field=models.ManyToManyField(blank=True, related_name='like_places_set', to='accounts.Place'),
        ),
        migrations.AddField(
            model_name='user',
            name='visit_places',
            field=models.ManyToManyField(blank=True, related_name='visit_places_set', to='accounts.Place'),
        ),
    ]
