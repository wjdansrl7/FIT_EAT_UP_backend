# Generated by Django 3.2.15 on 2022-11-01 08:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0003_alter_place_phone'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='avatar',
            field=models.ImageField(blank=True, help_text='48px * 48px 크기의 png/jpg 파일을 업로드해주세요.', null=True, upload_to='accounts/avatar/%Y/%m/%d'),
        ),
    ]