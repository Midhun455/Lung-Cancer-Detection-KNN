from django.db import models
from django.contrib.auth.models import AbstractUser


# Create your models here.


class Login(AbstractUser):
    userType = models.CharField(max_length=100)
    viewPass = models.CharField(max_length=100, null=True)

    def __str__(self):
        return self.username


class Doctor(models.Model):
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    phone = models.CharField(max_length=100)
    specialization = models.CharField(max_length=100)
    gender = models.CharField(max_length=100)
    image = models.FileField(max_length=100)
    address = models.CharField(max_length=300, null=True)
    loginid = models.ForeignKey(Login, on_delete=models.CASCADE)

    def __str__(self):
        return self.name


class Patient(models.Model):
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    phone = models.CharField(max_length=100)
    gender = models.CharField(max_length=100)
    dob = models.DateField()
    image = models.FileField(max_length=100)
    address = models.CharField(max_length=300)
    loginid = models.ForeignKey(Login, on_delete=models.CASCADE)


class Prediction(models.Model):
    image = models.ImageField(null=True)
    result = models.CharField(max_length=255,null=True)
    user=models.ForeignKey(Patient,on_delete=models.CASCADE,null=True)