from django.db import models

# Create your models here.
class TaggedImage(models.Model):
  image = models.ImageField(upload_to="images/", height_field=None, width_field=None, max_length=None)
  uploaded_at = models.DateTimeField(auto_now_add=True)

class APIEndpoint(models.Model):
  url = models.TextField(default="")
  name = models.TextField(default="")
