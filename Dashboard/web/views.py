from django.shortcuts import render
from . import models
from django import forms

class ImageForm(forms.ModelForm):
    class Meta:
        model = models.TaggedImage
        fields = ('image', )

# Create your views here.
def dashboard(request):
  data = {"endpoints": models.APIEndpoint.objects.all()}
  if(request.method == "POST") and 'image' in request.FILES:
    form = ImageForm(request.POST, request.FILES)
    if form.is_valid():
      model = form.save()
      data['image'] = model.image.url
      data['image_id'] = model.id

  return render(request, 'web/dashboard.html', data)