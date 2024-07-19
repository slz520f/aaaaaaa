from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16 # type: ignore
from tensorflow.keras.models import save_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array# type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions # type: ignore
from io import BytesIO
import os
model = VGG16(weights='imagenet')
save_model(model, 'vgg16.h5')



def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        # POSTリクエストによるアクセス時の処理を記述
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            print(model_path )
            model = load_model(model_path)

            # 4章で、判定結果のロジックを追加
            predictions = model.predict(img_array)
            raw_predictions = model.predict(preprocessed_img)
            top_predictions = decode_predictions(raw_predictions, top=5)[0]


        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form, 'predictions': top_predictions})


