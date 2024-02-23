from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth import authenticate, login
from .models import *

# Create your views here.


def index(request):
    docData = Doctor.objects.filter(loginid__is_active=0)
    print(docData)
    return render(request, "index.html", {"docData": docData})


def doctorReg(request):
    if request.POST:
        name = request.POST["name"]
        email = request.POST["email"]
        phone = request.POST["phone"]
        specialization = request.POST["specialization"]
        gender = request.POST["gender"]
        address = request.POST["address"]
        password = request.POST["password"]
        image = request.FILES["imgfile"]

        if not Login.objects.filter(username=email).exists():
            logQry = Login.objects.create_user(
                username=email,
                password=password,
                userType="Doctor",
                viewPass=password,
                is_active=0,
            )
            logQry.save()
            if logQry:
                regQry = Doctor.objects.create(
                    name=name,
                    email=email,
                    phone=phone,
                    specialization=specialization,
                    gender=gender,
                    image=image,
                    loginid=logQry,
                )
                regQry.save()
                if regQry:
                    return HttpResponse(
                        "<script>alert('Registration Successful');window.location.href='/login';</script>"
                    )
        else:
            return HttpResponse(
                "<script>alert('Email Already Exists');window.location.href='/doctorReg';</script>"
            )
    return render(request, "COMMON/doctorReg.html")


def patientReg(request):
    if request.POST:
        name = request.POST["name"]
        email = request.POST["email"]
        phone = request.POST["phone"]
        dob = request.POST["dob"]
        gender = request.POST["gender"]
        address = request.POST["address"]
        password = request.POST["password"]
        image = request.FILES["imgfile"]

        if not Login.objects.filter(username=email).exists():
            logQry = Login.objects.create_user(
                username=email,
                password=password,
                userType="Patient",
                viewPass=password,
            )
            logQry.save()

            if logQry:
                regQry = Patient.objects.create(
                    name=name,
                    email=email,
                    gender=gender,
                    phone=phone,
                    dob=dob,
                    image=image,
                    address=address,
                    loginid=logQry,
                )
                regQry.save()
                if regQry:
                    return HttpResponse(
                        "<script>alert('Registration Successful');window.location.href='/login';</script>"
                    )
        else:
            return HttpResponse(
                "<script>alert('Email Already Exists');window.location.href='/doctorReg';</script>"
            )
    return render(request, "COMMON/patientReg.html")


def signin(request):
    if request.POST:
        email = request.POST["email"]
        password = request.POST["password"]
        if Login.objects.filter(username=email, viewPass=password).exists():
            data = authenticate(username=email, password=password)
            if data is not None:
                login(request, data)
                if data.userType == "Patient":
                    id = data.id
                    request.session["uid"] = id
                    resp = '<script>alert("Login Success"); window.location.href = "/userhome";</script>'
                    return HttpResponse(resp)
                elif data.userType == "Doctor":
                    id = data.id
                    request.session["uid"] = id
                    resp = '<script>alert("Login Success"); window.location.href = "/doctorHome";</script>'
                    return HttpResponse(resp)
                elif data.userType == "Admin":
                    resp = '<script>alert("Login Success"); window.location.href = "/adminHome";</script>'
                    return HttpResponse(resp)
            else:
                return HttpResponse(
                    "<script>alert('You are not Approved');window.location.href='/login'</script>"
                )
        else:
            return HttpResponse(
                "<script>alert('Invalid Username/Password');window.location.href='/login'</script>"
            )
    return render(request, "COMMON/login.html")


def adminHome(request):
    return render(request, "ADMIN/adminHome.html")


def viewPatients(request):
    docData = Patient.objects.all()
    print(docData)
    return render(request, "ADMIN/viewPatients.html", {"docData": docData})


def userHome(request):
    return render(request, "USER/userHome.html")


# from PIL import Image
# from io import BytesIO
# from tensorflow.keras.models import load_model
# import numpy as np


# def preprocess_image(imagefile, target_size=(500, 500)):
#     img = Image.open(imagefile)
#     img = img.resize(target_size, Image.Resampling.LANCZOS).convert(
#         "L"
#     )  # Convert to grayscale
#     img_array = (
#         np.array(img).reshape(target_size + (1,)) / 255.0
#     )  # Reshape to match model input shape and normalize
#     return img_array


# def predict_image(model, image_array):
#     prediction = model.predict(np.expand_dims(image_array, axis=0))[0]
#     result = "Positive" if prediction >= 0.5 else "Negative"
#     return result, prediction


# ###############   DETECTION  ##################


# def predict(request):
#     uid = request.session["uid"]
#     user = Patient.objects.get(loginid=uid)
#     print(user)
#     if request.method == "POST":
#         imagefile = request.FILES.get("imagefile")
#         if imagefile:
#             uploaded_image = preprocess_image(imagefile)

#             # Load pre-trained model
#             model_path = "./models/pneu_cnn_model.h5"
#             model = load_model(model_path)

#             # Predict using the pre-trained model
#             predicted_class, confidence = predict_image(model, uploaded_image)

#             print("PREDICT_CLASS", predicted_class)
#             print("CONFIDENCE", confidence)

#             p = Prediction.objects.create(
#                 image=imagefile,
#                 result=predicted_class,
#                 user=user,
#             )
#             p.save()
#             image_path = p.image
#             return render(
#                 request,
#                 "USER/predict.html",
#                 {
#                     "prediction": predicted_class,
#                     "confidence": confidence,
#                     "image_path": image_path,
#                 },
#             )

#     return render(request, "USER/predict.html")


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from PIL import Image
import os
import pandas as pd


def preprocess_image(imagefile, target_size=(500, 500)):
    img = Image.open(imagefile)
    img = img.resize(target_size, Image.Resampling.LANCZOS).convert(
        "L"
    )  # Convert to grayscale
    img_array = np.array(img).reshape(-1, 1)  # Reshape to 2D array
    return img_array


def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def predict_image_knn(knn, image_array):
    # Flatten the image array into a 1D array
    flattened_image = image_array.flatten()

    # Reshape the flattened image into a 2D array with a single row
    reshaped_image = flattened_image.reshape(1, -1)

    # Predict the class label using the KNeighborsClassifier
    prediction = knn.predict(reshaped_image)[0]

    return prediction


def predict(request):
    uid = request.session["uid"]
    user = Patient.objects.get(loginid=uid)
    print(user)
    if request.method == "POST":
        imagefile = request.FILES.get("imagefile")
        if imagefile:
            # Define paths to your dataset and labels
            dataset_dir = r"D:\@MIDHUN\PROJECTS\PYTHON\TEST\LungCancer\Lungcancer\DataSet\images\images"  # Directory containing your images
            label_file = r"D:\@MIDHUN\PROJECTS\PYTHON\TEST\LungCancer\Lungcancer\DataSet\jsrt_metadata.csv"  # File containing image labels

            # Read labels from CSV file
            labels_df = pd.read_csv(label_file)

            # Load labels
            with open(label_file, "r") as file:
                labels = file.readlines()
            labels = [label.strip() for label in labels]  # Remove newline characters

            # Initialize lists to store feature vectors (X_train) and labels (y_train)
            X_train = np.zeros(shape=(247, 250000))
            y_train = []

            # Inside the loop for reading the CSV file and processing images
            for index, row in labels_df.iterrows():
                image_filename = row["study_id"]
                label = row["diagnosis"]

                image_path = os.path.join(dataset_dir, image_filename)

                img_features = preprocess_image(image_path)
                X_train[index] = (
                    img_features.flatten()
                )  # Flatten and append the feature vector to X_train
                y_train.append(label)

            # Convert lists to numpy arrays
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            uploaded_image = preprocess_image(imagefile)

            print("SHAPE ", X_train.shape)
            # Flatten each image into a 1D array
            X_train_flattened = X_train.reshape(X_train.shape[0], -1)

            print(
                "Flattened shape of X_train:", X_train_flattened.shape
            )  # Check the flattened
            # Load data
            # Assuming X_train and y_train are loaded from somewhere
            # X_train: Feature vectors of training images
            # y_train: Corresponding labels

            # Train KNN model
            knn_model = train_knn(X_train, y_train)
            print("KKKKKKKKKKK")
            print(knn_model)
            print(uploaded_image)
            # Predict using KNN model
            predicted_class = predict_image_knn(knn_model, uploaded_image)

            print("PREDICT_CLASS", predicted_class)

            if predicted_class == "nan":
                predicted_class = "Negative"
            else:
                predicted_class = "Positive"

            p = Prediction.objects.create(
                image=imagefile,
                result=predicted_class,
                user=user,
            )
            p.save()
            image_path = p.image
            return render(
                request,
                "USER/predict.html",
                {
                    "prediction": predicted_class,
                    "image_path": image_path,
                },
            )

    return render(request, "USER/predict.html")
