from django.shortcuts import render
import joblib
import numpy as np
import re

SVM_clf = joblib.load(r'E:\Project\Major\Major_Project_1\mlapp\model\svm_language_identifier.pkl')
vectorizer = joblib.load(r'E:\Project\Major\Major_Project_1\mlapp\model\tfidf_vectorizer.pkl')

def home(request):
    if request.method == 'POST':
        inp = request.POST.get('text_input')
        text = inp.lower()

        text = re.sub(r'[^\w\s]', '', text)

# Step 3: Transform the input using the same vectorizer that was used for training
        out = vectorizer.transform([text])  # Note that we're using `transform`, not `fit_transform`
        out = out.toarray()
# Step 4: Predict the language using the trained SVM model
        predicted_label = SVM_clf.predict(out)[0]
        language_map = {
            0: 'English',
            1: 'French',
            2: 'German',
            3: 'Spanish'
        }
        
        # Get the predicted language name based on the model output
        predicted_language = language_map.get(predicted_label, "Unknown")
        

        if len(inp.split('.')) > 1:  # Check if there are multiple sentences
            # Take the first sentence
            input_text_to_show = inp.split('.')[0] + '.'  # Keep the sentence structure
        else:
            input_text_to_show = inp  # Use the original input if it's short

        # Render the result on the home page
        return render(request, 'home_page.html', {
            'predicted_language': predicted_language,
            'input_text': input_text_to_show  # Pass the trimmed input text
        })
    return render(request, 'home_page.html')

def about(request):
    return render(request, 'about_us.html')

def started(request):
    return render(request, 'getting_started.html')

def contact(request):
    return render(request, 'contact.html')