# Melodify 	

**Melodify** is a web application which is a facial expression recognition-based music suggestion website that cheers up users and saves time while searching for a song that matches their mood.
1. It recognizes facial expression based on the 7 categories i.e., angry, sad, fear, happy, disgust, surprise and neutral.
2. Based on the emotion it shows playlist to the user.
3. When user clicks on songs it redirects them to Spotify website.
4. Enjoy songs !

## How we built it
Python is the programming language used to create the emotion recognition model and deploy it on the web application using flask. CV2, TensorFlow, NumPy, matplotlib, and other libraries are also utilized. The model is build using the transfer learning approach for which MobileNet model is used. The FER-2013 dataset, which comprises around 35000 photos, was utilized for model training and validation. This model is deployed on a website created with HTML and CSS using the flask framework. Based on the seven emotions, a new dataset of movies and music was constructed. The data from movies and songs was utilized to create the various templates that correlate to various emotions. 
## Challenges we ran into
It was a challenging task to get the website to access 7 song templates from the songs button based on the emotion recognized. Dynamic links were used as a means to access the templates for the songs corresponding to the output of the model.

## What's next for **Moodify**
The next step is to improve the model's accuracy and enhance the UI/UX of the web app.
