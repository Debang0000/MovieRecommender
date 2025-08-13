# Movie Recommender Web Application

This folder contains the frontend and static assets for the Movie Recommender web application.

## Structure

```
web/
├── templates/          # HTML templates
│   └── index.html     # Main page template
├── static/            # Static assets
│   ├── css/          # Stylesheets
│   │   └── style.css # Main stylesheet
│   └── js/           # JavaScript files
│       └── script.js # Main JavaScript file
└── README.md         # This file
```

## Features

- **Modern UI**: Dark theme with gradient accents
- **Movie Posters**: Displays TMDB movie posters when available
- **Autocomplete**: Search suggestions as you type
- **Responsive Design**: Works on desktop and mobile devices
- **Loading States**: Visual feedback during API calls
- **Error Handling**: User-friendly error messages

## Technologies Used

- **HTML5**: Semantic markup
- **CSS3**: Modern styling with CSS variables
- **JavaScript**: Vanilla JS for interactivity
- **Font Awesome**: Icons
- **Google Fonts**: Inter font family

## Running the Application

From the project root directory, run:

```bash
python run.py
```

Then open your browser to `http://127.0.0.1:5000` 